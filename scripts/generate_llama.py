"""llama.cpp generate backend. MCP owns llama-server (not Transformers).

Starts llama-server on first call, kills it on process exit. Stdio stays
clean so MCP can use it. Logs go to outputs/llama-server.log.

  ML_ROUTER_GGUF
  ML_ROUTER_LLAMA_SERVER
  ML_ROUTER_LLAMA_HOST   (default 127.0.0.1)
  ML_ROUTER_LLAMA_PORT   (default 8765)
  ML_ROUTER_N_GPU_LAYERS (default 99)
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = ROOT / "outputs" / "qwen25-3b-q4_k_m.gguf"
DEFAULT_SERVER = Path(
    r"E:\Github\llama.cpp\build_vs2022\bin\Release\llama-server.exe"
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_NGL = 99
LOG = ROOT / "outputs" / "llama-server.log"

_proc: subprocess.Popen | None = None
_owned = False


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


def _conf() -> dict:
    return {
        "gguf": _env_path("ML_ROUTER_GGUF", DEFAULT_GGUF),
        "server": _env_path("ML_ROUTER_LLAMA_SERVER", DEFAULT_SERVER),
        "host": os.environ.get("ML_ROUTER_LLAMA_HOST", DEFAULT_HOST),
        "port": int(os.environ.get("ML_ROUTER_LLAMA_PORT", DEFAULT_PORT)),
        "ngl": int(os.environ.get("ML_ROUTER_N_GPU_LAYERS", DEFAULT_NGL)),
    }


def _base_url() -> str:
    conf = _conf()
    return f"http://{conf['host']}:{conf['port']}"


def _healthy(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(_base_url() + "/health", timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _stop() -> None:
    global _proc, _owned
    if not _owned or _proc is None:
        return
    if _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None
    _owned = False


def ensure_server(timeout: float = 120.0) -> None:
    global _proc, _owned
    if _healthy():
        return

    conf = _conf()
    if not conf["gguf"].is_file():
        raise SystemExit(f"missing GGUF {conf['gguf']}")
    if not conf["server"].is_file():
        raise SystemExit(f"missing llama-server {conf['server']}")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = LOG.open("ab")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cmd = [
        str(conf["server"]),
        "-m",
        str(conf["gguf"]),
        "--host",
        conf["host"],
        "--port",
        str(conf["port"]),
        "-ngl",
        str(conf["ngl"]),
        "-c",
        "2048",
        "--alias",
        "mcp-slm",
        "--jinja",
    ]
    print(f"starting llama-server  {_base_url()}  {conf['gguf'].name}", file=sys.stderr, flush=True)
    _proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    _owned = True
    atexit.register(_stop)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _proc.poll() is not None:
            raise SystemExit(
                f"llama-server exited {_proc.returncode}. See {LOG}"
            )
        if _healthy():
            print("llama-server ready", file=sys.stderr, flush=True)
            return
        time.sleep(0.25)
    raise SystemExit(f"llama-server did not become healthy in {timeout:.0f}s. See {LOG}")


def generate(system: str, user: str, max_new_tokens: int) -> tuple[str, int]:
    ensure_server()
    body = {
        "model": "mcp-slm",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }
    req = urllib.request.Request(
        _base_url() + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llama-server request failed: {exc}") from exc

    choices = payload.get("choices") or []
    msg = (choices[0].get("message") if choices else {}) or {}
    raw = str(msg.get("content") or "").strip()
    usage = payload.get("usage") or {}
    n_in = int(usage.get("prompt_tokens") or 0)
    return raw, n_in
