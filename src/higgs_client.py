# Dipendenze: solo libreria standard (+ PowerShell per misurare la memoria video). Optional helper for programs that want to start the server on demand.
# Accende il server Higgs quando serve: prima controlla che la scheda abbia memoria libera (se non misurabile = occupata,
# non parte), poi lo avvia nascosto e aspetta che risponda. Il server si spegne da solo quando resta inattivo.
from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

HIGGS_DIR = Path(__file__).resolve().parent
LAUNCHER = HIGGS_DIR.parent / "scripts" / "start_server.bat"
DEFAULT_URL = "http://127.0.0.1:8090"
REQUIRED_FREE_GB = 16.0  # picco misurato: ~13 GB con 8 paragrafi; caso peggiore (12 lunghi) ~14,5 + margine
COMFY_QUEUE_URL = "http://127.0.0.1:8188/queue"  # optional: if ComfyUI runs on this port its queue is checked


class ServerUnavailable(RuntimeError):
    pass


def health(base_url: str = DEFAULT_URL, timeout: float = 10) -> dict | None:
    """The server's /health answer, or None when nobody answers."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=timeout) as response:
            if response.status != 200:
                return None
            body = response.read()
    except (OSError, ValueError):
        return None
    try:
        answer = json.loads(body or b"{}")
    except ValueError:
        return {}  # risponde ma senza dettagli (server vecchio): conta come pronto
    return answer if isinstance(answer, dict) else {}


def is_running(base_url: str = DEFAULT_URL) -> bool:
    """True only when the model is loaded and requests are accepted (older servers without 'ready' count as ready)."""
    answer = health(base_url)
    return bool(answer) and answer.get("ready", True)


def port_in_use(base_url: str = DEFAULT_URL) -> bool:
    """True when something already listens on the server port (a server loading or busy): never start a second one."""
    parsed = urllib.parse.urlparse(base_url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=2):
            return True
    except OSError:
        return False


def _powershell(command: str) -> str:
    completed = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True,
                               timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return completed.stdout.strip()


def free_video_memory_gb() -> float | None:
    """Free dedicated video memory in GB, or None when it cannot be measured."""
    try:
        used = _powershell("((Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples"
                           " | Measure-Object CookedValue -Sum).Sum")
        total = _powershell("(Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\"
                            "{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue"
                            " | ForEach-Object { $_.'HardwareInformation.qwMemorySize' } | Measure-Object -Maximum).Maximum")
        return (float(total) - float(used)) / 1024 ** 3
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def comfyui_is_generating() -> bool:
    try:
        with urllib.request.urlopen(COMFY_QUEUE_URL, timeout=3) as response:
            queue = json.loads(response.read())
        return bool(queue.get("queue_running") or queue.get("queue_pending"))
    except (OSError, ValueError):
        return False  # ComfyUI spento: non sta generando


def ensure_server(base_url: str = DEFAULT_URL, wait_seconds: int = 240, required_free_gb: float = REQUIRED_FREE_GB) -> None:
    """Starts the server if nobody listens on its port, then waits until it is ready. A server already loading or
    busy is waited for, never duplicated. Raises ServerUnavailable with a plain-language reason."""
    if is_running(base_url):
        return
    if port_in_use(base_url):
        return _wait_ready(base_url, wait_seconds)
    if comfyui_is_generating():
        raise ServerUnavailable("ComfyUI sta generando: la scheda video è occupata, riprova quando ha finito.")
    free_gb = free_video_memory_gb()
    if free_gb is None:
        raise ServerUnavailable("Non riesco a misurare la memoria della scheda video: per sicurezza non avvio la voce.")
    if free_gb < required_free_gb:
        raise ServerUnavailable(f"Memoria della scheda video insufficiente: liberi {free_gb:.1f} GB, "
                                f"ne servono {required_free_gb:.0f}. Chiudi altri programmi che usano la scheda.")
    subprocess.Popen(["cmd", "/c", str(LAUNCHER)], cwd=str(HIGGS_DIR),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _wait_ready(base_url, wait_seconds)


def _wait_ready(base_url: str, wait_seconds: float) -> None:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if is_running(base_url):
            return
        time.sleep(3)
    raise ServerUnavailable(f"Il server della voce non è pronto entro {wait_seconds:.0f} secondi.")
