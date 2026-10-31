"""Общие утилиты: пути, JSON, окружение, клиент OpenAI-совместимого API."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
import urllib.request
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"

_PACKAGES = (
    "vllm",
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "sentence-transformers",
    "faiss-cpu",
    "datasets",
    "openai",
)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


# --- файлы ------------------------------------------------------------------


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(rows: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- окружение --------------------------------------------------------------


def gpu_info() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def env_info() -> dict:
    """Версии пакетов и GPU — сохраняются в каждый файл результатов."""
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "gpus": gpu_info(),
        "packages": versions,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --- HTTP -------------------------------------------------------------------


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_get_json(url: str, timeout: float = 5.0) -> Any:
    with _no_proxy_opener().open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout: float = 900, poll: float = 3.0, proc=None) -> None:
    """Ждёт, пока /v1/models начнёт отвечать. Если процесс сервера умер — падает сразу."""
    url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM завершился с кодом {proc.returncode}, подробности в логе")
        try:
            http_get_json(url)
            return
        except (OSError, ValueError):
            time.sleep(poll)
    raise TimeoutError(f"сервер {base_url} не поднялся за {timeout:.0f} с")


def make_async_client(base_url: str, api_key: str = "EMPTY", timeout: float = 600.0, max_retries: int = 0):
    """AsyncOpenAI для локального vLLM.

    max_retries=0 по умолчанию: в бенчмарке повтор запроса исказил бы задержки.
    Для локального адреса переменные прокси игнорируются.
    """
    import httpx
    from openai import AsyncOpenAI

    local = urlparse(base_url).hostname in _LOCAL_HOSTS
    http_client = httpx.AsyncClient(
        trust_env=not local,
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=512),
    )
    return AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=max_retries)


def json_schema_format(name: str, schema: dict) -> dict:
    """response_format для structured output (OpenAI API, поддерживается vLLM)."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def parse_json_object(text: str | None) -> dict | None:
    """Достаёт JSON-объект из ответа модели: чистый JSON, блок ```json``` или первый {...}."""
    if not text:
        return None
    candidates = [text.strip(), *(m.strip() for m in _FENCE.findall(text))]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# --- вывод ------------------------------------------------------------------


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    cells = [[str(c) for c in row] for row in rows]
    widths = [max([len(h), *(len(r[i]) for r in cells)]) for i, h in enumerate(headers)]
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for row in cells:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
