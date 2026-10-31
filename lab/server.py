"""Запуск и остановка vLLM-сервера в фоне — чтобы в ноутбуке Kaggle не держать ячейку занятой.

    python -m lab.server start Qwen/Qwen2.5-1.5B-Instruct -- --enable-prefix-caching
    python -m lab.server log
    python -m lab.server stop

Всё после «--» передаётся в `vllm serve` как есть.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from lab.common import LOGS_DIR, wait_for_server

PID_FILE = LOGS_DIR / "vllm.pid"
LOG_FILE = LOGS_DIR / "vllm.log"


def tail(lines: int = 40) -> str:
    if not LOG_FILE.exists():
        return ""
    return "\n".join(LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def start(
    model: str,
    extra_args: list[str],
    port: int = 8000,
    gpus: str | None = "0",
    max_model_len: int = 4096,
    gpu_util: float = 0.90,
    timeout: int = 1200,
) -> None:
    if PID_FILE.exists():
        print("найден запущенный сервер — останавливаю")
        stop()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # T4 (Turing) не умеет bf16, поэтому всегда fp16
    cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--dtype",
        "half",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_util),
        *extra_args,
    ]
    env = os.environ.copy()
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    print("$ " + " ".join(cmd))
    log = LOG_FILE.open("w", encoding="utf-8")  # остаётся открытым у дочернего процесса
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    PID_FILE.write_text(str(proc.pid))
    try:
        wait_for_server(f"http://localhost:{port}/v1", timeout=timeout, proc=proc)
    except (RuntimeError, TimeoutError):
        print(tail())
        stop()
        raise
    print(f"vLLM готов: http://localhost:{port}/v1 (лог: {LOG_FILE})")


def stop(timeout: int = 60) -> None:
    if not PID_FILE.exists():
        print("сервер не запущен")
        return
    pid = int(PID_FILE.read_text())
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return
    deadline = time.monotonic() + timeout
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(1)
    if _alive(pid):
        os.killpg(pid, signal.SIGKILL)
    PID_FILE.unlink(missing_ok=True)
    time.sleep(5)  # драйверу нужно время, чтобы освободить память GPU
    print("сервер остановлен")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(description="vLLM-сервер в фоне")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_start = sub.add_parser("start")
    p_start.add_argument("model")
    p_start.add_argument("--port", type=int, default=8000)
    p_start.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES, для двух T4 — «0,1»")
    p_start.add_argument("--max-model-len", type=int, default=4096)
    p_start.add_argument("--gpu-util", type=float, default=0.90)
    p_start.add_argument("--timeout", type=int, default=1200)
    sub.add_parser("stop")
    sub.add_parser("log")
    args = parser.parse_args(argv)

    if args.cmd == "start":
        start(args.model, extra, args.port, args.gpus, args.max_model_len, args.gpu_util, args.timeout)
    elif args.cmd == "stop":
        stop()
    else:
        print(tail(80))


if __name__ == "__main__":
    main()
