"""Фейковый OpenAI-совместимый сервер: проверяем клиентскую часть без GPU и без vLLM."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # без keep-alive: стрим заканчивается закрытием соединения

    def log_message(self, *args):  # не засоряем вывод pytest
        pass

    def do_GET(self):
        self._send_json(200, {"object": "list", "data": [{"id": "fake", "object": "model"}], "version": "fake-0"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        reply = self.server.responder(body)
        if reply is None:
            self._send_json(500, {"error": {"message": "boom", "type": "server_error"}})
        elif body.get("stream"):
            self._stream(body, reply)
        else:
            self._send_json(200, self._completion(body, reply))

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _completion(body: dict, message: dict) -> dict:
        return {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _stream(self, body: dict, n_tokens: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": "cmpl-1", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
        n = min(n_tokens, body.get("max_tokens") or n_tokens)
        for i in range(n):
            time.sleep(self.server.token_delay)
            chunk = {**base, "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        usage = {**base, "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": n, "total_tokens": 7 + n}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class FakeOpenAI:
    """responder(body) возвращает: для stream — число токенов, иначе — dict сообщения; None — HTTP 500."""

    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.server.requests = []
        self.server.token_delay = 0.002
        self.server.responder = lambda body: 4 if body.get("stream") else {"content": "ok"}
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    @property
    def requests(self) -> list[dict]:
        return self.server.requests

    def set_responder(self, fn: Callable[[dict], object]) -> None:
        self.server.responder = fn

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_openai():
    server = FakeOpenAI()
    yield server
    server.close()
