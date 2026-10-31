import asyncio

import pytest

from lab.bench import RequestResult, build_messages, run_level
from lab.common import make_async_client
from lab.prompts import build_shared_prefix


def _run_level(url: str, **kwargs) -> dict:
    async def go():
        client = make_async_client(url)
        try:
            return await run_level(client, "fake", **kwargs)
        finally:
            await client.close()

    return asyncio.run(go())


def test_run_level_measures_ttft_tpot_and_throughput(fake_openai):
    fake_openai.set_responder(lambda body: 6)
    level = _run_level(fake_openai.url, concurrency=3, num_requests=6, scenario="unique", max_tokens=6)
    assert level["errors"] == 0
    assert level["output_tokens"]["mean"] == 6  # берётся из usage, а не по числу чанков
    assert level["prompt_tokens"]["mean"] == 7
    assert 0 < level["ttft_s"]["p50"] < level["e2e_s"]["p50"]
    assert level["tpot_s"]["p50"] > 0
    assert level["output_throughput_tok_s"] > 0
    body = fake_openai.requests[0]
    assert body["stream"] is True and body["ignore_eos"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_run_level_counts_server_errors(fake_openai):
    fake_openai.set_responder(lambda body: None)
    level = _run_level(fake_openai.url, concurrency=2, num_requests=4, scenario="unique", max_tokens=4)
    assert level["errors"] == 4
    assert level["ttft_s"] is None
    assert len(fake_openai.requests) == 4  # без повторов: ретраи исказили бы задержки


def test_messages_scenarios():
    prefix = build_shared_prefix(4)
    unique = [build_messages("unique", i, "")[0]["content"] for i in range(3)]
    assert len(set(unique)) == 3 and all(m.startswith("Запрос №") for m in unique)
    shared = build_messages("shared-prefix", 1, prefix)
    assert shared[0] == {"role": "system", "content": prefix}
    with pytest.raises(ValueError):
        build_messages("other", 0, "")


def test_tpot_needs_two_tokens():
    assert RequestResult(ok=True, ttft=0.1, e2e=0.5, output_tokens=1).tpot is None
    assert RequestResult(ok=True, ttft=0.1, e2e=0.5, output_tokens=5).tpot == pytest.approx(0.1)
