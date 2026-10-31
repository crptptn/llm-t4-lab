from lab.common import save_json
from lab.report import build_report

ENV = {"python": "3.11", "gpus": ["Tesla T4, 15360 MiB"], "packages": {"vllm": "0.32.0"}, "timestamp": "-"}


def _level(c, ttft):
    stats = {"mean": ttft, "p50": ttft, "p90": ttft * 2, "p99": ttft * 3, "n": 4}
    return {
        "concurrency": c,
        "errors": 0,
        "output_throughput_tok_s": 100.0 * c,
        "request_throughput_req_s": 1.0,
        "ttft_s": stats,
        "tpot_s": {**stats, "p50": 0.02},
    }


def test_empty_report(tmp_path):
    assert "Пока пусто" in build_report(tmp_path, plots=False)


def test_report_with_results(tmp_path):
    for tag, ttft in (("prefix_on", 0.05), ("prefix_off", 0.2)):
        save_json(
            {
                "model": "m",
                "scenario": "shared-prefix",
                "max_tokens": 256,
                "mean_prompt_tokens": 1800,
                "levels": [_level(1, ttft), _level(8, ttft * 2)],
                "env": ENV,
            },
            tmp_path / f"bench_{tag}.json",
        )
    save_json(
        {
            "n": 10,
            "base_model": "m",
            "modes": {"lora": {"accuracy": 0.9, "macro_f1": 0.88, "invalid_rate": 0.0}},
            "env": ENV,
        },
        tmp_path / "appeals_eval.json",
    )
    report = build_report(tmp_path, plots=True, base=tmp_path)
    assert "Tesla T4" in report
    assert "×4.0" in report  # 0.2 / 0.05
    assert "| lora | 0.900 | 0.880 | 0.0% |" in report
    assert (tmp_path / "plots" / "bench_ttft.png").exists()
    assert "](plots/bench_throughput.png)" in report
