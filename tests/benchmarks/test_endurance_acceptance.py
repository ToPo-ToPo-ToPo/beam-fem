from benchmarks.endurance_acceptance import _rss_bytes, run_endurance


def test_short_endurance_probe_is_finite_deterministic_and_bounded():
    evidence = run_endurance(
        minimum_seconds=0.0, minimum_evaluations=3, memory_limit_mb=1024.0
    )
    assert evidence["passed"]
    assert all(evidence["checks"].values())
    assert evidence["completed_evaluations"] >= 3
    assert evidence["environment"]["python"]
    assert _rss_bytes() > 0
