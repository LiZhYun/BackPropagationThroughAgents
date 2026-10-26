"""A standalone process diagnostic must preserve native traces and fail closed."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
# Spawn imports the standalone module by name in child interpreters.
sys.path.insert(0, str(ROOT / "scripts"))
import profile_environment_processes as profiler


@pytest.fixture
def config():
    return {"environment": "prey", "env_config": {"episode_limit": 3, "penalty": -1},
            "seeds": [1091, 6290], "decisions_per_environment": 8, "repeats": 2, "timeout_seconds": 20}


def test_real_prey_traces_match_across_processes_resets_and_repeats(tmp_path, config):
    python_before, numpy_before = random.getstate(), np.random.get_state()
    before_children = {p.pid for p in mp.active_children()}
    result = profiler.profile(config, tmp_path / "profile")
    assert result["status"] == "completed"
    assert all(p["trace_equal"] and p["repeat_trace_equal"] for p in result["pairs"])
    assert [r["mode"] for r in result["runs"]] == ["sequential", "process", "process", "sequential"]
    for record in result["runs"]:
        assert record["decisions"] == 16
        assert all(count >= 2 for count in record["completed_episodes"])
        assert record["close_errors"] == []
        if record["mode"] == "process":
            assert len(record["cleanup"]) == 2
            assert all(c["graceful"] and not c["forced"] and not c["still_alive"] and c["exit_code"] == 0
                       for c in record["cleanup"])
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert {p.pid for p in mp.active_children()} == before_children
    assert json.loads((tmp_path / "profile/report.json").read_text()) == result


def test_trace_preserves_terminal_observation_and_separate_reset(tmp_path, config):
    config.update(repeats=1, decisions_per_environment=3)
    result = profiler.profile(config, tmp_path / "profile")
    records = [json.loads(line) for line in (tmp_path / "profile/trial0_process.jsonl").read_text().splitlines()]
    events = [r["event"] for r in records if r["env_index"] == 0]
    final_transition = next(e for e in events if e.get("step") == 2)["transition"]
    assert final_transition[3] is True
    terminal_obs = final_transition[0]["obs"]
    reset_obs = next(e for e in events if e.get("after_step") == 2)["reset"][0]["obs"]
    assert terminal_obs["dtype"] == reset_obs["dtype"] == "float32"
    assert terminal_obs["values"][0][-1] == 1
    assert reset_obs["values"][0][-1] == 0
    assert result["pairs"][0]["trace_equal"]


def test_unknown_worker_command_preserves_error_and_closes_owned_process():
    worker = profiler.ProcessEnv("prey", {"episode_limit": 3}, 7, 10)
    try:
        assert worker.receive()["num_agents"] == 2
        worker.send("unknown")
        with pytest.raises(RuntimeError, match="unknown environment command"):
            worker.receive()
    finally:
        cleanup = worker.close()
    assert not cleanup["still_alive"] and cleanup["exit_code"] != 0
    assert worker.close() == cleanup


def test_dead_worker_is_detected_without_waiting_for_timeout():
    worker = profiler.ProcessEnv("prey", {}, 7, 10)
    worker.receive()
    worker.process.terminate()  # This test's owned child only.
    worker.process.join(timeout=2)
    worker.pending = "step"
    try:
        with pytest.raises(RuntimeError, match="exited"):
            worker.receive()
    finally:
        cleanup = worker.close()
    assert not cleanup["still_alive"] and not cleanup["graceful"]


def test_close_drains_pending_result_before_worker_shutdown():
    worker = profiler.ProcessEnv("prey", {}, 7, 10)
    worker.receive()
    worker.send("reset")
    cleanup = worker.close()
    assert cleanup["graceful"] and not cleanup["forced"] and not cleanup["still_alive"]


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="POSIX stopped-child cleanup")
def test_unresponsive_owned_worker_times_out_and_is_reaped():
    worker = profiler.ProcessEnv("prey", {}, 7, 10)
    worker.receive()
    os.kill(worker.process.pid, signal.SIGSTOP)
    worker.timeout_seconds = .05
    try:
        worker.send("reset")
        with pytest.raises(TimeoutError, match="timed out during reset"):
            worker.receive()
    finally:
        cleanup = worker.close()
    assert cleanup["forced"] and not cleanup["graceful"] and not cleanup["still_alive"]


def test_constructor_failure_retains_failed_report_and_no_live_workers(tmp_path, config):
    config["env_config"]["episode_limit"] = 0
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(ValueError, match="episode_limit"):
        profiler.profile(config, tmp_path / "failed")
    result = json.loads((tmp_path / "failed/report.json").read_text())
    assert result["status"] == result["runs"][0]["status"] == "failed"
    assert "episode_limit" in result["error"]["message"]
    assert {p.pid for p in mp.active_children()} == before


def test_process_constructor_failure_cleans_every_started_worker(tmp_path, config):
    config["env_config"]["episode_limit"] = 0
    before = {p.pid for p in mp.active_children()}
    record = {}
    with pytest.raises(RuntimeError, match="episode_limit"):
        profiler.run_variant(config, "process", tmp_path / "trace.jsonl", record)
    assert record["status"] == "failed" and len(record["cleanup"]) == 2
    assert all(not c["still_alive"] for c in record["cleanup"])
    assert {p.pid for p in mp.active_children()} == before


def test_trace_mismatch_is_failure_and_not_a_speedup_result(tmp_path, config, monkeypatch):
    config["repeats"] = 1
    original = profiler.run_variant

    def corrupt(*args):
        original(*args)
        if args[1] == "process":
            args[3]["per_environment_trace_sha256"][0] = "mismatch"

    monkeypatch.setattr(profiler, "run_variant", corrupt)
    with pytest.raises(ValueError, match="trace mismatch"):
        profiler.profile(config, tmp_path / "mismatch")
    result = json.loads((tmp_path / "mismatch/report.json").read_text())
    assert result["status"] == "failed"
    assert result["pairs"][0]["trace_equal"] is False
    assert "median_sequential_over_process_call_ratio" not in result


@pytest.mark.parametrize("key,value", [
    ("seeds", []), ("seeds", [0, 0]), ("seeds", [1, 2, 3]), ("seeds", [True]),
    ("seeds", [-1]), ("decisions_per_environment", True), ("decisions_per_environment", 0),
    ("decisions_per_environment", 10000), ("repeats", 0), ("timeout_seconds", float("nan")),
    ("environment", "unavailable"), ("env_config", {"seed": 1}),
])
def test_invalid_bounds_create_no_outputs(tmp_path, config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        profiler.profile(config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_existing_results_and_wrong_source_are_preserved(tmp_path, config):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "marker").write_text("keep")
    with pytest.raises(FileExistsError):
        profiler.profile(config, output)
    assert list(output.iterdir()) == [output / "marker"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_sha256": {"wrong.py": "wrong"}}))
    with pytest.raises(ValueError, match="loaded package differs"):
        profiler.profile(config, tmp_path / "wrong_source", manifest)
    assert not (tmp_path / "wrong_source").exists()


def test_cli_spawn_runs_without_optional_simulator_packages(tmp_path, config):
    config.update(repeats=1, decisions_per_environment=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "cli"
    subprocess.run([sys.executable, str(ROOT / "scripts/profile_environment_processes.py"),
                    "--config", str(config_path), "--output", str(output)], check=True, capture_output=True)
    assert json.loads((output / "report.json").read_text())["status"] == "completed"
