"""Compare raw sequential and spawned environment calls; never train a policy.

Full transition traces must match before reporting paired call-time ratios.
This diagnostic does not integrate processes into the training collector.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import socket
import statistics
import time
import traceback

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    import bta_repro
    return {p.name: digest(p) for p in sorted(Path(bta_repro.__file__).parent.glob("*.py"))}


def make_env(environment, config, seed):
    if environment == "smac":
        from bta_repro.smac_env import SmacEnv
        return SmacEnv({**config, "seed": seed})
    if environment == "prey":
        from bta_repro.prey_env import PreyEnv
        return PreyEnv({**config, "seed": seed})
    raise ValueError("diagnostic supports smac or prey")


def specification(env):
    names = ("is_discrete", "num_agents", "obs_dim", "state_dim", "num_actions",
             "episode_limit", "config", "versions")
    return {name: getattr(env, name) for name in names}


def _worker(connection, environment, config, seed):
    env, command = None, "ready"
    try:
        env = make_env(environment, config, seed)
        connection.send((command, True, specification(env)))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command == "reset":
                result = env.reset()
            elif command == "step":
                result = env.step(argument)
            else:
                raise ValueError(f"unknown environment command: {command}")
            connection.send((command, True, result))
    except BaseException:
        try:
            connection.send((command, False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        try:
            if env is not None:
                env.close()
            connection.send(("closed", True, None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            connection.close()


class ProcessEnv:
    """One owned environment per spawned process, with explicit reply ordering."""

    def __init__(self, environment, config, seed, timeout_seconds):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, environment, config, seed))
        self.timeout_seconds = timeout_seconds
        self.pending = "ready"
        self.closed = False
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()

    def send(self, command, argument=None):
        if self.closed or self.pending is not None:
            raise RuntimeError("worker is closed or has an outstanding reply")
        self.connection.send((command, argument))
        self.pending = command

    def receive(self):
        if self.closed or self.pending is None:
            raise RuntimeError("no outstanding environment reply")
        if not self.connection.poll(self.timeout_seconds):
            raise TimeoutError(f"environment worker timed out during {self.pending}")
        try:
            command, ok, result = self.connection.recv()
        except EOFError as error:
            raise RuntimeError(f"environment worker exited during {self.pending}") from error
        expected, self.pending = self.pending, None
        if command != expected or not ok:
            raise RuntimeError(f"environment worker {command}: {result}")
        return result

    def close(self):
        if self.closed:
            return self.cleanup
        graceful = False
        try:
            if self.process.is_alive():
                self.connection.send(("close", None))
            # Drain an outstanding result so a worker can finish its send and close.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.connection.poll(.05):
                    try:
                        command, ok, _ = self.connection.recv()
                    except EOFError:
                        break
                    if command == "closed" and ok:
                        graceful = True
                        break
                elif not self.process.is_alive():
                    break
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            self.process.join(timeout=1)
            forced = self.process.is_alive()
            if forced:
                self.process.terminate()  # Only the process created by this object.
                self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.connection.close()
            self.closed = True
            self.cleanup = {"pid": self.process.pid, "graceful": graceful, "forced": forced,
                            "exit_code": self.process.exitcode, "still_alive": self.process.is_alive()}
        return self.cleanup


def json_value(value):
    """Preserve array dtype, shape and every value in the equality trace."""
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def validate(config):
    allowed = {"environment", "env_config", "seeds", "decisions_per_environment", "repeats", "timeout_seconds"}
    if not isinstance(config, dict) or config.keys() != allowed:
        raise ValueError(f"configuration must contain exactly {sorted(allowed)}")
    if config["environment"] not in ("smac", "prey"):
        raise ValueError("environment must be smac or prey")
    if not isinstance(config["env_config"], dict) or "seed" in config["env_config"]:
        raise ValueError("env_config must be an object without seed")
    seeds = config["seeds"]
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("declare one or two distinct uint32 environment seeds")
    for key, maximum in (("decisions_per_environment", 8192), ("repeats", 5)):
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError(f"{key} must be an integer in [1, {maximum}]")
    timeout = config["timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("timeout_seconds must be finite and in (0, 300]")


def dispatch(envs, mode, command, arguments=None):
    arguments = arguments if arguments is not None else [None] * len(envs)
    if mode == "process":
        for env, argument in zip(envs, arguments):
            env.send(command, argument)
        return [env.receive() for env in envs]
    return [env.step(argument) if command == "step" else env.reset()
            for env, argument in zip(envs, arguments)]


def run_variant(config, mode, trace_path, record):
    envs = []
    hashes = [hashlib.sha256() for _ in config["seeds"]]
    controllers = [np.random.default_rng(np.random.SeedSequence([0x425441, 0x50524F46, 1, seed]))
                   for seed in config["seeds"]]
    record.update(mode=mode, status="running", timings={}, completed_episodes=[0] * len(hashes),
                  max_planned_restarts=[0] * len(hashes))
    started = time.perf_counter()
    try:
        for seed in config["seeds"]:
            envs.append(ProcessEnv(config["environment"], config["env_config"], seed, config["timeout_seconds"])
                        if mode == "process" else make_env(config["environment"], config["env_config"], seed))
        specs = [env.receive() if mode == "process" else specification(env) for env in envs]
        record["timings"]["construction_seconds"] = time.perf_counter() - started
        record["environment_specs"] = specs
        with Path(trace_path).open("x") as trace:
            def emit(index, event):
                encoded = json.dumps(json_value(event), sort_keys=True, separators=(",", ":"), allow_nan=False)
                hashes[index].update((encoded + "\n").encode())
                trace.write(json.dumps({"env_index": index, "event": json.loads(encoded)}, separators=(",", ":")) + "\n")

            reset_started = time.perf_counter()
            initial = dispatch(envs, mode, "reset")
            record["timings"]["initial_reset_seconds"] = time.perf_counter() - reset_started
            current = [value[0] for value in initial]
            for i, value in enumerate(initial):
                emit(i, {"initial_reset": value, "specification": specs[i]})
            step_seconds, reset_seconds = 0., 0.
            loop_started = time.perf_counter()
            for step in range(config["decisions_per_environment"]):
                actions = []
                for observation, rng in zip(current, controllers):
                    available = np.asarray(observation["available_actions"], dtype=bool)
                    actions.append(np.array([rng.choice(np.flatnonzero(mask)) for mask in available]))
                call_started = time.perf_counter()
                transitions = dispatch(envs, mode, "step", actions)
                step_seconds += time.perf_counter() - call_started
                done = []
                for i, (observation, reward, terminated, truncated, info) in enumerate(transitions):
                    if (np.asarray(reward).ndim != 0 or not math.isfinite(float(reward))
                            or not isinstance(terminated, (bool, np.bool_))
                            or not isinstance(truncated, (bool, np.bool_))):
                        raise ValueError("invalid scalar reward or episode boundary")
                    emit(i, {"step": step, "action": actions[i], "transition": transitions[i]})
                    current[i] = observation
                    record["max_planned_restarts"][i] = max(
                        record["max_planned_restarts"][i], int(info.get("planned_restarts", 0)))
                    if terminated or truncated:
                        done.append(i)
                        record["completed_episodes"][i] += 1
                if done:
                    call_started = time.perf_counter()
                    resets = dispatch([envs[i] for i in done], mode, "reset")
                    reset_seconds += time.perf_counter() - call_started
                    for i, value in zip(done, resets):
                        current[i] = value[0]
                        record["max_planned_restarts"][i] = max(
                            record["max_planned_restarts"][i], int(value[1].get("planned_restarts", 0)))
                        emit(i, {"after_step": step, "reset": value})
            record["timings"].update(step_seconds=step_seconds, subsequent_reset_seconds=reset_seconds,
                                     loop_seconds_including_trace_io=time.perf_counter() - loop_started)
        record.update(status="completed", trace_sha256=digest(trace_path),
                      per_environment_trace_sha256=[value.hexdigest() for value in hashes],
                      decisions=config["decisions_per_environment"] * len(envs))
    except BaseException as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        close_started = time.perf_counter()
        cleanup, close_errors = [], []
        for env in reversed(envs):
            try:
                if mode == "process":
                    cleanup.append(env.close())
                else:
                    env.close()
            except Exception as error:
                close_errors.append(str(error))
        record["cleanup"] = cleanup
        record["close_errors"] = close_errors
        record["timings"]["close_seconds"] = time.perf_counter() - close_started
        record["wall_seconds_including_trace_io"] = time.perf_counter() - started
        if close_errors or any(not c["graceful"] or c["forced"] or c["still_alive"] or c["exit_code"] != 0 for c in cleanup):
            record["status"] = "failed"
            if "error" not in record:
                raise RuntimeError("environment cleanup failed; diagnostic is invalid")


def profile(config, output, source_manifest=None):
    validate(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError("diagnostic output must be a new directory")
    source = source_hashes()
    if source_manifest is not None and source != json.loads(Path(source_manifest).read_text())["source_sha256"]:
        raise ValueError("loaded package differs from the selected frozen source")
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Raw environment trace and call-time diagnostic; no policy learning or training speedup claim.",
              "recorded_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "source_sha256": source, "profiler_sha256": digest(__file__),
              "host": socket.gethostname(), "python": platform.python_version(), "numpy": np.__version__,
              "status": "running", "runs": [], "pairs": [],
              "limits": ["Native+IPC call times exclude policy inference, normalization, optimization and trace I/O.",
                         "Whole-loop and wall times include diagnostic trace serialization and I/O.",
                         "Initial construction/reset and final cleanup are timed separately.",
                         "A forced worker cleanup invalidates the diagnostic; native descendants require allocation cleanup.",
                         "Slurm batch-step MaxRSS must be inspected before adopting process execution."]}
    cpu_info = Path("/proc/cpuinfo")
    report["cpu"] = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith("model name")), None) if cpu_info.exists() else platform.processor()
    try:
        first_trace = None
        for trial in range(config["repeats"]):
            modes = ("sequential", "process") if trial % 2 == 0 else ("process", "sequential")
            pair = {}
            for mode in modes:
                record = {"trial": trial, "trace_path": f"trial{trial}_{mode}.jsonl"}
                report["runs"].append(record)
                run_variant(config, mode, output / record["trace_path"], record)
                pair[mode] = record
            equal = pair["sequential"]["per_environment_trace_sha256"] == pair["process"]["per_environment_trace_sha256"]
            current_trace = pair["sequential"]["per_environment_trace_sha256"]
            repeat_equal = first_trace is None or current_trace == first_trace
            checked = {"trial": trial, "trace_equal": equal, "repeat_trace_equal": repeat_equal}
            report["pairs"].append(checked)
            if not equal or not repeat_equal:
                raise ValueError(f"sequential/process native trace mismatch in trial {trial}")
            first_trace = current_trace
            times = {mode: pair[mode]["timings"]["step_seconds"] + pair[mode]["timings"]["subsequent_reset_seconds"]
                     for mode in modes}
            checked.update(call_seconds=times, sequential_over_process_call_ratio=times["sequential"] / times["process"])
        report["median_sequential_over_process_call_ratio"] = statistics.median(
            p["sequential_over_process_call_ratio"] for p in report["pairs"])
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    report = profile(json.loads(args.config.read_text()), args.output, args.source_manifest)
    print(json.dumps({key: report[key] for key in ("status", "pairs", "median_sequential_over_process_call_ratio")}, indent=2))


if __name__ == "__main__":
    main()
