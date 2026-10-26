"""Time CPU simulator collection and PPO updates without changing their calls.

Use a short, explicitly labeled diagnostic configuration. The timing report
records instrumentation separately from the trainer's source-file hashes.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import time

import torch
from bta_repro import simulator_train as training


class Timings:
    def __init__(self):
        self.phase = 'outside_updates'
        self.calls = {}

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            record = self.calls.setdefault(name, {'calls': 0, 'seconds': 0.})
            record['calls'] += 1
            record['seconds'] += time.perf_counter() - started

    def wrap(self, name, function):
        def measured(*args, **kwargs):
            previous = self.phase
            self.phase = name
            try:
                with self.measure(name):
                    return function(*args, **kwargs)
            finally:
                self.phase = previous
        return measured


class TimedEnv:
    """Forward adapter calls and attributes; no observation or action copying."""

    def __init__(self, env, timings):
        self.env, self.timings = env, timings

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        with self.timings.measure(self.timings.phase + '.env_reset'):
            return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        with self.timings.measure(self.timings.phase + '.env_step'):
            return self.env.step(*args, **kwargs)

    def close(self):
        with self.timings.measure(self.timings.phase + '.env_close'):
            return self.env.close()


def profile(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('profiling output must be a new directory')
    if torch.device(config.device).type != 'cpu':
        raise ValueError('this wall-clock profiler supports CPU execution only')
    timings = Timings()
    originals = {name: getattr(training, name) for name in
                 ('make_env', 'collect_rollout', 'update_rollout', 'evaluate')}

    def make_env(*args, **kwargs):
        with timings.measure(timings.phase + '.env_construct'):
            return TimedEnv(originals['make_env'](*args, **kwargs), timings)

    training.make_env = make_env
    for name in ('collect_rollout', 'update_rollout', 'evaluate'):
        setattr(training, name, timings.wrap(name, originals[name]))
    started = time.perf_counter()
    error = None
    try:
        return training.train(config, output)
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        elapsed = time.perf_counter() - started
        for name, function in originals.items():
            setattr(training, name, function)
        if output.is_dir():
            cpu_info = Path('/proc/cpuinfo')
            cpu = next((line.split(':', 1)[1].strip() for line in cpu_info.read_text().splitlines()
                        if line.startswith('model name')), None) if cpu_info.exists() else platform.processor()
            metadata_path = output / 'config.json'
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
            report = {
                'scope': 'Instrumented runtime diagnostic; short-run rewards are not benchmark reproduction evidence.',
                'recorded_utc': datetime.now(timezone.utc).isoformat(), 'status': 'failed' if error else 'completed',
                'error': error, 'wall_seconds': elapsed, 'timings': timings.calls,
                'host': socket.gethostname(), 'cpu': cpu, 'python': platform.python_version(),
                'profiler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'training_metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata else None,
                'source_sha256': metadata['source_sha256'] if metadata else None,
                'interpretation': [
                    'collect_rollout includes its env_step and env_reset timings; do not add nested times twice.',
                    'evaluate includes its own environment construction, resets, steps and close calls.',
                    'outside_updates includes environment setup and final cleanup; logging/checkpoint/other work is not individually timed.',
                    'All original calls, arguments, results and exceptions pass through unchanged; timing itself adds small overhead.',
                    'Native timing fractions apply to this configuration and machine, not automatically to other batches, policies or hardware.',
                ],
            }
            with (output / 'runtime_profile.json').open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = training.Config(**json.loads(args.config.read_text()))
    summary = profile(config, args.output)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
