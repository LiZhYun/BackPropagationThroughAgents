"""Portable checkpoints with optimizer state and RNG state for exact continuation."""
from pathlib import Path
import random
import numpy as np
import torch


def save_checkpoint(path, *, policy, critic, optimizers, config, step, extra_state=None):
    """Save optional preprocessing state using weights_only-safe dictionaries."""
    if extra_state is not None and not isinstance(extra_state, dict):
        raise ValueError("extra_state must be a dictionary or None")
    path = Path(path)
    np_state = np.random.get_state()
    payload = {
        "format_version": 1, "policy": policy.state_dict(), "critic": critic.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "config": config, "step": step,
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4])},
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, policy, critic, optimizers):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    payload.setdefault("extra_state", {})
    if not isinstance(payload["extra_state"], dict):
        raise ValueError("Checkpoint extra_state must be a dictionary")
    if set(payload["optimizers"]) != set(optimizers):
        raise ValueError("Checkpoint optimizer names do not match")
    policy.load_state_dict(payload["policy"])
    critic.load_state_dict(payload["critic"])
    for name, opt in optimizers.items():
        opt.load_state_dict(payload["optimizers"][name])
    rng = payload["rng"]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    n = rng["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    return payload
