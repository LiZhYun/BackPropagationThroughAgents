"""Small per-instance partition extensions for the papers' legacy MuJoCo tasks.

Reuse upstream anatomical nodes, graph edges, observations and physics. Only
the explicit grouping comes from the author repository; no module is patched.
"""

from copy import deepcopy

import numpy as np


AUTHOR_COMMIT = "13607d0de0fd02c6e66c4811d6e2ca3ed06bf8a0"
PARTITION_EXTENSION_VERSION = "paper-partitions-v1"

# Policy action component order, expressed in physical global actuator indices.
ACTUATOR_MAPPINGS = {
    ("HalfCheetah-v2", "2x3"): ((2, 1, 0), (5, 4, 3)),
    ("HalfCheetah-v2", "3x2"): ((2, 1), (0, 5), (4, 3)),
    ("HalfCheetah-v2", "6x1"): ((2,), (1,), (0,), (5,), (4,), (3,)),
    ("Ant-v2", "2x4"): ((2, 3, 4, 5), (6, 7, 0, 1)),
    ("Ant-v2", "2x4d"): ((2, 3, 6, 7), (4, 5, 0, 1)),
    ("Ant-v2", "4x2"): ((2, 3), (4, 5), (6, 7), (0, 1)),
    ("Ant-v2", "8x1"): ((2,), (3,), (4,), (5,), (6,), (7,), (0,), (1,)),
    ("Walker2d-v2", "2x3"): ((2, 1, 0), (5, 4, 3)),
    ("Walker2d-v2", "3x2"): ((2, 1), (0, 5), (4, 3)),
    ("Walker2d-v2", "6x1"): ((2,), (1,), (0,), (5,), (4,), (3,)),
    ("Humanoid-v2", "17x1"): ((14,), (15,), (2,), (0,), (1,), (11,), (12,), (13,),
                                  (16,), (7,), (9,), (8,), (3,), (5,), (4,), (6,), (10,)),
    ("HumanoidStandup-v2", "17x1"): ((14,), (15,), (2,), (0,), (1,), (11,), (12,), (13,),
                                         (16,), (7,), (9,), (8,), (3,), (5,), (4,), (6,), (10,)),
}

_HUMANOID_GROUPS = tuple((name,) for name in (
    "left_shoulder1", "left_shoulder2", "abdomen_x", "abdomen_y", "abdomen_z",
    "right_shoulder1", "right_shoulder2", "right_elbow", "left_elbow",
    "left_hip_x", "left_hip_y", "left_hip_z", "right_hip_x", "right_hip_y",
    "right_hip_z", "right_knee", "left_knee",
))

# Each value is (existing upstream grouping used to load the graph, author groups).
EXTENSIONS = {
    ("HalfCheetah-v2", "3x2"): ("2x3", (("bfoot", "bshin"), ("bthigh", "ffoot"), ("fshin", "fthigh"))),
    ("Ant-v2", "8x1"): ("4x2", tuple((name,) for name in
                                      ("hip1", "ankle1", "hip2", "ankle2", "hip3", "ankle3", "hip4", "ankle4"))),
    ("Walker2d-v2", "3x2"): ("2x3", (("foot_joint", "leg_joint"), ("thigh_joint", "foot_left_joint"),
                                     ("leg_left_joint", "thigh_left_joint"))),
    ("Walker2d-v2", "6x1"): ("2x3", tuple((name,) for name in
                                          ("foot_joint", "leg_joint", "thigh_joint", "foot_left_joint",
                                           "leg_left_joint", "thigh_left_joint"))),
    ("Humanoid-v2", "17x1"): ("9|8", _HUMANOID_GROUPS),
    ("HumanoidStandup-v2", "17x1"): ("9|8", _HUMANOID_GROUPS),
}


def bootstrap_partition(scenario: str, partition: str) -> str:
    """Select an upstream graph loader; its temporary grouping is never exposed."""
    return EXTENSIONS.get((scenario, partition), (partition,))[0]


def _model_actuator(node, model) -> int:
    """Resolve by anatomical joint name, cross-checking observation coordinates."""
    name = node.label
    if name.startswith(("hip", "ankle")) and name[-1:].isdigit():
        name = f"{name[:-1]}_{name[-1]}"  # Ant XML uses hip_1, ankle_1, etc.
    joint = model.joint_name2id(name)
    if model.jnt_qposadr[joint] != node.qpos_ids % model.nq or model.jnt_dofadr[joint] != node.qvel_ids % model.nv:
        raise ValueError(f"observation coordinates for {node.label} do not match the model joint")
    matches = np.flatnonzero((np.asarray(model.actuator_trntype) == 0)
                             & (np.asarray(model.actuator_trnid)[:, 0] == joint))
    if len(matches) != 1:
        raise ValueError(f"joint {node.label} must have exactly one joint-transmission actuator")
    return int(matches[0])


def _neighborhoods(partitions, edges, distance):
    from multiagent_mujoco.obsk import get_joints_at_kdist

    return [get_joints_at_kdist(index, partitions, edges, k=distance, kagents=False)
            for index in range(len(partitions))]


def _spaces(backend):
    from gym.spaces import Box

    # Upstream slices contiguous bounds. These must instead follow act_ids.
    bounds = backend.env.action_space
    actions = tuple(Box(low=bounds.low[[node.act_ids for node in part]],
                        high=bounds.high[[node.act_ids for node in part]], dtype=np.float32)
                    for part in backend.agent_partitions)
    observations = [Box(low=-np.inf, high=np.inf, shape=(backend.obs_size,), dtype=np.float32)
                    for _ in backend.agent_partitions]
    return actions, observations


def extend_backend_partition(backend, partition: str) -> bool:
    """Replace one private backend's grouping and rebuild all dependent caches.

    The graph is copied as a whole so edges and groups refer to the same copied
    nodes. Existing environments and upstream globals are never modified.
    """
    extension = EXTENSIONS.get((backend.scenario, partition))
    if extension is None:
        return False
    _, groups = extension
    old_parts, edges, globals_ = deepcopy((backend.agent_partitions, backend.mujoco_edges, backend.mujoco_globals))
    nodes = {node.label: node for part in old_parts for node in part}
    labels = [label for group in groups for label in group]
    if len(labels) != len(set(labels)) or set(labels) != set(nodes):
        raise ValueError("extended partition must use every upstream anatomical node exactly once")
    parts = [tuple(nodes[label] for label in group) for group in groups]
    corrections = []
    for node in nodes.values():
        actuator = _model_actuator(node, backend.env.model)
        if node.act_ids != actuator:
            corrections.append({"joint": node.label, "old": int(node.act_ids), "new": actuator})
        node.act_ids = actuator
    actual = tuple(tuple(node.act_ids for node in part) for part in parts)
    if actual != ACTUATOR_MAPPINGS[(backend.scenario, partition)]:
        raise ValueError("model-derived actuator mapping differs from the audited paper partition")
    backend.agent_partitions, backend.mujoco_edges, backend.mujoco_globals = parts, edges, globals_
    backend.agent_conf = partition
    backend.args = backend.args._replace(agent_conf=partition)
    backend.n_agents = backend.n = len(parts)
    backend.n_actions = len(parts[0])
    if any(len(part) != backend.n_actions for part in parts):
        raise ValueError("paper partitions require equal local action dimensions")
    if backend.agent_obsk is not None:
        backend.k_dicts = _neighborhoods(parts, edges, backend.agent_obsk)
    # Old obs_size would incorrectly pad the new observations before measuring.
    del backend.obs_size
    backend.obs_size = backend.get_obs_size()
    backend.action_space, backend.observation_space = _spaces(backend)
    backend._bta_partition_corrections = corrections
    return True
