from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

import bta_repro.mujoco_partitions as partitions


class Node:
    def __init__(self, label, coordinate, actuator):
        self.label = label
        self.qpos_ids = self.qvel_ids = coordinate
        self.act_ids = actuator
        self.bodies = [1, 2]
        self.extra_obs = {"test": lambda env: env}


class FakeBackend:
    def __init__(self, scenario, requested, obsk=1):
        bootstrap, groups = partitions.EXTENSIONS[(scenario, requested)]
        self.scenario = scenario
        self.agent_conf = bootstrap
        self.args = namedtuple("Args", "agent_conf")(bootstrap)
        self.agent_obsk = obsk
        labels = [label for group in groups for label in group]
        actions = [action for group in partitions.ACTUATOR_MAPPINGS[(scenario, requested)] for action in group]
        count = len(labels)
        nodes = [Node(label, index-count, action) for index, (label, action) in enumerate(zip(labels, actions))]
        # Temporary bootstrap grouping differs deliberately from requested groups.
        self.agent_partitions = [tuple(nodes)]
        self.mujoco_edges = [set((left, right)) for left, right in zip(nodes, nodes[1:])]
        self.mujoco_globals = {"joints": [nodes[0]]}
        self.n_agents = self.n = 1
        self.n_actions = count
        self.obs_size = count * 3
        self.k_dicts = [{0: nodes}]
        joint_names = {self._joint_name(label): index for index, label in enumerate(labels)}
        actuator_trnid = np.zeros((count, 2), dtype=int)
        for index, actuator in enumerate(actions):
            actuator_trnid[actuator, 0] = index
        model = SimpleNamespace(nq=count+7, nv=count+6, joint_name2id=joint_names.__getitem__,
                                jnt_qposadr=np.arange(count)+7, jnt_dofadr=np.arange(count)+6,
                                actuator_trntype=np.zeros(count, dtype=int), actuator_trnid=actuator_trnid)
        bounds = SimpleNamespace(low=-np.arange(1, count+1, dtype=float), high=np.arange(2, count+2, dtype=float))
        self.env = SimpleNamespace(model=model, action_space=bounds)

    @staticmethod
    def _joint_name(label):
        return f"{label[:-1]}_{label[-1]}" if label.startswith(("hip", "ankle")) and label[-1].isdigit() else label

    def get_obs_size(self):
        assert not hasattr(self, "obs_size"), "old observation padding must be discarded before measuring"
        if self.agent_obsk is None:
            return 101
        return max(sum(len(layer) for layer in neighborhood.values()) for neighborhood in self.k_dicts) * 3


@pytest.fixture
def pure_backend(monkeypatch):
    def neighborhoods(groups, edges, distance):
        result = []
        for group in groups:
            layers = {0: sorted(group, key=lambda node: node.label)}
            seen = set(group)
            frontier = set(group)
            for step in range(1, distance+1):
                adjacent = set().union(*(edge for edge in edges if edge & frontier)) - seen
                layers[step] = sorted(adjacent, key=lambda node: node.label)
                seen |= adjacent
                frontier = adjacent
            result.append(layers)
        return result

    def spaces(backend):
        indices = [[node.act_ids for node in part] for part in backend.agent_partitions]
        bounds = backend.env.action_space
        return [(bounds.low[index], bounds.high[index]) for index in indices], [(backend.obs_size,)] * backend.n_agents

    monkeypatch.setattr(partitions, "_neighborhoods", neighborhoods)
    monkeypatch.setattr(partitions, "_spaces", spaces)
    return FakeBackend


def test_manifest_mujoco_suite_is_exactly_covered_without_action_padding():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "configs/paper_targets.json").read_text())
    targets = {tuple(entry["task"].rsplit("_", 1)) for entry in manifest["benchmarks"] if entry["suite"] == "mamujoco"}
    assert targets == set(partitions.ACTUATOR_MAPPINGS)
    assert len(targets) == 12
    for mapping in partitions.ACTUATOR_MAPPINGS.values():
        assert len({len(group) for group in mapping}) == 1
        flat = [action for group in mapping for action in group]
        assert sorted(flat) == list(range(len(flat)))


@pytest.mark.parametrize("scenario,requested", partitions.EXTENSIONS)
@pytest.mark.parametrize("obsk", [None, 0, 1])
def test_extensions_preserve_graph_and_rebuild_dependent_metadata(pure_backend, scenario, requested, obsk):
    backend = pure_backend(scenario, requested, obsk)
    original_nodes = list(backend.agent_partitions[0])
    original_edges = [{node.label for node in edge} for edge in backend.mujoco_edges]
    original_function = original_nodes[0].extra_obs["test"]
    assert partitions.extend_backend_partition(backend, requested)
    actual = tuple(tuple(node.label for node in part) for part in backend.agent_partitions)
    assert actual == partitions.EXTENSIONS[(scenario, requested)][1]
    assert backend.agent_conf == backend.args.agent_conf == requested
    assert backend.n == backend.n_agents == len(actual)
    assert backend.n_actions == len(actual[0])
    assert [{node.label for node in edge} for edge in backend.mujoco_edges] == original_edges
    copied_nodes = {node for part in backend.agent_partitions for node in part}
    assert not copied_nodes.intersection(original_nodes)
    assert all(edge <= copied_nodes for edge in backend.mujoco_edges)
    assert backend.mujoco_globals["joints"][0] in copied_nodes
    assert backend.mujoco_globals["joints"][0].extra_obs["test"] is original_function
    if obsk is None:
        assert backend.obs_size == 101
    else:
        assert len(backend.k_dicts) == backend.n_agents
        assert all(set(neighborhood[0]) == set(part) for part, neighborhood in zip(backend.agent_partitions, backend.k_dicts))
    for part, (low, high) in zip(backend.agent_partitions, backend.action_space):
        index = [node.act_ids for node in part]
        np.testing.assert_array_equal(low, backend.env.action_space.low[index])
        np.testing.assert_array_equal(high, backend.env.action_space.high[index])
    assert backend.observation_space == [(backend.obs_size,)] * backend.n_agents


def test_actuator_repair_uses_physical_joint_and_leaves_other_graph_unchanged(pure_backend):
    backend = pure_backend("Humanoid-v2", "17x1")
    original = backend.agent_partitions[0][0]
    original.act_ids = 0  # shoulder incorrectly linked to abdomen
    partitions.extend_backend_partition(backend, "17x1")
    assert original.act_ids == 0
    assert backend.agent_partitions[0][0].act_ids == 14
    assert backend._bta_partition_corrections == [{"joint": "left_shoulder1", "old": 0, "new": 14}]


@pytest.mark.parametrize("failure", ["wrong_observation", "missing_actuator", "duplicate_actuator", "missing_node"])
def test_inconsistent_anatomy_fails_before_changing_backend(pure_backend, failure):
    backend = pure_backend("HalfCheetah-v2", "3x2")
    original_parts = backend.agent_partitions
    if failure == "wrong_observation":
        backend.env.model.jnt_qposadr[0] += 1
    elif failure == "missing_actuator":
        backend.env.model.actuator_trntype[2] = 1
    elif failure == "duplicate_actuator":
        backend.env.model.actuator_trnid[1, 0] = backend.env.model.actuator_trnid[2, 0]
    else:
        backend.agent_partitions = [original_parts[0][1:]]
        original_parts = backend.agent_partitions
    with pytest.raises(ValueError):
        partitions.extend_backend_partition(backend, "3x2")
    assert backend.agent_partitions is original_parts
    assert backend.agent_conf == "2x3"


def test_original_supported_partition_is_left_unchanged():
    backend = SimpleNamespace(scenario="Ant-v2")
    assert not partitions.extend_backend_partition(backend, "2x4d")
    assert vars(backend) == {"scenario": "Ant-v2"}
    assert partitions.bootstrap_partition("Ant-v2", "2x4d") == "2x4d"
    assert partitions.bootstrap_partition("HumanoidStandup-v2", "17x1") == "9|8"


def test_author_humanoid_agent_order_is_explicit():
    expected = ("left_shoulder1", "left_shoulder2", "abdomen_x", "abdomen_y", "abdomen_z",
                "right_shoulder1", "right_shoulder2", "right_elbow", "left_elbow",
                "left_hip_x", "left_hip_y", "left_hip_z", "right_hip_x", "right_hip_y",
                "right_hip_z", "right_knee", "left_knee")
    for scenario in ("Humanoid-v2", "HumanoidStandup-v2"):
        assert partitions.EXTENSIONS[(scenario, "17x1")][1] == tuple((name,) for name in expected)
