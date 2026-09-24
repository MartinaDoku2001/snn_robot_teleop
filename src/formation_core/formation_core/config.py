"""YAML-backed configuration and the reproducibility contract.

An :class:`EpisodeConfig` fully determines an episode: same config, same seed,
same trajectory, on any machine. Every result file records the config that
produced it, so a number in the paper can always be traced back to one YAML
document.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List

import yaml

from .contract import CONTRACT_VERSION, ContractConfig
from .dynamics import MotionLimits, ProcessNoise


@dataclass
class ComponentConfig:
    """A named, parameterised component (controller, policy, predictor)."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, value, default_name='always'):
        if value is None:
            return cls(default_name, {})
        if isinstance(value, str):
            return cls(value, {})
        # Accept anything already carrying (name, params) -- including a
        # PathConfig or a ComponentConfig -- so re-parsing is idempotent and
        # dataclasses.replace() round-trips cleanly.
        if hasattr(value, 'name') and hasattr(value, 'params'):
            return cls(str(value.name), dict(value.params))
        if isinstance(value, dict):
            data = dict(value)
            name = data.pop('name', default_name)
            params = data.pop('params', None)
            # Allow both {name: periodic, k: 4} and {name: periodic, params: {k: 4}}
            merged = dict(params) if params else {}
            merged.update(data)
            return cls(str(name), merged)
        raise TypeError(f'cannot parse component config from {value!r}')

    def describe(self):
        if not self.params:
            return self.name
        inner = ','.join(f'{k}={v}' for k, v in sorted(self.params.items()))
        return f'{self.name}({inner})'


@dataclass
class PathConfig:
    name: str = 'oval'
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, value):
        component = ComponentConfig.parse(value, default_name='oval')
        return cls(component.name, component.params)


@dataclass
class LeaderConfig:
    """Leader pure-pursuit settings."""

    target_speed: float = 0.6
    lookahead: float = 0.5
    lookahead_gain: float = 0.3
    curvature_slowdown: float = 0.6


@dataclass
class RewardConfig:
    """Reward weights.

    Unused by the analytic controller, defined now so the RL phase inherits a
    reward that is already part of the recorded config. Reward is
    ``w_progress * progress - (w_formation * euclidean_error
    + w_heading * |heading_error| + w_path * leader_path_error
    + w_comm * messages + w_action * ||action||^2)``.

    ``w_path`` arrived with contract v2.0: the controller now drives the leader,
    so staying on the reference path is its job and has to be paid for. It was
    free in v1.x, where a scripted pure-pursuit leader tracked the path.

    ``w_comm`` counts messages summed over BOTH robots. It is 0.0 everywhere in
    Phase 2 and is turned on in Phase 4, but it is plumbed through now so that
    turning it on is a config change rather than a code change.

    ``w_progress`` REWARDS travelling along the reference path, credited as a
    fraction of the distance a leader cruising at ``leader.target_speed`` would
    cover, clipped to [-1, 1] so there is no bonus for exceeding the target
    speed. Without it the task has a trivial optimum: every other term is a
    penalty, so parking both robots on the path in perfect formation scores ~0
    and beats actually driving. Under v1.x this could not happen, because the
    scripted leader always drove and the follower had to keep up; centralizing
    the controller removed the thing that forced motion.

    ``terminal_penalty`` is charged once when an episode ends EARLY -- the
    formation broke or the leader lost the path. It is not charged on
    truncation, which is just the clock running out.

    It is deliberately MODEST. Most of the cost of crashing is already the
    progress reward it forfeits for the rest of the episode, which the value
    function accounts for on its own. An earlier 50.0 was large enough to
    dominate every other term during early exploration -- a policy crashing
    every ~17 steps saw -2.5 per step and never discovered driving at all.
    """

    w_formation: float = 1.0
    w_heading: float = 0.1
    w_path: float = 1.0
    w_progress: float = 0.5
    w_comm: float = 0.0
    w_action: float = 0.0
    terminal_penalty: float = 10.0


@dataclass
class EpisodeConfig:
    """Everything needed to reproduce one episode."""

    seed: int = 0
    dt: float = 0.05
    duration: float = 60.0
    offset_d: float = 0.8
    path: PathConfig = field(default_factory=PathConfig)
    noise: ProcessNoise = field(default_factory=ProcessNoise)
    limits: MotionLimits = field(default_factory=MotionLimits)
    contract: ContractConfig = field(default_factory=ContractConfig)
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    controller: ComponentConfig = field(
        default_factory=lambda: ComponentConfig('analytic', {}))
    policy: ComponentConfig = field(default_factory=lambda: ComponentConfig('always', {}))
    predictor: ComponentConfig = field(
        default_factory=lambda: ComponentConfig('constant_velocity', {}))
    reward: RewardConfig = field(default_factory=RewardConfig)
    #: Episode ends early if the follower falls this far from its slot (metres).
    max_formation_error: float = 5.0
    #: ... or if the LEADER leaves the reference path by this much (metres).
    #: v2.0 only: the controller drives the leader, so it can now lose the path,
    #: and a training episode that has should end rather than run to time.
    max_path_error: float = 3.0
    #: Follower starting offset from its slot (metres, behind), for a settling
    #: transient at t=0 that is identical across policies.
    start_offset: float = 0.0
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        self.path = PathConfig.parse(self.path)
        self.controller = ComponentConfig.parse(self.controller, 'analytic')
        self.policy = ComponentConfig.parse(self.policy, 'always')
        self.predictor = ComponentConfig.parse(self.predictor, 'constant_velocity')
        if isinstance(self.noise, dict):
            self.noise = ProcessNoise(**self.noise)
        if isinstance(self.limits, dict):
            self.limits = MotionLimits(**self.limits)
        if isinstance(self.contract, dict):
            self.contract = ContractConfig(**self.contract)
        if isinstance(self.leader, dict):
            self.leader = LeaderConfig(**self.leader)
        if isinstance(self.reward, dict):
            self.reward = RewardConfig(**self.reward)
        if self.dt <= 0.0:
            raise ValueError('dt must be positive')
        if self.duration <= 0.0:
            raise ValueError('duration must be positive')
        if self.offset_d <= 0.0:
            raise ValueError('offset_d must be positive')

    @property
    def steps(self):
        """Number of control steps in the episode."""
        return int(round(self.duration / self.dt))

    def with_overrides(self, **overrides):
        """Return a copy with top-level fields replaced (e.g. seed, policy)."""
        return replace(copy.deepcopy(self), **overrides)

    def to_dict(self):
        return asdict(self)

    def to_yaml(self, path):
        with open(path, 'w') as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    @classmethod
    def from_dict(cls, data):
        return cls(**(data or {}))

    @classmethod
    def from_yaml(cls, path):
        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle))


@dataclass
class SuiteConfig:
    """A set of episodes run over fixed seeds and aggregated.

    The evaluation suite and the stress suite are both this, with different
    seeds and episode settings.
    """

    name: str = 'eval'
    seeds: List[int] = field(default_factory=lambda: list(range(8)))
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    description: str = ''

    def __post_init__(self):
        if isinstance(self.episode, dict):
            self.episode = EpisodeConfig.from_dict(self.episode)
        self.seeds = [int(s) for s in self.seeds]
        if not self.seeds:
            raise ValueError('a suite needs at least one seed')

    def episodes(self, **overrides):
        """Yield one :class:`EpisodeConfig` per seed."""
        for seed in self.seeds:
            yield self.episode.with_overrides(seed=seed, **overrides)

    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'seeds': list(self.seeds),
            'episode': self.episode.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**(data or {}))

    @classmethod
    def from_yaml(cls, path):
        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle))
