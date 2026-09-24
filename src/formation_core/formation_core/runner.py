"""Episode and suite runners.

The control loop here is backend-agnostic: it takes any
:class:`~formation_core.env.FormationEnv` and any
:class:`~formation_core.controllers.Controller`. The ROS evaluation node uses
the same loop body, so an episode measured in the fast twin and one measured in
Gazebo are produced by identical code paths.
"""

from __future__ import annotations

import os

from .config import EpisodeConfig, SuiteConfig
from .contract import CONTRACT_VERSION
from .controllers import make_controller
from .env import FastFormationEnv
from .metrics import EpisodeRecorder, EpisodeResult, SuiteResult


def build_env(config):
    """Construct the fast twin for ``config``."""
    return FastFormationEnv(config)


def build_controller(config):
    """Construct the centralized controller named in ``config``.

    ``leader`` carries the task's speed settings, which a controller driving the
    leader needs and which are not part of the observation. It reaches only
    controllers that ask for it (see ``Controller.wants_leader_config``).
    """
    return make_controller(
        config.controller.name, config=config.contract, leader=config.leader,
        **config.controller.params)


def run_episode(config, env=None, controller=None, recorder=None, label=''):
    """Run one episode and return an :class:`EpisodeResult`.

    Args:
        config: :class:`~formation_core.config.EpisodeConfig`.
        env: optional pre-built env (defaults to the fast twin). Pass the
            Gazebo env here to measure the same episode on the real simulator.
        controller: optional pre-built controller.
        recorder: optional :class:`EpisodeRecorder` to append into.
        label: short name used in plots and summary rows.
    """
    env = build_env(config) if env is None else env
    controller = build_controller(config) if controller is None else controller
    recorder = EpisodeRecorder(dt=config.dt) if recorder is None else recorder
    recorder.metadata.setdefault('contract_version', CONTRACT_VERSION)
    recorder.metadata.setdefault('seed', config.seed)

    controller.reset()
    obs, info = env.reset(seed=config.seed)

    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.record(info)

    metrics = recorder.metrics(comm_stats=getattr(env, 'comm_stats', None))
    metrics['terminated_early'] = int(terminated)
    # The BUILT policy, so a parameter left to its default is still recorded.
    # config.policy.describe() would write a bare 'random' for RandomTransmit(p=0.5).
    policy = getattr(env, 'policy', None)
    metrics['policy'] = (
        policy.describe() if policy is not None else config.policy.describe())
    metrics['controller'] = config.controller.describe()
    metrics['predictor'] = config.predictor.describe()
    metrics['path'] = config.path.name
    return EpisodeResult(
        config=config,
        metrics=metrics,
        recorder=recorder,
        terminated=terminated,
        truncated=truncated,
        label=label or metrics['policy'],
    )


def run_suite(suite, label='', overrides=None, env_factory=None, progress=None):
    """Run every seed in ``suite`` and aggregate.

    Args:
        suite: :class:`~formation_core.config.SuiteConfig`.
        label: name for this configuration in plots (defaults to the policy).
        overrides: dict of :class:`EpisodeConfig` fields to replace per episode
            (e.g. ``{'policy': ComponentConfig('periodic', {'k': 4})}``).
        env_factory: optional ``config -> env`` hook, for running a suite on a
            different backend.
        progress: optional ``callable(index, total, result)`` for reporting.
    """
    overrides = overrides or {}
    results = SuiteResult(name=suite.name, label=label)
    configs = list(suite.episodes(**overrides))
    for index, config in enumerate(configs):
        env = env_factory(config) if env_factory is not None else None
        result = run_episode(config, env=env, label=label)
        results.episodes.append(result)
        if progress is not None:
            progress(index + 1, len(configs), result)
    if not results.label and results.episodes:
        results.label = results.episodes[0].label
    return results


def load_config(path):
    """Load an :class:`EpisodeConfig` or :class:`SuiteConfig` from YAML.

    A document with a ``seeds`` list is a suite; anything else is an episode.
    """
    import yaml

    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    if 'seeds' in data or 'episode' in data:
        return SuiteConfig.from_dict(data)
    return EpisodeConfig.from_dict(data)


def config_path(name):
    """Path to a packaged config, e.g. ``config_path('eval_suite.yaml')``."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, 'configs', name)
