"""formation_core: task definition and evaluation harness for networked
leader-follower control.

Pure Python, no RL or robotics dependencies, so it runs anywhere and stays fast
enough to train in later. The public surface:

* :mod:`~formation_core.contract` -- the FROZEN observation/action contract.
* :mod:`~formation_core.env` -- ``FormationEnv`` (the one interface) and
  ``FastFormationEnv`` (the kinematic twin backend).
* :mod:`~formation_core.comm` -- communication interface, with the channel hook
  for Phase 3 delay/loss.
* :mod:`~formation_core.policies` -- transmission policies (always, periodic,
  random, event-triggered) behind one interface.
* :mod:`~formation_core.controllers` -- controller interface plus the analytic
  follower and the leader's pure-pursuit tracker.
* :mod:`~formation_core.metrics` -- control and communication metrics, CSV, and
  aggregation with confidence intervals.
* :mod:`~formation_core.runner` -- episode and suite runners.
"""

from .comm import Channel, CommInterface, PerfectChannel
from .config import ComponentConfig, EpisodeConfig, SuiteConfig
from .contract import (
    ACTION_DIM,
    CONTRACT_VERSION,
    OBS_DIM,
    OBS_NAMES,
    ContractConfig,
    action_space,
    build_observation,
    decode_observation,
    formation_errors,
    observation_space,
    scale_action,
)
from .controllers import AnalyticFollower, Controller, PurePursuitLeader, make_controller
from .env import FastFormationEnv, FormationEnv
from .geometry import RobotState
from .metrics import EpisodeRecorder, EpisodeResult, SuiteResult, mean_ci
from .paths import ClosedPath, make_path
from .policies import (
    AlwaysTransmit,
    EventTriggered,
    PeriodicTransmit,
    RandomTransmit,
    TransmissionPolicy,
    make_policy,
)
from .predictor import ConstantVelocityPredictor, Predictor, make_predictor
from .runner import run_episode, run_suite

__version__ = '0.1.0'

__all__ = [
    'ACTION_DIM',
    'CONTRACT_VERSION',
    'OBS_DIM',
    'OBS_NAMES',
    'AlwaysTransmit',
    'AnalyticFollower',
    'Channel',
    'ClosedPath',
    'CommInterface',
    'ComponentConfig',
    'ConstantVelocityPredictor',
    'ContractConfig',
    'Controller',
    'EpisodeConfig',
    'EpisodeRecorder',
    'EpisodeResult',
    'EventTriggered',
    'FastFormationEnv',
    'FormationEnv',
    'PerfectChannel',
    'PeriodicTransmit',
    'Predictor',
    'PurePursuitLeader',
    'RandomTransmit',
    'RobotState',
    'SuiteConfig',
    'SuiteResult',
    'TransmissionPolicy',
    'action_space',
    'build_observation',
    'decode_observation',
    'formation_errors',
    'make_controller',
    'make_path',
    'make_policy',
    'make_predictor',
    'mean_ci',
    'observation_space',
    'run_episode',
    'run_suite',
    'scale_action',
]
