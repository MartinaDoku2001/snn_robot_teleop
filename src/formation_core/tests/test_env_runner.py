"""Environment semantics, determinism, and the backend-agnostic runner."""

import os
import tempfile

import numpy as np
import pytest

from formation_core.config import ComponentConfig, EpisodeConfig, SuiteConfig
from formation_core.contract import ACTION_DIM, OBS_DIM, observation_space
from formation_core.controllers import AnalyticCentralized, ZeroController
from formation_core.env import FastFormationEnv, FormationEnv
from formation_core.metrics import STEP_FIELDS
from formation_core.runner import build_controller, run_episode, run_suite


def short_config(**overrides):
    base = EpisodeConfig(duration=6.0)
    return base.with_overrides(**overrides) if overrides else base


def test_env_implements_the_shared_interface():
    env = FastFormationEnv(short_config())
    assert isinstance(env, FormationEnv)


def test_reset_and_step_mirror_gymnasium_signatures():
    env = FastFormationEnv(short_config())
    obs, info = env.reset(seed=3)
    assert observation_space(env.config.contract).contains(obs)
    assert 'leader_state' in info and 'transmitted' in info
    # v2.0: the follower has an uplink of its own.
    assert 'follower_estimate' in info and 'follower_age' in info

    out = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert observation_space(env.config.contract).contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_episode_truncates_at_configured_length():
    config = short_config()
    env = FastFormationEnv(config)
    env.reset(seed=0)
    controller = AnalyticCentralized(config.contract, leader=config.leader)
    steps = 0
    terminated = truncated = False
    obs, _ = env.reset(seed=0)
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = env.step(controller.act(obs))
        steps += 1
    assert truncated and not terminated
    assert steps == config.steps


def test_same_seed_is_bit_for_bit_reproducible():
    config = short_config()
    trajectories = []
    for _ in range(2):
        env = FastFormationEnv(config)
        obs, _ = env.reset(seed=11)
        controller = AnalyticCentralized(config.contract, leader=config.leader)
        rows = []
        for _ in range(60):
            obs, reward, _, _, info = env.step(controller.act(obs))
            rows.append(np.concatenate([obs, [reward, info['leader_state'].x]]))
        trajectories.append(np.array(rows))
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def test_different_seeds_differ():
    config = short_config()
    finals = []
    for seed in (0, 1):
        env = FastFormationEnv(config)
        obs, _ = env.reset(seed=seed)
        controller = AnalyticCentralized(config.contract, leader=config.leader)
        for _ in range(60):
            obs, _, _, _, info = env.step(controller.act(obs))
        finals.append(info['leader_state'].to_array())
    assert not np.allclose(finals[0], finals[1])


def test_policy_choice_does_not_disturb_the_leader():
    """Separate RNG streams: changing the policy must not change the leader.

    Under v2.0 the leader is commanded rather than scripted, so the commands
    are held fixed here; what is being tested is that a stochastic transmission
    policy never draws from the leader's noise stream.
    """
    base = short_config()
    command = np.array([0.4, 0.1, 0.4, 0.0], dtype=np.float32)
    leaders = []
    for policy in (ComponentConfig('always'), ComponentConfig('random', {'p': 0.3})):
        env = FastFormationEnv(base.with_overrides(policy=policy))
        env.reset(seed=5)
        track = []
        for _ in range(50):
            _, _, _, _, info = env.step(command)
            track.append(info['leader_state'].to_array())
        leaders.append(np.array(track))
    np.testing.assert_allclose(leaders[0], leaders[1], atol=1e-12)


def test_observation_is_built_from_the_estimates_not_ground_truth():
    """The core guarantee of the task: the controller never sees the truth.

    v2.0 checks BOTH uplinks: with transmission effectively switched off, each
    robot's estimate drifts away from its true state, and the observation has
    to follow the estimates.
    """
    from formation_core.contract import build_observation

    config = short_config(policy=ComponentConfig('periodic', {'k': 10_000}))
    env = FastFormationEnv(config)
    obs, _ = env.reset(seed=2)
    for _ in range(40):
        obs, _, _, _, info = env.step(np.array([0.5, 0.2, 0.3, 0.0], dtype=np.float32))

    def rebuild(leader, follower):
        lookahead, tangent = env.path.lookahead_pose(
            leader.xy, config.contract.lookahead_distance)
        return build_observation(
            leader, follower, lookahead, tangent,
            info['age'], info['follower_age'], config.offset_d, config.contract)

    np.testing.assert_allclose(
        obs, rebuild(info['leader_estimate'], info['follower_estimate']), atol=1e-9)
    # Both estimates really have drifted off their true states by now.
    assert info['estimate_error'] > 1e-3
    assert info['follower_estimate_error'] > 1e-3
    assert not np.allclose(
        obs, rebuild(info['leader_state'], info['follower_state']), atol=1e-6)


def test_actions_are_clipped_to_limits():
    config = short_config()
    env = FastFormationEnv(config)
    env.reset(seed=0)
    for _ in range(20):
        _, _, _, _, info = env.step(np.array([50.0, -50.0, 50.0, -50.0]))
    for robot in ('leader_state', 'follower_state'):
        assert abs(info[robot].v) <= config.limits.v_max + 1e-9
        assert abs(info[robot].w) <= config.limits.w_max + 1e-9


def test_analytic_controller_holds_formation_with_perfect_communication():
    result = run_episode(short_config(duration=30.0))
    assert result.metrics['formation_rms'] < 0.10
    assert result.metrics['formation_max'] < 0.30
    assert result.metrics['path_rms'] < 0.05          # leader tracks its path
    assert result.metrics['comm_rate'] == pytest.approx(1.0)


def test_degraded_communication_hurts_formation():
    """The task premise, in one assertion."""
    good = run_episode(short_config(duration=30.0, policy=ComponentConfig('always')))
    poor = run_episode(short_config(
        duration=30.0, policy=ComponentConfig('periodic', {'k': 40})))
    assert poor.metrics['formation_rms'] > good.metrics['formation_rms']
    assert poor.metrics['comm_rate'] < good.metrics['comm_rate']


def test_runner_accepts_any_backend_implementing_the_interface():
    """Proves the Gazebo backend can slot in without touching the runner."""

    class StubEnv(FormationEnv):
        def __init__(self, config):
            self.config = config
            self.observation_space = observation_space(config.contract)
            self.n = 0

        def reset(self, *, seed=None, options=None):
            self.n = 0
            return np.zeros(OBS_DIM, dtype=np.float32), self._info()

        def step(self, action):
            self.n += 1
            done = self.n >= 5
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, done, self._info()

        def _info(self):
            from formation_core.geometry import RobotState
            state = RobotState()
            return {
                'step': self.n, 'time': self.n * 0.05,
                'leader_state': state, 'follower_state': state, 'leader_estimate': state,
                'transmitted': True, 'received': True, 'age': 0,
                'prediction_error': 0.0, 'estimate_error': 0.0, 'path_error': 0.0,
                'errors': {'longitudinal': 0.0, 'lateral': 0.0,
                           'heading': 0.0, 'euclidean': 0.0},
                'action': np.zeros(ACTION_DIM), 'reward': 0.0,
            }

    config = short_config()
    result = run_episode(config, env=StubEnv(config), controller=ZeroController())
    assert result.metrics['steps'] == 5


def test_episode_csv_has_the_frozen_columns():
    result = run_episode(short_config(duration=2.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = result.write_csv(os.path.join(tmp, 'episode.csv'))
        with open(path) as handle:
            header = handle.readline().strip().split(',')
            rows = handle.readlines()
    assert tuple(header) == STEP_FIELDS
    assert len(rows) == result.metrics['steps']


def test_suite_runs_every_seed_and_aggregates_with_confidence_intervals():
    suite = SuiteConfig(
        name='test', seeds=[0, 1, 2], episode=short_config(duration=4.0))
    results = run_suite(suite, label='always')
    assert results.seeds == [0, 1, 2]
    aggregate = results.aggregate()
    assert aggregate['episodes'] == 3
    assert 'formation_rms_mean' in aggregate and 'formation_rms_ci95' in aggregate
    assert aggregate['formation_rms_ci95'] >= 0.0


def test_config_yaml_round_trip_and_immutability_of_overrides():
    config = EpisodeConfig(seed=4, policy=ComponentConfig('periodic', {'k': 3}))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'cfg.yaml')
        config.to_yaml(path)
        loaded = EpisodeConfig.from_yaml(path)
    assert loaded.to_dict() == config.to_dict()

    modified = config.with_overrides(seed=99)
    assert modified.seed == 99 and config.seed == 4     # original untouched
    modified.policy.params['k'] = 7
    assert config.policy.params['k'] == 3               # deep-copied


@pytest.mark.parametrize('name', ['default.yaml', 'eval_suite.yaml', 'stress_suite.yaml'])
def test_packaged_configs_load(name):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, 'configs', name)
    if 'suite' in name:
        suite = SuiteConfig.from_yaml(path)
        assert len(suite.seeds) >= 8                    # the >=8 episode requirement
        assert suite.episode.steps > 0
    else:
        assert EpisodeConfig.from_yaml(path).steps > 0


def test_stress_suite_is_actually_harder():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evaluation = SuiteConfig.from_yaml(os.path.join(here, 'configs', 'eval_suite.yaml'))
    stress = SuiteConfig.from_yaml(os.path.join(here, 'configs', 'stress_suite.yaml'))
    assert stress.episode.leader.target_speed > evaluation.episode.leader.target_speed
    assert stress.episode.noise.w_bound > evaluation.episode.noise.w_bound
    assert stress.episode.path.params['radius'] < evaluation.episode.path.params['radius']


def test_build_controller_from_config():
    config = short_config(controller=ComponentConfig('analytic', {'k_bearing': 3.0}))
    controller = build_controller(config)
    assert isinstance(controller, AnalyticCentralized)
    assert controller.follower.gains.k_bearing == 3.0
    # The leader's speed settings reach the controller without being observations.
    assert controller.target_speed == config.leader.target_speed


def test_the_centralized_controller_drives_both_robots():
    """v2.0's defining property: one act() call moves the leader too."""
    config = short_config()
    env = FastFormationEnv(config)
    obs, _ = env.reset(seed=0)
    controller = build_controller(config)
    action = controller.act(obs)
    assert np.asarray(action).shape == (ACTION_DIM,)
    for _ in range(40):
        obs, _, _, _, info = env.step(controller.act(obs))
    # The leader is being driven along the path by the controller, not scripted.
    assert info['leader_state'].v > 0.1
    assert info['path_error'] < 0.1
