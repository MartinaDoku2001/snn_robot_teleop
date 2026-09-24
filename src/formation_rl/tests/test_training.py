"""The Gymnasium wrapper and the PPO loop.

These are correctness checks, not quality checks: that the wrapper reports the
contract's spaces, that a training run produces a usable actor, and that the
reward the agent optimizes cannot be gamed by crashing. Whether the learned
policy is any GOOD is a benchmark, not a unit test -- see
``python3 -m formation_rl benchmark``.
"""

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('gymnasium')

from formation_core.config import EpisodeConfig  # noqa: E402
from formation_core.contract import ACTION_DIM, OBS_DIM  # noqa: E402


def _tiny_config(**overrides):
    return EpisodeConfig(duration=3.0).with_overrides(**overrides)


def test_gym_wrapper_exposes_the_contract_spaces():
    from formation_rl.gym_env import FormationGymEnv

    env = FormationGymEnv(_tiny_config())
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACTION_DIM,)
    assert np.all(env.observation_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)

    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    # The wrapper adds nothing to info: the backends must stay comparable.
    assert 'leader_state' in info and 'follower_estimate' in info


def test_crashing_is_worse_than_surviving():
    """Without the terminal penalty, PPO learns to drive off the path on purpose.

    Every reward term is negative, so an episode that ends early accumulates
    less total penalty than one that holds formation to the end. This pins the
    fix: diverging must cost more than finishing.
    """
    from formation_rl.gym_env import FormationGymEnv

    config = _tiny_config(duration=20.0)
    assert config.reward.terminal_penalty > 0.0

    # Drive the leader flat out in a straight line -- it leaves the oval at the
    # first cap -- while the follower sits still, so the formation breaks too.
    # (Full throttle WITH full yaw rate would just circle, and circling on the
    # spot stays inside both tolerances.)
    env = FormationGymEnv(config)
    env.reset(seed=0)
    crash_return, steps = 0.0, 0
    for _ in range(config.steps):
        _, reward, terminated, truncated, _ = env.step(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        crash_return += reward
        steps += 1
        if terminated or truncated:
            break
    assert terminated, 'the deliberate-crash policy should end the episode early'

    from formation_core.runner import run_episode
    survived = run_episode(config)
    survive_return = float(np.nansum(survived.recorder.column('reward')))

    assert survive_return > crash_return, (
        f'crashing after {steps} steps returned {crash_return:.1f}, '
        f'holding formation returned {survive_return:.1f}')


def test_training_produces_a_usable_actor(tmp_path):
    """One tiny run, end to end: the loop runs and the actor it returns works."""
    from formation_rl.ppo import PPOConfig, train

    ppo = PPOConfig(total_steps=256, num_envs=2, rollout_steps=32,
                    minibatches=2, update_epochs=1, seed=0)
    actor, history = train(_tiny_config(), ppo=ppo)

    assert history and history[-1]['steps'] >= ppo.batch_size
    action = actor.act(np.zeros(OBS_DIM, dtype=np.float32))
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0)

    # ... and it can immediately drive the real env through the normal path.
    from formation_core.runner import run_episode
    from formation_rl.policy import RLController

    result = run_episode(_tiny_config(), controller=RLController(actor=actor))
    assert result.metrics['steps'] > 0
    assert np.isfinite(result.metrics['formation_rms'])


def test_vector_env_gives_each_worker_a_different_episode_seed():
    """Otherwise every worker sees the same leader noise and the policy overfits."""
    from formation_rl.ppo import VectorEnv

    envs = VectorEnv(_tiny_config(), num_envs=4, seeds=list(range(16)))
    first = envs.reset()
    # Four workers, four different starting observations is too weak a check --
    # the task resets identically -- so compare the seeds actually handed out.
    assert len(set(envs._cursor)) == 4
    assert first.shape == (4, OBS_DIM)


def test_standing_still_is_worse_than_driving():
    """The other way PPO can cheat: refuse to move.

    Every error term is a penalty, so parking both robots on the path in
    perfect formation scores ~0 and beats actually traversing it. Under v1.x
    this was impossible -- the scripted leader always drove and the follower
    had to keep up -- and centralizing the controller is what removed the thing
    forcing motion. ``w_progress`` puts it back.
    """
    from formation_core.config import EpisodeConfig
    from formation_core.controllers import ZeroController
    from formation_core.runner import run_episode

    config = EpisodeConfig(duration=20.0)
    assert config.reward.w_progress > 0.0

    parked = run_episode(config, controller=ZeroController())
    driving = run_episode(config)          # the analytic controller

    parked_return = float(np.nansum(parked.recorder.column('reward')))
    driving_return = float(np.nansum(driving.recorder.column('reward')))
    assert driving_return > parked_return, (
        f'parking returned {parked_return:.1f}, driving returned '
        f'{driving_return:.1f} -- the reward still pays for doing nothing')

    # And the reason: the parked leader covers no ground.
    assert np.abs(parked.recorder.column('progress')).sum() < 0.1
    assert np.abs(driving.recorder.column('progress')).sum() > 5.0
