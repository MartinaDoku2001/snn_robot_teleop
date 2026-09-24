"""A thin Gymnasium wrapper over :class:`FastFormationEnv`.

``FastFormationEnv`` already mirrors Gymnasium's ``reset``/``step`` signatures,
so this really is thin: it attaches real ``gymnasium.spaces`` objects and
nothing else. No reward shaping, no observation rewriting, no frame stacking --
anything done here would be something the Gazebo backend and the analytic
controller do not see, and the whole point of the shared contract is that all
three see the same thing.

``formation_core`` deliberately does not depend on gymnasium (it ships its own
minimal ``Box``), which is why this lives in ``formation_rl``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from formation_core.contract import ACTION_DIM, OBS_DIM
from formation_core.env import FastFormationEnv


class FormationGymEnv(gym.Env):
    """Gymnasium view of one formation episode on the fast twin."""

    metadata = {'render_modes': []}

    def __init__(self, config, seed=None):
        self.config = config
        self.env = FastFormationEnv(config)
        self._seed = seed
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=self._seed if seed is None else seed,
                                   options=options)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (np.asarray(obs, dtype=np.float32), float(reward),
                bool(terminated), bool(truncated), info)

    def close(self):
        self.env.close()


def make_env(config, seed):
    """A thunk for Gymnasium's vector wrappers, which want a zero-arg factory."""
    def _init():
        return FormationGymEnv(config, seed=seed)
    return _init
