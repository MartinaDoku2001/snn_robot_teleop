"""The learned controller, in the shape ``formation_core`` expects.

:class:`RLController` is a :class:`formation_core.controllers.Controller`: it
takes the frozen 16-dim observation and returns 4 normalized numbers. All it
does is hold an actor and forward to :meth:`actor.act`, which is what keeps the
actor swappable for a spiking network -- see :mod:`formation_rl.actor`.

Importing ``formation_rl`` registers this under the name ``'rl'``, so
everything that already selects a controller by name works unchanged::

    python3 -m formation_core suite --controller rl        # fast twin
    ros2 launch formation_gazebo formation.launch.py controller:=rl
"""

from __future__ import annotations

import os

import numpy as np

from formation_core.contract import ACTION_DIM
from formation_core.controllers import Controller

#: Where a trained actor is looked up when no explicit path is given. Keeping a
#: default means the ROS launch file does not need a weights argument for the
#: common case.
DEFAULT_WEIGHTS = os.environ.get(
    'FORMATION_RL_WEIGHTS', 'results/rl/actor.pt')


class RLController(Controller):
    """Runs a trained actor as the centralized controller.

    Args:
        config: :class:`~formation_core.contract.ContractConfig` (unused by the
            network, which works entirely in normalized units, but accepted so
            the constructor matches every other controller).
        weights: path to an actor saved by :meth:`MlpActor.save`.
        actor: an already-constructed actor, which takes precedence over
            ``weights``. This is the seam a spiking actor comes in through.
    """

    def __init__(self, config=None, weights=None, actor=None):
        self.config = config
        self.weights = weights or DEFAULT_WEIGHTS
        if actor is not None:
            self.actor = actor
            self.weights = getattr(actor, 'source_path', self.weights)
        else:
            self.actor = self._load(self.weights)

    @staticmethod
    def _load(path):
        # Imported lazily so that merely importing formation_rl -- which the
        # controller registry does -- never requires torch to be installed.
        from .actor import MlpActor

        if not os.path.exists(path):
            raise FileNotFoundError(
                f'no trained actor at {path!r}. Train one with:\n'
                f'    python3 -m formation_rl train --out results/rl\n'
                f'or point the controller at one with '
                f"controller params {{'weights': '<path>'}}.")
        actor = MlpActor.load(path)
        actor.source_path = path
        return actor

    def act(self, obs):
        action = np.asarray(self.actor.act(obs), dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f'actor returned shape {action.shape}, expected ({ACTION_DIM},)')
        return np.clip(action, -1.0, 1.0)

    @property
    def name(self):
        return 'rl'

    @property
    def params(self):
        # The weights path identifies WHICH policy ran, which is the one thing
        # a results CSV needs in order to be reproducible.
        return {'weights': os.path.basename(self.weights)}
