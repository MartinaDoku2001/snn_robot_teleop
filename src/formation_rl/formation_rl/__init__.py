"""Learned centralized controller for the formation task (Phase 2d).

Importing this package registers the ``'rl'`` controller, so every existing
entry point selects it by name with no other change::

    python3 -m formation_core suite --controller rl
    ros2 launch formation_gazebo formation.launch.py controller:=rl

Layout, in the order the pieces matter:

* :mod:`formation_rl.actor`   -- the network. **The ANN -> SNN swap point.**
* :mod:`formation_rl.policy`  -- wraps an actor as a ``formation_core`` Controller.
* :mod:`formation_rl.ppo`     -- the training loop; imports the actor, does not define it.
* :mod:`formation_rl.gym_env` -- a thin Gymnasium view of ``FastFormationEnv``.

``formation_core`` never imports this package, and keeps no dependency on torch
or gymnasium.
"""

from formation_core.controllers import register_controller

from .policy import RLController

register_controller('rl', RLController)

__all__ = ['RLController']
