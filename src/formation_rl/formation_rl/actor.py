"""The actor network. **This module is the ANN -> SNN swap point.**

Everything else in this package -- the PPO trainer, the controller that plugs
into ``formation_core``, the ROS nodes -- talks to an actor through one method::

    action = actor.act(obs)      # (16,) float in [-1, 1]  ->  (4,) in [-1, 1]

A population-coded spiking actor (PopSAN-style, for SpiNNaker) replaces
:class:`MlpActor` by implementing that method and nothing else. Nothing in
training, evaluation or the ROS graph needs to change, and a spiking actor can
be dropped into a run that was trained here by loading the same weights into an
ANN-to-SNN conversion.

**Architecture constraints, deliberate and load-bearing.** The actor is a plain
feedforward MLP:

* **no recurrence** -- a spiking conversion of an RNN needs the network to be
  unrolled in time, and the SpiNNaker deployment has no such machinery;
* **no attention** -- same reason, plus it is meaningless at 16 inputs;
* **no batch or layer normalization** -- normalization statistics do not
  survive conversion to spike rates, and would silently change what the
  network computes on hardware;
* **bounded inputs and outputs** -- the contract already guarantees inputs in
  [-1, 1], and the output head is a tanh, which is what a population decoder
  reproduces.

Keeping to those rules costs nothing here (the task is 16 inputs and 4 outputs)
and is the difference between a Phase 2 policy that can be converted later and
one that has to be retrained from scratch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from formation_core.contract import ACTION_DIM, OBS_DIM

#: Hidden layer sizes. Two layers of 64 is the smallest network that solved the
#: task reliably in trials; population coding spends neurons per unit, so the
#: spiking budget is a reason to keep this small rather than to grow it.
DEFAULT_HIDDEN = (64, 64)


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """A plain feedforward stack. No normalization layers -- see the module doc."""
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class MlpActor(nn.Module):
    """Deterministic policy network: 16 observations -> 4 normalized commands.

    The output head is a tanh, so the action is already in the contract's
    [-1, 1] range and :func:`formation_core.contract.scale_action` maps it to
    the velocity limits. The network never sees or produces physical units.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=DEFAULT_HIDDEN):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.net = mlp((self.obs_dim, *self.hidden, self.action_dim),
                       activation=nn.Tanh, output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)

    # ------------------------------------------------------------ swap point

    @torch.no_grad()
    def act(self, obs):
        """``(16,) -> (4,)`` in [-1, 1]. The ONLY method a spiking actor needs.

        Accepts and returns numpy, so callers never import torch.
        """
        tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        return self.forward(tensor).squeeze(0).numpy().astype(np.float32)

    # --------------------------------------------------------- serialization

    @property
    def spec(self):
        """Everything needed to rebuild this network before loading weights."""
        return {'obs_dim': self.obs_dim, 'action_dim': self.action_dim,
                'hidden': list(self.hidden)}

    def save(self, path):
        torch.save({'spec': self.spec, 'state_dict': self.state_dict()}, path)
        return path

    @classmethod
    def load(cls, path, map_location='cpu'):
        payload = torch.load(path, map_location=map_location)
        actor = cls(**payload['spec'])
        actor.load_state_dict(payload['state_dict'])
        actor.eval()
        return actor
