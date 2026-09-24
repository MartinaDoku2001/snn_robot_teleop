"""The actor's architecture constraints, and the controller that wraps it.

The architecture rules are not style preferences: the actor becomes a
population-coded spiking network on SpiNNaker, and recurrence, attention or
normalization layers would make that conversion impossible or silently wrong.
They are asserted here because nothing else would notice them being broken
until conversion time, which is a phase away.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from formation_core.contract import ACTION_DIM, OBS_DIM  # noqa: E402
from formation_rl.actor import MlpActor  # noqa: E402


def test_actor_matches_the_frozen_contract():
    actor = MlpActor()
    assert actor.obs_dim == OBS_DIM == 16
    assert actor.action_dim == ACTION_DIM == 4
    out = actor.act(np.zeros(OBS_DIM, dtype=np.float32))
    assert out.shape == (ACTION_DIM,)


def test_actor_output_is_bounded_by_construction():
    """A tanh head, so no action can exceed the contract range whatever it sees."""
    actor = MlpActor()
    for scale in (0.0, 1.0, 50.0, -50.0):
        out = actor.act(np.full(OBS_DIM, scale, dtype=np.float32))
        assert np.all(np.abs(out) <= 1.0)


def test_actor_is_a_plain_feedforward_mlp():
    """No recurrence, no attention, no normalization -- the SNN constraints."""
    actor = MlpActor()
    allowed = (torch.nn.Linear, torch.nn.Tanh, torch.nn.Sequential, MlpActor)
    for module in actor.modules():
        assert isinstance(module, allowed), f'{type(module).__name__} is not SNN-convertible'

    forbidden = (
        torch.nn.RNNBase, torch.nn.LSTM, torch.nn.GRU,           # recurrence
        torch.nn.MultiheadAttention,                             # attention
        torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.GroupNorm,  # normalization
    )
    assert not any(isinstance(m, forbidden) for m in actor.modules())


def test_actor_is_small_enough_to_population_code():
    """Population coding spends neurons per unit; width is the budget."""
    actor = MlpActor()
    assert len(actor.hidden) <= 3
    assert max(actor.hidden) <= 128
    assert sum(p.numel() for p in actor.parameters()) < 20_000


def test_actor_is_deterministic():
    """What gets deployed is act(); exploration lives in the trainer, not here."""
    actor = MlpActor()
    obs = np.random.default_rng(0).uniform(-1, 1, OBS_DIM).astype(np.float32)
    np.testing.assert_array_equal(actor.act(obs), actor.act(obs))


def test_actor_round_trips_through_a_file(tmp_path):
    actor = MlpActor(hidden=(32, 32))
    obs = np.random.default_rng(1).uniform(-1, 1, OBS_DIM).astype(np.float32)
    before = actor.act(obs)

    path = actor.save(str(tmp_path / 'actor.pt'))
    restored = MlpActor.load(path)
    assert restored.hidden == (32, 32)          # the spec travels with the weights
    np.testing.assert_allclose(before, restored.act(obs), atol=1e-6)


# ------------------------------------------------------------------ controller

def test_rl_is_registered_as_a_controller():
    import formation_rl  # noqa: F401  -- importing is what registers it
    from formation_core.controllers import CONTROLLERS

    assert 'rl' in CONTROLLERS


def test_controller_forwards_to_the_actor_and_clips(tmp_path):
    from formation_rl.policy import RLController

    path = MlpActor().save(str(tmp_path / 'actor.pt'))
    controller = RLController(weights=path)
    obs = np.random.default_rng(2).uniform(-1, 1, OBS_DIM).astype(np.float32)
    action = controller.act(obs)
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0)
    assert controller.name == 'rl'


def test_a_spiking_actor_can_be_swapped_in_without_touching_anything_else():
    """The point of isolating the actor: any object with act() is a controller."""
    from formation_rl.policy import RLController

    class FakeSpikingActor:
        """Stands in for the PopSAN actor: same interface, no torch."""

        def act(self, obs):
            del obs
            return np.array([0.5, -0.25, 0.5, 0.0], dtype=np.float32)

    controller = RLController(actor=FakeSpikingActor())
    np.testing.assert_allclose(
        controller.act(np.zeros(OBS_DIM, dtype=np.float32)), [0.5, -0.25, 0.5, 0.0])


def test_missing_weights_say_how_to_make_them():
    from formation_rl.policy import RLController

    with pytest.raises(FileNotFoundError, match='formation_rl train'):
        RLController(weights='/nonexistent/actor.pt')
