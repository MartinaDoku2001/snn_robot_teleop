"""The contract is frozen: these tests are the guard against silent changes."""

import numpy as np
import pytest

from formation_core.contract import (
    ACTION_DIM,
    OBS_DIM,
    OBS_NAMES,
    ContractConfig,
    action_space,
    build_observation,
    decode_observation,
    formation_errors,
    observation_space,
    scale_action,
    slot_position,
    unscale_action,
)
from formation_core.geometry import RobotState

CFG = ContractConfig()


def test_layout_is_frozen():
    assert OBS_DIM == 10
    assert ACTION_DIM == 2
    assert OBS_NAMES == (
        'dx', 'dy', 'sin_dtheta', 'cos_dtheta', 'ex', 'ey',
        'v_self', 'w_self', 'v_leader_est', 'w_leader_est')


def test_spaces_are_normalized():
    obs_space = observation_space(CFG)
    act_space = action_space(CFG)
    assert obs_space.shape == (OBS_DIM,)
    assert act_space.shape == (ACTION_DIM,)
    assert np.all(obs_space.low == -1.0) and np.all(obs_space.high == 1.0)
    assert np.all(act_space.low == -1.0) and np.all(act_space.high == 1.0)


def test_observation_is_bounded_even_when_far_away():
    """A leader beyond max_range must saturate, never leave [-1, 1]."""
    follower = RobotState(0.0, 0.0, 0.0)
    leader = RobotState(500.0, -400.0, 3.0, v=99.0, w=-99.0)
    obs = build_observation(follower, leader, offset_d=0.8, config=CFG)
    assert obs.shape == (OBS_DIM,)
    assert observation_space(CFG).contains(obs)
    assert np.all(np.abs(obs) <= 1.0)


def test_observation_is_relative_not_absolute():
    """Translating and rotating the whole scene must not change the observation."""
    follower = RobotState(1.0, 2.0, 0.3)
    leader = RobotState(1.8, 2.2, 0.5, v=0.6, w=0.1)
    obs_a = build_observation(follower, leader, 0.8, CFG)

    angle = 1.1
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([-7.0, 4.0])
    moved_follower = RobotState(*(rot @ follower.xy + shift), follower.theta + angle)
    moved_leader = RobotState(
        *(rot @ leader.xy + shift), leader.theta + angle, leader.v, leader.w)
    obs_b = build_observation(moved_follower, moved_leader, 0.8, CFG)
    np.testing.assert_allclose(obs_a, obs_b, atol=1e-6)


def test_perfect_formation_gives_zero_slot_error():
    leader = RobotState(2.0, 1.0, np.pi / 4, v=0.5, w=0.0)
    slot = slot_position(leader, 0.8)
    follower = RobotState(slot[0], slot[1], np.pi / 4)
    obs = build_observation(follower, leader, 0.8, CFG)
    decoded = decode_observation(obs, CFG)
    np.testing.assert_allclose(decoded['slot_rel'], [0.0, 0.0], atol=1e-6)
    assert decoded['dtheta'] == pytest.approx(0.0, abs=1e-6)
    # The leader itself is still 0.8 m straight ahead.
    np.testing.assert_allclose(decoded['leader_rel'], [0.8, 0.0], atol=1e-6)


def test_angles_use_sin_cos_without_wraparound():
    """Headings on either side of +-pi must give nearly identical observations."""
    follower = RobotState(0.0, 0.0, np.pi - 1e-4)
    leader_a = RobotState(1.0, 0.0, np.pi - 1e-4)
    leader_b = RobotState(1.0, 0.0, -np.pi + 1e-4)
    obs_a = build_observation(follower, leader_a, 0.8, CFG)
    obs_b = build_observation(follower, leader_b, 0.8, CFG)
    np.testing.assert_allclose(obs_a, obs_b, atol=1e-3)


def test_decode_round_trips_within_range():
    follower = RobotState(0.0, 0.0, 0.2, v=0.4, w=-0.3)
    leader = RobotState(1.2, 0.6, 0.5, v=0.7, w=0.2)
    obs = build_observation(follower, leader, 0.8, CFG)
    decoded = decode_observation(obs, CFG)
    assert decoded['v_self'] == pytest.approx(0.4, abs=1e-5)
    assert decoded['w_self'] == pytest.approx(-0.3, abs=1e-5)
    assert decoded['v_leader_est'] == pytest.approx(0.7, abs=1e-5)
    assert decoded['dtheta'] == pytest.approx(0.3, abs=1e-5)


def test_action_scaling_round_trip_and_clipping():
    assert scale_action([1.0, -1.0], CFG) == (CFG.v_max, -CFG.w_max)
    assert scale_action([0.0, 0.0], CFG) == (0.0, 0.0)
    # Out-of-range actions are clipped, never amplified past the limits.
    assert scale_action([5.0, -9.0], CFG) == (CFG.v_max, -CFG.w_max)
    back = unscale_action(0.5, 1.0, CFG)
    np.testing.assert_allclose(back, [0.5 / CFG.v_max, 1.0 / CFG.w_max], atol=1e-6)


def test_action_shape_is_validated():
    with pytest.raises(ValueError):
        scale_action([0.1, 0.2, 0.3], CFG)


def test_formation_error_signs():
    """Errors point from the follower TO the slot (the correction owed)."""
    leader = RobotState(0.0, 0.0, 0.0, v=0.5)
    slot = slot_position(leader, 0.8)          # (-0.8, 0)
    behind = RobotState(slot[0] - 0.2, slot[1], 0.0)
    errors = formation_errors(behind, leader, 0.8)
    assert errors['longitudinal'] == pytest.approx(0.2)
    assert errors['lateral'] == pytest.approx(0.0, abs=1e-9)
    assert errors['euclidean'] == pytest.approx(0.2)

    # Follower drifted to the LEFT of the slot => the slot lies to its right
    # => lateral error is negative.
    left_of_slot = RobotState(slot[0], slot[1] + 0.3, 0.0)
    errors = formation_errors(left_of_slot, leader, 0.8)
    assert errors['lateral'] == pytest.approx(-0.3)
    assert errors['euclidean'] == pytest.approx(0.3)

    right_of_slot = RobotState(slot[0], slot[1] - 0.3, 0.0)
    assert formation_errors(right_of_slot, leader, 0.8)['lateral'] == pytest.approx(0.3)


def test_slot_is_behind_the_leader_in_its_own_frame():
    leader = RobotState(0.0, 0.0, np.pi / 2)   # facing +y
    slot = slot_position(leader, 1.0)
    np.testing.assert_allclose(slot, [0.0, -1.0], atol=1e-9)


def test_contract_config_rejects_bad_limits():
    with pytest.raises(ValueError):
        ContractConfig(max_range=0.0)
    with pytest.raises(ValueError):
        ContractConfig(v_max=-1.0)
