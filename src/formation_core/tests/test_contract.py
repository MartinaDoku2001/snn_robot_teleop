"""The contract is frozen: these tests are the guard against silent changes."""

import numpy as np
import pytest

from formation_core.contract import (
    ACTION_DIM,
    ACTION_NAMES,
    AGE_NAMES,
    CONTRACT_VERSION,
    COUPLING_BLOCK,
    FOLLOWER_BLOCK,
    LEADER_BLOCK,
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

#: A straight path along +x through the origin, as (lookahead, tangent), for
#: tests that do not care where the path is.
STRAIGHT = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))


def _obs(leader, follower, path=STRAIGHT, leader_age=0, follower_age=0, offset_d=0.8):
    """build_observation with the argument order pinned in one place."""
    lookahead, tangent = path
    return build_observation(
        leader, follower, lookahead, tangent, leader_age, follower_age, offset_d, CFG)


def test_layout_is_frozen():
    """v2.0: one observation covering both robots, one action driving both."""
    assert OBS_DIM == 16
    assert ACTION_DIM == 4
    assert CONTRACT_VERSION == '2.0'
    assert OBS_NAMES == (
        'lead_look_x', 'lead_look_y', 'lead_sin_tangent', 'lead_cos_tangent',
        'lead_v', 'lead_w', 'lead_age',
        'foll_v', 'foll_w', 'foll_age', 'slot_ex', 'slot_ey',
        'rel_dx', 'rel_dy', 'rel_sin_dtheta', 'rel_cos_dtheta')
    assert ACTION_NAMES == ('lead_v', 'lead_w', 'foll_v', 'foll_w')


def test_the_blocks_partition_the_observation():
    """7 + 5 + 4 = 16, with no gap and no overlap."""
    covered = (list(range(*LEADER_BLOCK.indices(OBS_DIM)))
               + list(range(*FOLLOWER_BLOCK.indices(OBS_DIM)))
               + list(range(*COUPLING_BLOCK.indices(OBS_DIM))))
    assert covered == list(range(OBS_DIM))
    assert len(OBS_NAMES[LEADER_BLOCK]) == 7
    assert len(OBS_NAMES[FOLLOWER_BLOCK]) == 5
    assert len(OBS_NAMES[COUPLING_BLOCK]) == 4


def test_both_estimate_ages_are_present():
    """Mandatory: the one coupling point between control and communication.

    They read ~0 under perfect comms, which is why nothing else in Phase 2
    would notice if they went missing. Phase 4 cannot add them retroactively
    without invalidating every policy trained before, so they are pinned here.
    """
    assert AGE_NAMES == ('lead_age', 'foll_age')
    for name in AGE_NAMES:
        assert name in OBS_NAMES
    lead_idx = OBS_NAMES.index('lead_age')
    foll_idx = OBS_NAMES.index('foll_age')

    leader = RobotState(1.0, 0.0, 0.0, v=0.5)
    follower = RobotState(0.2, 0.0, 0.0, v=0.5)
    fresh = _obs(leader, follower, leader_age=0, follower_age=0)
    stale = _obs(leader, follower, leader_age=10, follower_age=4)

    assert fresh[lead_idx] == pytest.approx(0.0)
    assert fresh[foll_idx] == pytest.approx(0.0)
    assert stale[lead_idx] == pytest.approx(10.0 / CFG.age_max)
    assert stale[foll_idx] == pytest.approx(4.0 / CFG.age_max)


def test_age_saturates_rather_than_leaving_the_range():
    leader, follower = RobotState(1.0, 0.0, 0.0), RobotState(0.2, 0.0, 0.0)
    obs = _obs(leader, follower, leader_age=10_000, follower_age=10_000)
    assert obs[OBS_NAMES.index('lead_age')] == pytest.approx(1.0)
    assert obs[OBS_NAMES.index('foll_age')] == pytest.approx(1.0)


def test_spaces_are_normalized():
    obs_space = observation_space(CFG)
    act_space = action_space(CFG)
    assert obs_space.shape == (OBS_DIM,)
    assert act_space.shape == (ACTION_DIM,)
    assert np.all(obs_space.low == -1.0) and np.all(obs_space.high == 1.0)
    assert np.all(act_space.low == -1.0) and np.all(act_space.high == 1.0)


def test_observation_is_bounded_even_when_far_away():
    """Anything beyond max_range must saturate, never leave [-1, 1]."""
    follower = RobotState(0.0, 0.0, 0.0)
    leader = RobotState(500.0, -400.0, 3.0, v=99.0, w=-99.0)
    obs = _obs(leader, follower, path=(np.array([900.0, -300.0]), np.array([0.0, 1.0])),
               leader_age=999, follower_age=999)
    assert obs.shape == (OBS_DIM,)
    assert observation_space(CFG).contains(obs)
    assert np.all(np.abs(obs) <= 1.0)


def test_observation_is_relative_not_absolute():
    """Translating and rotating the whole scene must not change the observation."""
    follower = RobotState(1.0, 2.0, 0.3)
    leader = RobotState(1.8, 2.2, 0.5, v=0.6, w=0.1)
    lookahead = np.array([2.4, 2.5])
    tangent = np.array([np.cos(0.5), np.sin(0.5)])
    obs_a = _obs(leader, follower, path=(lookahead, tangent))

    angle = 1.1
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([-7.0, 4.0])
    moved_follower = RobotState(*(rot @ follower.xy + shift), follower.theta + angle)
    moved_leader = RobotState(
        *(rot @ leader.xy + shift), leader.theta + angle, leader.v, leader.w)
    obs_b = _obs(moved_leader, moved_follower,
                 path=(rot @ lookahead + shift, rot @ tangent))
    np.testing.assert_allclose(obs_a, obs_b, atol=1e-6)


def test_perfect_formation_gives_zero_slot_error():
    leader = RobotState(2.0, 1.0, np.pi / 4, v=0.5, w=0.0)
    slot = slot_position(leader, 0.8)
    follower = RobotState(slot[0], slot[1], np.pi / 4)
    decoded = decode_observation(_obs(leader, follower), CFG)
    np.testing.assert_allclose(decoded['slot_rel'], [0.0, 0.0], atol=1e-6)
    assert decoded['dtheta'] == pytest.approx(0.0, abs=1e-6)
    # The leader itself is still 0.8 m straight ahead.
    np.testing.assert_allclose(decoded['leader_rel'], [0.8, 0.0], atol=1e-6)


def test_lookahead_is_expressed_in_the_leader_frame():
    """The leader block is the leader's view, not the follower's."""
    leader = RobotState(0.0, 0.0, np.pi / 2, v=0.4)     # facing +y
    follower = RobotState(0.0, -0.8, np.pi / 2)
    # A point 1 m further along +y is 1 m straight AHEAD of the leader.
    decoded = decode_observation(
        _obs(leader, follower, path=(np.array([0.0, 1.0]), np.array([0.0, 1.0]))), CFG)
    np.testing.assert_allclose(decoded['lookahead_rel'], [1.0, 0.0], atol=1e-6)
    assert decoded['tangent_rel'] == pytest.approx(0.0, abs=1e-6)


def test_angles_use_sin_cos_without_wraparound():
    """Headings on either side of +-pi must give nearly identical observations."""
    follower = RobotState(0.0, 0.0, np.pi - 1e-4)
    leader_a = RobotState(1.0, 0.0, np.pi - 1e-4)
    leader_b = RobotState(1.0, 0.0, -np.pi + 1e-4)
    np.testing.assert_allclose(
        _obs(leader_a, follower), _obs(leader_b, follower), atol=1e-3)


def test_decode_round_trips_within_range():
    follower = RobotState(0.0, 0.0, 0.2, v=0.4, w=-0.3)
    leader = RobotState(1.2, 0.6, 0.5, v=0.7, w=0.2)
    decoded = decode_observation(_obs(leader, follower, leader_age=3, follower_age=7), CFG)
    assert decoded['v_follower_est'] == pytest.approx(0.4, abs=1e-5)
    assert decoded['w_follower_est'] == pytest.approx(-0.3, abs=1e-5)
    assert decoded['v_leader_est'] == pytest.approx(0.7, abs=1e-5)
    assert decoded['dtheta'] == pytest.approx(0.3, abs=1e-5)
    assert decoded['leader_age'] == pytest.approx(3.0, abs=1e-4)
    assert decoded['follower_age'] == pytest.approx(7.0, abs=1e-4)


def test_action_scaling_round_trip_and_clipping():
    leader, follower = scale_action([1.0, -1.0, -1.0, 1.0], CFG)
    assert leader == (CFG.v_max, -CFG.w_max)
    assert follower == (-CFG.v_max, CFG.w_max)
    assert scale_action([0.0, 0.0, 0.0, 0.0], CFG) == ((0.0, 0.0), (0.0, 0.0))
    # Out-of-range actions are clipped, never amplified past the limits.
    assert scale_action([5.0, -9.0, 5.0, -9.0], CFG)[0] == (CFG.v_max, -CFG.w_max)

    back = unscale_action((0.5, 1.0), (-0.25, 0.5), CFG)
    np.testing.assert_allclose(
        back,
        [0.5 / CFG.v_max, 1.0 / CFG.w_max, -0.25 / CFG.v_max, 0.5 / CFG.w_max],
        atol=1e-6)


def test_action_shape_is_validated():
    with pytest.raises(ValueError):
        scale_action([0.1, 0.2], CFG)          # a v1.0-shaped action
    with pytest.raises(ValueError):
        scale_action([0.1, 0.2, 0.3, 0.4, 0.5], CFG)


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
    for bad in ({'max_range': 0.0}, {'v_max': -1.0}, {'age_max': 0.0},
                {'lookahead_distance': -0.1}):
        with pytest.raises(ValueError):
            ContractConfig(**bad)
