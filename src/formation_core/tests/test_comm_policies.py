"""Transmission policies and the communication interface.

These tests pin down the semantics the whole comparison rests on: what a
"transmission rate" means, what the estimate contains between messages, and
that the event trigger really bounds the estimate error.
"""

import numpy as np
import pytest

from formation_core.comm import Channel, CommInterface, PerfectChannel
from formation_core.dynamics import MotionLimits, unicycle_step
from formation_core.geometry import RobotState
from formation_core.policies import (
    AlwaysTransmit,
    EventTriggered,
    PeriodicTransmit,
    RandomTransmit,
    make_policy,
)
from formation_core.predictor import ConstantVelocityPredictor

DT = 0.05
LIMITS = MotionLimits()


def _turning_leader(steps=200, w=0.6):
    """A leader that keeps turning, so dead reckoning steadily drifts."""
    state = RobotState(0.0, 0.0, 0.0, v=0.6, w=0.0)
    out = []
    for _ in range(steps):
        state = unicycle_step(state, 0.6, w, DT, LIMITS)
        out.append(state.copy())
    return out


def _run(policy, states, heading_weight=0.0):
    comm = CommInterface(
        policy=policy, predictor=ConstantVelocityPredictor(), dt=DT,
        heading_weight=heading_weight)
    comm.reset(states[0], np.random.default_rng(0))
    results = []
    for state in states:
        results.append(comm.update(state))
    return comm, results


def test_always_transmits_every_step_and_estimate_is_exact():
    states = _turning_leader()
    comm, results = _run(AlwaysTransmit(), states)
    assert comm.stats.rate == 1.0
    assert all(r.transmitted and r.received for r in results)
    assert all(r.age == 0 for r in results)
    for state, result in zip(states, results):
        assert np.hypot(state.x - result.estimate.x, state.y - result.estimate.y) < 1e-12


def test_periodic_transmits_on_every_kth_step():
    states = _turning_leader(steps=100)
    comm, results = _run(PeriodicTransmit(k=5), states)
    sent = [i for i, r in enumerate(results) if r.transmitted]
    assert sent == list(range(0, 100, 5))
    assert comm.stats.rate == pytest.approx(0.2)
    assert comm.stats.inter_transmission_intervals == [5] * (len(sent) - 1)


def test_periodic_k1_matches_always():
    states = _turning_leader(steps=50)
    _, periodic = _run(PeriodicTransmit(k=1), states)
    _, always = _run(AlwaysTransmit(), states)
    assert [r.transmitted for r in periodic] == [r.transmitted for r in always]


def test_random_rate_matches_p_and_is_seed_deterministic():
    states = _turning_leader(steps=4000)
    comm_a, results_a = _run(RandomTransmit(p=0.3), states)
    comm_b, results_b = _run(RandomTransmit(p=0.3), states)
    assert comm_a.stats.rate == pytest.approx(0.3, abs=0.03)
    # Same seed in, same decisions out.
    assert [r.transmitted for r in results_a] == [r.transmitted for r in results_b]


def test_event_triggered_bounds_the_estimate_error():
    """The trigger must keep prediction error near delta, not merely reduce it."""
    delta = 0.05
    states = _turning_leader(steps=400)
    comm, results = _run(EventTriggered(delta=delta), states)
    assert 0.0 < comm.stats.rate < 1.0
    errors = np.array([
        np.hypot(s.x - r.estimate.x, s.y - r.estimate.y)
        for s, r in zip(states, results)])
    # After a transmission the error is 0; between them it may reach delta plus
    # at most one step of drift before the next message corrects it.
    assert errors.max() <= delta + 0.6 * DT + 1e-9


def test_event_triggered_is_free_on_perfectly_predictable_motion():
    """A steady turn is EXACTLY dead-reckonable, so it should cost ~nothing.

    This is why the reference path mixes segments and the leader carries
    process noise: without them a constant-velocity predictor would need
    almost no messages anywhere, and the task would be trivial.
    """
    states = _turning_leader(steps=300, w=0.6)
    comm, results = _run(EventTriggered(delta=0.01), states)
    # Only the mandatory first message, plus a few while the yaw rate ramps up
    # to its commanded value under the acceleration limit.
    assert comm.stats.transmissions <= 5
    assert comm.stats.rate < 0.02


def test_event_triggered_delta_zero_transmits_whenever_prediction_is_imperfect():
    # Yaw rate changes every step, so dead reckoning is never exact.
    states = []
    state = RobotState(0.0, 0.0, 0.0, v=0.6, w=0.0)
    for i in range(60):
        state = unicycle_step(state, 0.6, 1.5 * np.sin(0.3 * i), DT, LIMITS)
        states.append(state.copy())
    comm, _ = _run(EventTriggered(delta=0.0), states)
    assert comm.stats.rate == 1.0


def test_event_triggered_spends_messages_at_curvature_changes():
    """Messages must concentrate where the generic predictor actually fails.

    A leader that drives straight, then turns, then straightens: the predictor
    is exact within each segment and wrong at the two transitions, so that is
    where the trigger should fire.
    """
    state = RobotState(0.0, 0.0, 0.0, v=0.6, w=0.0)
    states = []
    for i in range(300):
        yaw_rate = 1.0 if 100 <= i < 200 else 0.0
        state = unicycle_step(state, 0.6, yaw_rate, DT, LIMITS)
        states.append(state.copy())

    comm, results = _run(EventTriggered(delta=0.02), states)
    sent = [i for i, r in enumerate(results) if r.transmitted and i > 0]
    assert sent, 'the trigger never fired'
    # Every message lands just after a curvature change, never mid-segment.
    near_transition = [i for i in sent if 100 <= i <= 130 or 200 <= i <= 230]
    assert len(near_transition) == len(sent)
    assert comm.stats.rate < 0.2


def test_event_triggered_max_silence_caps_age():
    """A stationary leader is perfectly predictable; max_silence still refreshes."""
    states = [RobotState(0.0, 0.0, 0.0) for _ in range(100)]
    comm, results = _run(EventTriggered(delta=0.5, max_silence=10), states)
    assert max(r.age for r in results) <= 10


def test_age_of_information_counts_steps_since_refresh():
    states = _turning_leader(steps=20)
    _, results = _run(PeriodicTransmit(k=5), states)
    ages = [r.age for r in results]
    assert ages[:10] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_estimate_between_messages_is_the_prediction_not_the_truth():
    states = _turning_leader(steps=30, w=1.2)
    _, results = _run(PeriodicTransmit(k=30), states)  # only step 0 transmits
    later = results[-1]
    assert not later.transmitted
    # Truth has turned away; the estimate kept dead-reckoning.
    assert np.hypot(states[-1].x - later.estimate.x, states[-1].y - later.estimate.y) > 0.05


def test_reset_starts_from_truth_and_clears_stats():
    states = _turning_leader(steps=10)
    comm, _ = _run(PeriodicTransmit(k=3), states)
    assert comm.stats.steps == 10
    estimate = comm.reset(states[0])
    assert comm.stats.steps == 0 and comm.stats.transmissions == 0
    assert (estimate.x, estimate.y) == (states[0].x, states[0].y)


def test_heading_weight_folds_orientation_into_the_trigger():
    states = _turning_leader(steps=200)
    comm_pos, _ = _run(EventTriggered(delta=0.05), states, heading_weight=0.0)
    comm_head, _ = _run(EventTriggered(delta=0.05), states, heading_weight=1.0)
    # Counting heading error can only make the trigger fire at least as often.
    assert comm_head.stats.rate >= comm_pos.stats.rate


class _DelayChannel(Channel):
    """Phase-3 style channel, implemented here to prove the hook is usable."""

    def __init__(self, delay_steps):
        self.delay = delay_steps
        self.queue = []

    def reset(self, rng=None):
        self.queue = []

    def send(self, state, step, time):
        self.queue.append((step + self.delay, state.copy()))

    def deliver(self, step, time):
        due = [s for (arrival, s) in self.queue if arrival == step]
        self.queue = [(a, s) for (a, s) in self.queue if a != step]
        return due[-1] if due else None


def test_channel_hook_supports_delay_without_touching_policies():
    states = _turning_leader(steps=60)
    comm = CommInterface(
        policy=AlwaysTransmit(), predictor=ConstantVelocityPredictor(), dt=DT,
        channel=_DelayChannel(delay_steps=3))
    comm.reset(states[0])
    results = [comm.update(s) for s in states]
    # Everything is sent, but the first few steps have nothing to receive yet.
    assert all(r.transmitted for r in results)
    assert not results[0].received and results[5].received
    assert comm.stats.transmissions == 60
    assert comm.stats.receptions < comm.stats.transmissions


def test_perfect_channel_delivers_immediately():
    channel = PerfectChannel()
    channel.send(RobotState(1.0, 2.0, 0.0), 0, 0.0)
    delivered = channel.deliver(0, 0.0)
    assert (delivered.x, delivered.y) == (1.0, 2.0)
    assert channel.deliver(1, 0.05) is None


def test_policy_registry_and_descriptions():
    assert make_policy('periodic', k=4).describe() == 'periodic(k=4)'
    assert make_policy('event_triggered', delta=0.1).describe() == 'event_triggered(delta=0.1)'
    assert make_policy('always').describe() == 'always'
    with pytest.raises(ValueError):
        make_policy('oracle')
    with pytest.raises(ValueError):
        PeriodicTransmit(k=0)
    with pytest.raises(ValueError):
        RandomTransmit(p=1.5)
