"""Communication interface between the leader's true state and the follower.

One rule, applied every step:

* the policy says TRANSMIT  -> estimate := true state (age resets to 0)
* otherwise                 -> estimate := predictor.step(estimate)

The follower's controller only ever sees the resulting estimate. Ground truth
is used for two things and no others: deciding when to transmit (the sender
knows its own state) and computing evaluation metrics.

Phase 3 will add delay and packet loss. The hook is :class:`Channel`: it sits
between "the sender decided to transmit" and "the receiver applies the update",
so latency, loss and jitter land in one place without touching policies,
controllers or metrics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional

from .geometry import RobotState, state_error_norm
from .policies import PolicyContext


class Channel(abc.ABC):
    """Carries a transmitted state from sender to receiver.

    Phase 1 ships only :class:`PerfectChannel`. A delaying/dropping channel
    implements the same two methods and needs no other change.
    """

    @abc.abstractmethod
    def send(self, state, step, time):
        """Hand a state to the channel at ``step``."""

    @abc.abstractmethod
    def deliver(self, step, time):
        """Return the state delivered at ``step``, or None if nothing arrives."""

    def reset(self, rng=None):
        del rng

    @property
    def name(self):
        return type(self).__name__


class PerfectChannel(Channel):
    """Zero delay, no loss: what is sent this step arrives this step."""

    def __init__(self):
        self._pending = None

    def reset(self, rng=None):
        del rng
        self._pending = None

    def send(self, state, step, time):
        del step, time
        self._pending = state.copy()

    def deliver(self, step, time):
        del step, time
        delivered, self._pending = self._pending, None
        return delivered

    @property
    def name(self):
        return 'perfect'


@dataclass
class CommResult:
    """Outcome of one communication step."""

    estimate: RobotState
    transmitted: bool
    received: bool
    age: int
    prediction_error: float


@dataclass
class CommStats:
    """Per-episode communication statistics."""

    steps: int = 0
    transmissions: int = 0
    receptions: int = 0
    transmit_steps: List[int] = field(default_factory=list)
    age_history: List[int] = field(default_factory=list)

    @property
    def rate(self):
        """Fraction of steps on which a message was sent, in [0, 1]."""
        return self.transmissions / self.steps if self.steps else 0.0

    @property
    def inter_transmission_intervals(self):
        """Gaps (in steps) between consecutive transmissions."""
        steps = self.transmit_steps
        return [b - a for a, b in zip(steps[:-1], steps[1:])]


class CommInterface:
    """Owns the receiver's estimate of the sender's state.

    Args:
        policy: a :class:`~formation_core.policies.TransmissionPolicy`.
        predictor: a :class:`~formation_core.predictor.Predictor`.
        dt: timestep in seconds.
        channel: optional :class:`Channel` (defaults to :class:`PerfectChannel`).
        heading_weight: metres per radian when measuring estimate error; 0 by
            default, so an event-triggered ``delta`` reads as pure distance.
    """

    def __init__(self, policy, predictor, dt, channel=None, heading_weight=0.0):
        self.policy = policy
        self.predictor = predictor
        self.dt = float(dt)
        self.channel = channel if channel is not None else PerfectChannel()
        self.heading_weight = float(heading_weight)
        self.estimate = RobotState()
        self.age = 0
        self.stats = CommStats()
        self._step = 0
        self._last_transmit_step: Optional[int] = None

    def reset(self, true_state, rng=None):
        """Start an episode with the receiver knowing the truth exactly."""
        self.policy.reset(rng)
        self.predictor.reset(true_state)
        self.channel.reset(rng)
        self.estimate = true_state.copy()
        self.age = 0
        self.stats = CommStats()
        self._step = 0
        self._last_transmit_step = None
        return self.estimate

    def update(self, true_state, time=None, last_command=None):
        """Advance one communication step and return the new estimate.

        Call this exactly once per control step, BEFORE building the
        observation the controller acts on.
        """
        time = self._step * self.dt if time is None else float(time)

        # What the receiver would hold if nothing is sent this step.
        predicted = self.predictor.step(self.estimate, self.dt, last_command)
        prediction_error = state_error_norm(true_state, predicted, self.heading_weight)

        context = PolicyContext(
            step=self._step,
            time=time,
            true_state=true_state.copy(),
            predicted_estimate=predicted.copy(),
            prediction_error=prediction_error,
            age=self.age + 1,
            last_transmit_step=self._last_transmit_step,
        )
        transmitted = bool(self.policy.should_transmit(context))

        if transmitted:
            self.channel.send(true_state, self._step, time)
            self.stats.transmissions += 1
            self.stats.transmit_steps.append(self._step)
            self._last_transmit_step = self._step

        received = self.channel.deliver(self._step, time)
        if received is not None:
            self.estimate = received.copy()
            self.age = 0
            self.stats.receptions += 1
        else:
            self.estimate = predicted
            self.age += 1

        self.stats.steps += 1
        self.stats.age_history.append(self.age)
        self._step += 1

        return CommResult(
            estimate=self.estimate.copy(),
            transmitted=transmitted,
            received=received is not None,
            age=self.age,
            prediction_error=prediction_error,
        )
