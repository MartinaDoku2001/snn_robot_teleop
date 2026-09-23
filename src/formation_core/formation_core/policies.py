"""Transmission policies: decide, each step, whether the leader sends its state.

All policies implement :class:`TransmissionPolicy`, so a learned policy (the
eventual RL/SNN scheduler) plugs in with no change to the comm interface, the
controller, or the metrics.

The policy sees a :class:`PolicyContext` holding everything a scheduler could
reasonably use, including the quantities a learned policy would take as input
(the prediction error it could avoid, and the age of the current estimate).
A policy must NOT mutate the context.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import RobotState


@dataclass(frozen=True)
class PolicyContext:
    """Everything a transmission policy may condition on.

    Attributes:
        step: index of the current step, starting at 0.
        time: simulated time in seconds.
        true_state: the sender's true state this step.
        predicted_estimate: what the receiver would hold if NOTHING is sent
            this step (the predictor already advanced it).
        prediction_error: ``||true_state - predicted_estimate||`` in metres,
            i.e. the error this transmission would remove.
        age: steps since the estimate was last refreshed.
        last_transmit_step: index of the previous transmission, or None.
    """

    step: int
    time: float
    true_state: RobotState
    predicted_estimate: RobotState
    prediction_error: float
    age: int
    last_transmit_step: Optional[int] = None


class TransmissionPolicy(abc.ABC):
    """Decides whether to transmit at the current step."""

    @abc.abstractmethod
    def should_transmit(self, context):
        """Return True to send the true state this step."""

    def reset(self, rng=None):
        """Reset per-episode state. ``rng`` is a seeded ``np.random.Generator``."""
        del rng

    @property
    def name(self):
        return type(self).__name__

    @property
    def params(self):
        """Parameters recorded alongside results (for plot legends and CSVs)."""
        return {}

    def describe(self):
        params = self.params
        if not params:
            return self.name
        inner = ','.join(f'{k}={v}' for k, v in sorted(params.items()))
        return f'{self.name}({inner})'


class AlwaysTransmit(TransmissionPolicy):
    """Transmit every step. The reference point: best accuracy, rate 1.0."""

    def should_transmit(self, context):
        del context
        return True

    @property
    def name(self):
        return 'always'


class PeriodicTransmit(TransmissionPolicy):
    """Transmit every ``k``-th step (k=1 is equivalent to always).

    Open loop: it spends messages on straights and turns alike, which is
    exactly the baseline event-triggered transmission should beat.
    """

    def __init__(self, k=2):
        k = int(k)
        if k < 1:
            raise ValueError('k must be >= 1')
        self.k = k

    def should_transmit(self, context):
        return context.step % self.k == 0

    @property
    def name(self):
        return 'periodic'

    @property
    def params(self):
        return {'k': self.k}


class RandomTransmit(TransmissionPolicy):
    """Transmit with independent probability ``p`` each step (Bernoulli).

    Rate-matched control for periodic: same average cost, no timing structure.
    """

    def __init__(self, p=0.5):
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError('p must be in [0, 1]')
        self.p = p
        self._rng = np.random.default_rng()

    def reset(self, rng=None):
        # Own generator, seeded from the episode seed, so transmission draws
        # never disturb the leader-noise stream.
        self._rng = np.random.default_rng() if rng is None else rng

    def should_transmit(self, context):
        del context
        return bool(self._rng.random() < self.p)

    @property
    def name(self):
        return 'random'

    @property
    def params(self):
        return {'p': self.p}


class EventTriggered(TransmissionPolicy):
    """Transmit when the receiver's estimate has drifted past ``delta`` metres.

    The classic self-triggered baseline: it spends messages exactly where the
    generic predictor fails (turns and noise) and stays silent on straights.
    ``max_silence`` optionally caps the age of information, which keeps the
    estimate from going stale indefinitely if the leader stops moving.
    """

    def __init__(self, delta=0.05, max_silence=None):
        delta = float(delta)
        if delta < 0.0:
            raise ValueError('delta must be non-negative')
        self.delta = delta
        self.max_silence = None if max_silence is None else int(max_silence)

    def should_transmit(self, context):
        # Always send on the first step so the receiver starts from truth.
        if context.last_transmit_step is None:
            return True
        if self.max_silence is not None and context.age >= self.max_silence:
            return True
        return context.prediction_error > self.delta

    @property
    def name(self):
        return 'event_triggered'

    @property
    def params(self):
        params = {'delta': self.delta}
        if self.max_silence is not None:
            params['max_silence'] = self.max_silence
        return params


POLICIES = {
    'always': AlwaysTransmit,
    'periodic': PeriodicTransmit,
    'random': RandomTransmit,
    'event_triggered': EventTriggered,
}


def make_policy(name='always', **kwargs):
    """Build a policy by name (used by configs and the sweep)."""
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(f'unknown policy {name!r}; available: {sorted(POLICIES)}') from None
    return cls(**kwargs)
