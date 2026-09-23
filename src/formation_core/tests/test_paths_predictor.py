"""Paths must mix curvature; the predictor must be exact where it should be."""

import numpy as np
import pytest

from formation_core.dynamics import MotionLimits, unicycle_step
from formation_core.geometry import RobotState
from formation_core.paths import figure_eight_path, make_path, oval_path
from formation_core.predictor import ConstantVelocityPredictor, make_predictor

LIMITS = MotionLimits()


def test_oval_is_closed_and_has_expected_length():
    path = oval_path(straight_length=4.0, radius=1.5)
    expected = 2 * 4.0 + 2 * np.pi * 1.5
    assert path.length == pytest.approx(expected, rel=1e-3)
    gap = np.linalg.norm(path.points[0] - path.points[-1])
    assert gap < 0.05  # closes back on itself


def test_oval_mixes_zero_and_high_curvature():
    """The premise of the task: some segments are trivially predictable."""
    path = oval_path(straight_length=4.0, radius=1.5)
    curvature = np.abs(path.curvature())
    assert np.percentile(curvature, 10) < 1e-6          # straights
    assert np.percentile(curvature, 90) == pytest.approx(1 / 1.5, rel=0.1)


def test_figure_eight_is_closed_and_curvature_changes_sign():
    path = figure_eight_path()
    assert np.linalg.norm(path.points[0] - path.points[-1]) < 0.05
    curvature = path.curvature()
    assert curvature.max() > 0 and curvature.min() < 0


def test_closest_and_tracking_error():
    path = oval_path()
    on_path = path.points[123]
    assert path.tracking_error(on_path) == pytest.approx(0.0, abs=1e-6)
    off_path = on_path + np.array([0.0, 0.0])
    idx, point, distance, arc = path.closest(off_path)
    assert idx == 123
    assert distance == pytest.approx(0.0, abs=1e-6)
    assert arc == pytest.approx(path.cumulative[123])


def test_lookahead_advances_and_wraps():
    path = oval_path()
    start = path.points[0]
    ahead = path.lookahead(start, 1.0)
    # 1 m along a straight section from the start point.
    assert np.linalg.norm(ahead - start) == pytest.approx(1.0, rel=0.02)
    # Wrapping past the end of the loop returns to the start.
    wrapped = path.point_at(path.length + 0.0)
    np.testing.assert_allclose(wrapped, path.points[0], atol=1e-6)


def test_make_path_registry_and_errors():
    assert make_path('oval').name == 'oval'
    assert make_path('figure8').name == 'figure8'
    with pytest.raises(ValueError):
        make_path('spiral')


def test_predictor_is_exact_on_straight_constant_velocity():
    """No turn, no noise => dead reckoning must match the true model exactly."""
    predictor = ConstantVelocityPredictor()
    state = RobotState(0.0, 0.0, 0.3, v=0.6, w=0.0)
    truth = state.copy()
    estimate = state.copy()
    for _ in range(40):
        truth = unicycle_step(truth, 0.6, 0.0, 0.05, LIMITS)
        estimate = predictor.step(estimate, 0.05)
    assert np.hypot(truth.x - estimate.x, truth.y - estimate.y) < 1e-9


def test_predictor_is_exact_on_steady_turn():
    predictor = ConstantVelocityPredictor()
    state = RobotState(0.0, 0.0, 0.0, v=0.6, w=0.4)
    truth = state.copy()
    estimate = state.copy()
    for _ in range(40):
        truth = unicycle_step(truth, 0.6, 0.4, 0.05, LIMITS)
        estimate = predictor.step(estimate, 0.05)
    # Same integrator, same constant inputs: identical to numerical precision.
    assert np.hypot(truth.x - estimate.x, truth.y - estimate.y) < 1e-9


def test_predictor_degrades_when_the_leader_starts_turning():
    """This degradation is what makes information demand vary."""
    predictor = ConstantVelocityPredictor()
    truth = RobotState(0.0, 0.0, 0.0, v=0.6, w=0.0)
    estimate = truth.copy()
    for _ in range(20):
        truth = unicycle_step(truth, 0.6, 1.0, 0.05, LIMITS)  # enters a turn
        estimate = predictor.step(estimate, 0.05)             # keeps going straight
    assert np.hypot(truth.x - estimate.x, truth.y - estimate.y) > 0.05


def test_constant_heading_variant_ignores_yaw_rate():
    predictor = make_predictor('constant_heading')
    estimate = RobotState(0.0, 0.0, 0.0, v=1.0, w=2.0)
    stepped = predictor.step(estimate, 0.1)
    assert stepped.w == 0.0
    assert stepped.theta == pytest.approx(0.0)
    assert predictor.name == 'constant_heading'


def test_predictor_ignores_last_command_but_accepts_it():
    """The hook exists for later phases; passing it must not change Phase 1."""
    predictor = ConstantVelocityPredictor()
    estimate = RobotState(0.0, 0.0, 0.0, v=0.5, w=0.0)
    a = predictor.step(estimate, 0.05)
    b = predictor.step(estimate, 0.05, last_command=(9.0, 9.0))
    assert (a.x, a.y, a.theta) == (b.x, b.y, b.theta)


def test_unknown_predictor_rejected():
    with pytest.raises(ValueError):
        make_predictor('kalman')
