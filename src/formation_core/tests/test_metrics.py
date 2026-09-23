"""Metrics are the ruler: check them against hand-computed values."""

import math

import numpy as np
import pytest

from formation_core.geometry import RobotState
from formation_core.metrics import EpisodeRecorder, SuiteResult, mean_ci, t95
from formation_core.metrics import EpisodeResult


def _info(step, euclidean, transmitted, age, lateral=0.0):
    state = RobotState(float(step), 0.0, 0.0, v=0.5, w=0.0)
    return {
        'step': step,
        'time': step * 0.05,
        'leader_state': state,
        'follower_state': state,
        'leader_estimate': state,
        'transmitted': transmitted,
        'received': transmitted,
        'age': age,
        'prediction_error': 0.01,
        'estimate_error': 0.02,
        'path_error': 0.03,
        'errors': {
            'longitudinal': euclidean,
            'lateral': lateral,
            'heading': 0.1,
            'euclidean': euclidean,
        },
        'action': np.array([0.5, 0.0]),
        'reward': -euclidean,
    }


def test_error_statistics_match_hand_computation():
    recorder = EpisodeRecorder(dt=0.05)
    for step, value in enumerate([0.0, 0.1, -0.2, 0.3]):
        recorder.record(_info(step, value, transmitted=True, age=0))
    metrics = recorder.metrics()
    assert metrics['formation_mean'] == pytest.approx(np.mean([0.0, 0.1, 0.2, 0.3]))
    assert metrics['formation_rms'] == pytest.approx(
        math.sqrt(np.mean(np.square([0.0, 0.1, 0.2, 0.3]))))
    assert metrics['formation_max'] == pytest.approx(0.3)
    # Longitudinal keeps its sign in the RMS but not in mean-absolute.
    assert metrics['longitudinal_mean'] == pytest.approx(0.15)


def test_communication_metrics():
    recorder = EpisodeRecorder(dt=0.05)
    pattern = [True, False, False, True, False, False, True, False]
    age = 0
    for step, transmitted in enumerate(pattern):
        age = 0 if transmitted else age + 1
        recorder.record(_info(step, 0.1, transmitted=transmitted, age=age))
    metrics = recorder.metrics()
    assert metrics['steps'] == 8
    assert metrics['messages'] == 3
    assert metrics['comm_rate'] == pytest.approx(3 / 8)
    assert metrics['messages_per_second'] == pytest.approx(3 / (8 * 0.05))
    assert metrics['iti_mean'] == pytest.approx(3.0)      # gaps 0->3->6
    assert metrics['iti_min'] == 3 and metrics['iti_max'] == 3
    assert metrics['aoi_mean'] == pytest.approx(np.mean([0, 1, 2, 0, 1, 2, 0, 1]))
    assert metrics['aoi_max'] == 2


def test_no_transmissions_gives_nan_intervals_not_a_crash():
    recorder = EpisodeRecorder(dt=0.05)
    for step in range(5):
        recorder.record(_info(step, 0.1, transmitted=False, age=step + 1))
    metrics = recorder.metrics()
    assert metrics['messages'] == 0
    assert metrics['comm_rate'] == 0.0
    assert math.isnan(metrics['iti_mean'])


def test_mean_ci_and_t_values():
    assert t95(7) == pytest.approx(2.365)
    assert t95(100) == pytest.approx(1.96)
    mean, half = mean_ci([1.0, 1.0, 1.0])
    assert mean == pytest.approx(1.0) and half == pytest.approx(0.0)
    mean, half = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    expected = 2.0 * t95(3) * (np.std([1, 2, 3, 4], ddof=1) / 2.0) / 2.0
    assert half == pytest.approx(expected, rel=1e-6)
    # A single sample has no spread to estimate.
    assert mean_ci([5.0]) == (5.0, 0.0)
    assert math.isnan(mean_ci([])[0])


def test_suite_aggregate_reports_mean_and_ci_per_metric():
    class _Config:
        seed = 0

    episodes = []
    for value in (0.1, 0.2, 0.3):
        recorder = EpisodeRecorder(dt=0.05)
        recorder.record(_info(0, value, transmitted=True, age=0))
        config = _Config()
        episodes.append(EpisodeResult(
            config=config, metrics=recorder.metrics(), recorder=recorder))
    suite = SuiteResult(name='s', episodes=episodes, label='always')
    aggregate = suite.aggregate()
    assert aggregate['episodes'] == 3
    assert aggregate['formation_mean_mean'] == pytest.approx(0.2)
    assert aggregate['formation_mean_ci95'] > 0.0


def test_recorder_column_access_and_metadata():
    recorder = EpisodeRecorder(dt=0.05)
    recorder.metadata['seed'] = 7
    for step in range(3):
        recorder.record(_info(step, 0.1 * step, transmitted=True, age=0))
    np.testing.assert_allclose(recorder.column('error_euclidean'), [0.0, 0.1, 0.2])
    assert recorder.metrics()['seed'] == 7
    assert len(recorder) == 3
