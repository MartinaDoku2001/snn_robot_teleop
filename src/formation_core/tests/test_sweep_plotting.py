"""The premise checker and the figures.

The checker decides whether the task is well posed, so it must fail loudly on
data where the premise does NOT hold -- otherwise it would rubber-stamp a
broken task.
"""

import os
import tempfile

import pytest

from formation_core.config import EpisodeConfig, SuiteConfig
from formation_core.runner import run_episode
from formation_core.sweep import (
    check_premise,
    interpolate_error_at_rate,
    sweep_family,
)


def _points(family, rates, errors):
    return [
        {'family': family, 'rate': r, 'error': e, 'rate_ci': 0.0, 'error_ci': 0.0,
         'label': f'{family[0]}={i}', 'value': float(i)}
        for i, (r, e) in enumerate(zip(rates, errors))
    ]


def test_interpolation_is_log_rate_and_never_extrapolates():
    points = _points('periodic', [0.1, 1.0], [0.5, 0.1])
    middle = interpolate_error_at_rate(points, 0.31622776601)  # geometric mean
    assert middle == pytest.approx(0.3, abs=1e-6)
    import math
    assert math.isnan(interpolate_error_at_rate(points, 0.01))   # below range
    assert math.isnan(interpolate_error_at_rate(points, 10.0))   # above range


def test_check_premise_passes_on_data_that_satisfies_it():
    series = {
        'periodic': _points('periodic', [0.01, 0.1, 0.5, 1.0], [0.8, 0.3, 0.06, 0.04]),
        'random': _points('random', [0.01, 0.1, 0.5, 1.0], [1.2, 0.4, 0.07, 0.04]),
        'event_triggered': _points(
            'event_triggered', [0.01, 0.1, 0.5, 1.0], [0.12, 0.05, 0.045, 0.04]),
    }
    ok, lines = check_premise(series, verbose=False)
    assert ok
    assert all('FAIL' not in line for line in lines)


def test_check_premise_fails_when_error_does_not_rise_as_rate_falls():
    """A flat trade-off means the task does not test communication at all."""
    flat = _points('periodic', [0.01, 0.1, 0.5, 1.0], [0.05, 0.05, 0.05, 0.05])
    series = {
        'periodic': flat,
        'event_triggered': _points(
            'event_triggered', [0.01, 0.1, 0.5, 1.0], [0.05, 0.05, 0.05, 0.05]),
    }
    ok, lines = check_premise(series, verbose=False)
    assert not ok
    assert any('FAIL' in line for line in lines)


def test_check_premise_fails_when_event_triggered_is_not_better():
    """If timing does not matter, there is nothing for a scheduler to learn."""
    series = {
        'periodic': _points('periodic', [0.01, 0.1, 0.5, 1.0], [0.4, 0.1, 0.05, 0.04]),
        'event_triggered': _points(
            'event_triggered', [0.01, 0.1, 0.5, 1.0], [0.9, 0.3, 0.08, 0.04]),
    }
    ok, lines = check_premise(series, verbose=False)
    assert not ok
    assert any('FAIL' in line and 'beats periodic' in line for line in lines)


def test_sweep_family_returns_points_with_rate_and_error():
    suite = SuiteConfig(
        name='tiny', seeds=[0, 1], episode=EpisodeConfig(duration=4.0))
    points = sweep_family(suite, 'periodic', [1, 8])
    assert len(points) == 2
    assert points[0]['rate'] == pytest.approx(1.0)
    assert points[1]['rate'] < points[0]['rate']
    assert points[0]['episodes'] == 2
    assert all(key in points[0] for key in ('error', 'error_ci', 'aoi_mean', 'label'))


def test_figures_are_written():
    matplotlib = pytest.importorskip('matplotlib')
    del matplotlib
    from formation_core.paths import make_path
    from formation_core.plotting import (
        plot_demand_profile,
        plot_error_timeseries,
        plot_pareto,
        plot_trajectory,
    )

    result = run_episode(EpisodeConfig(duration=3.0))
    series = {
        'periodic': _points('periodic', [0.1, 1.0], [0.3, 0.05]),
        'event_triggered': _points('event_triggered', [0.1, 1.0], [0.08, 0.05]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for path in (
            plot_error_timeseries(result, os.path.join(tmp, 'errors.png')),
            plot_trajectory(result, os.path.join(tmp, 'traj.png'),
                            reference_path=make_path('oval')),
            plot_demand_profile(result, os.path.join(tmp, 'demand.png')),
            plot_pareto(series, os.path.join(tmp, 'pareto.png')),
        ):
            assert os.path.getsize(path) > 5000  # a real figure, not an empty canvas


# ------------------------------------------------------------ presentation

def test_pareto_scales_and_relabels_for_a_talk():
    """The talk figure must be the SAME plot, only bigger and better named."""
    pytest.importorskip('matplotlib')
    from formation_core.plotting import plot_pareto

    series = {'event_triggered': _points('event_triggered', [0.1, 1.0], [0.08, 0.05])}
    with tempfile.TemporaryDirectory() as tmp:
        paper = plot_pareto(series, os.path.join(tmp, 'paper.png'))
        talk = plot_pareto(
            series, os.path.join(tmp, 'talk.png'), scale=1.4,
            family_labels={'event_triggered': 'event-triggered'},
            figsize=(11, 6.2))
        # Same data, larger canvas: the talk file is the heavier render.
        assert os.path.getsize(talk) > os.path.getsize(paper)


def test_presentation_figures_are_written():
    pytest.importorskip('matplotlib')
    from formation_core import figures

    # A deliberately tiny suite: these figures run their own episodes, and the
    # test is checking that they are produced, not what the numbers are.
    suite = SuiteConfig(name='tiny', seeds=[0, 1], episode=EpisodeConfig(duration=4.0))
    with tempfile.TemporaryDirectory() as tmp:
        written = [
            figures.figure_matched_budget(suite, tmp, progress=lambda *_: None),
            figures.figure_mechanism(suite.episode, tmp),
            figures.figure_trajectories(suite.episode, tmp),
        ]
        for path in written:
            assert os.path.getsize(path) > 5000     # a real figure, not a blank canvas
            assert os.path.exists(path.replace('.png', '.pdf'))  # print version too


def test_sim_to_sim_uses_each_gazebo_run_s_own_config(tmp_path):
    """The twin must re-run the config that Gazebo actually ran.

    The run directory is the only record of it -- the launch file's policy
    override never reaches the packaged YAML -- so reading the config from
    anywhere else silently compares two different episodes.
    """
    pytest.importorskip('matplotlib')
    from formation_core import figures

    run_dir = tmp_path / 'periodic_k4'
    run_dir.mkdir()
    EpisodeConfig(duration=4.0, policy={'name': 'periodic', 'k': 4}).to_yaml(
        str(run_dir / 'config.yaml'))
    (run_dir / 'metrics.csv').write_text(
        'formation_rms,comm_rate,messages\n0.0731,0.25,20\n')

    rows = figures._gazebo_rows(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]['config'].policy.describe() == 'periodic(k=4)'
    assert rows[0]['config'].duration == 4.0

    path = figures.figure_sim_to_sim(
        str(tmp_path), None, str(tmp_path), progress=lambda *_: None)
    assert os.path.getsize(path) > 5000
