"""The Phase 2 figure set is produced and is not blank.

Whether the figures are GOOD is a matter of looking at them; what a test can
pin is that each one runs against real data and writes both a slide PNG and a
print PDF.
"""

import os

import pytest

pytest.importorskip('torch')
pytest.importorskip('matplotlib')

from formation_core.config import EpisodeConfig, SuiteConfig  # noqa: E402


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A 2-seed, 4-second suite standing in for eval/stress, and a saved actor."""
    from formation_rl import figures
    from formation_rl.actor import MlpActor

    suite = SuiteConfig(name='tiny', seeds=[0, 1],
                        episode=EpisodeConfig(duration=4.0))
    monkeypatch.setattr(figures, '_suite', lambda name: suite)
    monkeypatch.setattr(figures, 'SUITES', ('tiny',))
    weights = MlpActor(hidden=(16, 16)).save(str(tmp_path / 'actor.pt'))
    return figures, weights, str(tmp_path)


def _written(path):
    assert path is not None
    assert os.path.getsize(path) > 5000          # a real figure, not a blank canvas
    assert os.path.exists(path.replace('.png', '.pdf'))


def test_architecture_needs_no_data(tmp_path):
    from formation_rl import figures

    _written(figures.figure_architecture(str(tmp_path)))


def test_comparison_figures_are_written(tiny):
    figures, weights, out = tiny
    rows = figures.measure(['tiny'], weights, progress=lambda *_: None)
    assert {r['controller'] for r in rows} == {'analytic', 'rl'}
    _written(figures.figure_controllers(rows, out))
    _written(figures.figure_tradeoff(rows, out))
    _written(figures.figure_trajectories(weights, out, suite='tiny'))


def test_training_curve_reads_the_history(tiny, tmp_path):
    figures, _, out = tiny
    history = tmp_path / 'training_history.csv'
    history.write_text(
        'update,steps,mean_reward\n'
        + '\n'.join(f'{i},{i * 2048},{-1.0 + i * 0.05}' for i in range(1, 40)) + '\n')
    _written(figures.figure_training(str(history), 0.37, out, window=5))


def test_sim_to_sim_is_skipped_without_gazebo_runs(tiny):
    figures, weights, out = tiny
    rows = figures.measure(['tiny'], weights, progress=lambda *_: None)
    assert figures.figure_sim_to_sim({'analytic': '', 'rl': ''}, rows, out) is None
