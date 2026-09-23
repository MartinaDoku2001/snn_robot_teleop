"""Figures: error time series, trajectories, and the Pareto plot.

The Pareto plot (formation error vs communication rate) is the key artifact:
it is how every later controller -- RL, then spiking -- gets compared against
the analytic baselines on one picture.

Conventions, applied consistently so figures read as one set:

* one categorical hue per policy family, assigned in fixed order and never
  cycled; each series also gets its own marker shape, so identity survives
  greyscale printing and colour-vision deficiency;
* every series is directly labelled as well as present in the legend, so
  identity is never carried by colour alone;
* a single y axis per figure (never a second scale), recessive grid and axes,
  and text in ink colours rather than the series colour.

matplotlib is an optional dependency: import this module only when plotting.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use('Agg')  # headless by default; figures are written to files
import matplotlib.pyplot as plt  # noqa: E402

#: Categorical slots, fixed order (blue, orange, aqua, violet), validated for
#: colour-vision deficiency against a light surface.
SERIES_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7')
SERIES_MARKERS = ('o', 's', '^', 'D')
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID = '#d8d7d2'
SURFACE = '#fcfcfb'

#: Stable colour/marker per policy family, so a family keeps its identity
#: across every figure in the paper.
POLICY_STYLE = {
    'periodic': (SERIES_COLORS[0], SERIES_MARKERS[0]),
    'random': (SERIES_COLORS[1], SERIES_MARKERS[1]),
    'event_triggered': (SERIES_COLORS[2], SERIES_MARKERS[2]),
    'always': (SERIES_COLORS[3], SERIES_MARKERS[3]),
}


def _style_axes(ax, xlabel, ylabel, title=None, scale=1.0):
    """Apply the shared chart styling.

    ``scale`` multiplies every type size: 1.0 for a figure read on paper,
    ~1.4 for one projected in a talk, where the same 9 pt tick label is
    illegible from the back of a room.
    """
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9 * scale)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10 * scale)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10 * scale)
    if title:
        ax.set_title(title, color=INK, fontsize=11 * scale, loc='left', pad=10)


def _style_for(label):
    """Colour/marker for a policy label like ``event_triggered(delta=0.05)``."""
    family = str(label).split('(')[0]
    return POLICY_STYLE.get(family, (SERIES_COLORS[0], SERIES_MARKERS[0]))


def plot_error_timeseries(result, path, show_transmissions=True):
    """Formation error over time, with transmission events underneath.

    The rug along the bottom shows when messages were sent, which makes the
    coupling between communication and error visible in one glance.
    """
    recorder = result.recorder
    time = recorder.column('time')
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.12})

    ax = axes[0]
    series = [
        ('formation error', recorder.column('error_euclidean'), 0),
        ('lateral error', np.abs(recorder.column('error_lateral')), 1),
        ('leader estimate error', recorder.column('estimate_error'), 2),
    ]
    for label, values, slot in series:
        ax.plot(time, values, color=SERIES_COLORS[slot], linewidth=1.6, label=label)
        # Direct label at the end of each trace, so colour is never the only cue.
        if len(time):
            ax.annotate(
                label, xy=(time[-1], values[-1]), xytext=(4, 0),
                textcoords='offset points', color=SERIES_COLORS[slot],
                fontsize=8, va='center')
    _style_axes(
        ax, '', 'error (m)',
        f"Formation error over time - {result.label or 'episode'}")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc='upper left', ncol=3)
    if len(time):
        # Room on the right for the direct labels, which would otherwise be
        # clipped at the axes edge.
        ax.set_xlim(time[0], time[-1] * 1.18)

    ax2 = axes[1]
    if show_transmissions:
        transmitted = recorder.column('transmitted')
        events = time[transmitted > 0.5]
        ax2.vlines(events, 0, 1, color=POLICY_STYLE['event_triggered'][0], linewidth=0.8)
        ax2.set_yticks([])
        rate = float(transmitted.mean()) if len(transmitted) else float('nan')
        ax2.annotate(
            f'{len(events)} messages, rate {rate:.3f}',
            xy=(0.005, 0.72), xycoords='axes fraction',
            color=INK_SECONDARY, fontsize=8)
    _style_axes(ax2, 'time (s)', 'messages')
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_trajectory(result, path, reference_path=None):
    """Leader and follower tracks, with the reference path behind them."""
    recorder = result.recorder
    fig, ax = plt.subplots(figsize=(7, 6))
    if reference_path is not None:
        pts = reference_path.points
        closed = np.vstack([pts, pts[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=GRID, linewidth=1.4,
                label='reference path', zorder=1)
    ax.plot(recorder.column('leader_x'), recorder.column('leader_y'),
            color=SERIES_COLORS[0], linewidth=1.8, label='leader', zorder=3)
    ax.plot(recorder.column('follower_x'), recorder.column('follower_y'),
            color=SERIES_COLORS[1], linewidth=1.4, label='follower', zorder=2)
    ax.set_aspect('equal', adjustable='datalim')
    _style_axes(ax, 'x (m)', 'y (m)', f"Trajectories - {result.label or 'episode'}")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_pareto(series, path, error_key='formation_rms',
                title='Formation error vs communication rate',
                ylabel='formation RMS error (m)', annotate_params=True,
                log_y=True, scale=1.0, family_labels=None, figsize=(8, 5.5)):
    """THE key artifact: error against communication rate, per policy family.

    Args:
        series: mapping ``family -> list of points``, each point a dict with
            ``rate``, ``rate_ci``, ``error``, ``error_ci`` and optionally
            ``label`` (the swept parameter value, e.g. ``k=10``).
        path: output file.
        error_key: only used for the default axis label.
        annotate_params: label the swept parameter next to each point.
        scale: type-size multiplier (see :func:`_style_axes`); >1 for a talk.
        family_labels: optional ``family -> display name`` mapping, so a slide
            can read 'event-triggered' where the code says 'event_triggered'.

    Down-and-left is better: less communication AND less error.
    """
    del error_key
    family_labels = family_labels or {}
    fig, ax = plt.subplots(figsize=figsize)

    for family, points in series.items():
        if not points:
            continue
        points = sorted(points, key=lambda p: p['rate'])
        color, marker = _style_for(family)
        rates = np.array([p['rate'] for p in points], dtype=float)
        errors = np.array([p['error'] for p in points], dtype=float)
        rate_ci = np.array([p.get('rate_ci', 0.0) for p in points], dtype=float)
        error_ci = np.array([p.get('error_ci', 0.0) for p in points], dtype=float)

        display = family_labels.get(family, family)
        ax.errorbar(
            rates, errors, yerr=error_ci, xerr=rate_ci,
            color=color, marker=marker, markersize=7 * scale, linewidth=1.8 * scale,
            elinewidth=1.0, capsize=2.5, label=display, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=0.8)

        # Direct label at the LOW-rate end, where the families are far apart.
        # At high rates they converge, so a label there would collide.
        # Down-and-LEFT of the point: up-and-right runs into the next family's
        # error bars, since the curves are stacked in that direction.
        ax.annotate(
            display, xy=(rates[0], errors[0]),
            xytext=(-10 * scale, -20 * scale), ha='right',
            textcoords='offset points', color=color, fontsize=9 * scale,
            fontweight='bold')

        if annotate_params:
            # One selective label per series: the low-rate extreme. The
            # high-rate ends of all families sit on top of each other at rate
            # 1.0, so labelling them there only produces collisions.
            for point in (points[0],):
                if point.get('label'):
                    # To the RIGHT: straight down is where the family label
                    # now lives, and straight up is the error bar.
                    ax.annotate(
                        point['label'], xy=(point['rate'], point['error']),
                        xytext=(10 * scale, -4 * scale), textcoords='offset points',
                        color=INK_SECONDARY, fontsize=7.5 * scale, ha='left')

    ax.set_xscale('log')
    if log_y:
        # Errors span two orders of magnitude; on a linear axis one bad
        # configuration flattens everything that matters.
        ax.set_yscale('log')
    _style_axes(ax, 'communication rate (messages per control step, log scale)',
                ylabel + (', log scale' if log_y else ''), title, scale=scale)
    ax.annotate(
        'better  \u2199', xy=(0.015, 0.05), xycoords='axes fraction',
        color=INK_SECONDARY, fontsize=9 * scale, style='italic')
    ax.legend(frameon=False, fontsize=9 * scale, labelcolor=INK_SECONDARY,
              loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_demand_profile(result, path, reference_path=None):
    """Where messages are spent along the path -- evidence that demand varies.

    Transmissions are binned by the leader's path curvature, showing that an
    event-triggered policy concentrates messages where a generic predictor
    fails rather than spreading them evenly.
    """
    recorder = result.recorder
    fig, ax = plt.subplots(figsize=(8, 4.5))
    time = recorder.column('time')
    ax.plot(time, recorder.column('prediction_error'),
            color=SERIES_COLORS[0], linewidth=1.4, label='prediction error')
    transmitted = recorder.column('transmitted')
    events = time[transmitted > 0.5]
    for event in events:
        ax.axvline(event, color=SERIES_COLORS[2], linewidth=0.6, alpha=0.6, zorder=1)
    ax.annotate('prediction error', xy=(0.01, 0.92), xycoords='axes fraction',
                color=SERIES_COLORS[0], fontsize=9)
    ax.annotate('transmissions', xy=(0.01, 0.84), xycoords='axes fraction',
                color=SERIES_COLORS[2], fontsize=9)
    _style_axes(ax, 'time (s)', 'estimate error if silent (m)',
                f"Information demand over time - {result.label or 'episode'}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
