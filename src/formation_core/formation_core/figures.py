"""Presentation figures: the five pictures that carry the result.

These are the talk versions of the analysis in :mod:`formation_core.plotting`
-- same data, same palette, same per-family colour identity, but sized and
labelled for a projector rather than a page. Each figure answers exactly one
question, in the order a talk asks them:

    1. matched_budget  At a fixed message budget, how much better is it?
    2. pareto          Does that hold across the whole budget range?
    3. mechanism       WHY does it win?
    4. trajectories    What does the failure actually look like?
    5. sim_to_sim      Does the fast twin's answer survive a real simulator?

Figures 1-4 run the episodes they need (the fast twin is quick enough to make
that cheaper than plumbing cached files around). Figure 2 reads the sweep that
``formation_core sweep`` already wrote; figure 5 reads Gazebo ``metrics.csv``
files produced by ``formation_gazebo``.

    python3 -m formation_core figures --out results/figures

matplotlib is an optional dependency: import this module only when plotting.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from .config import ComponentConfig, EpisodeConfig, SuiteConfig
from .metrics import mean_ci
from .paths import make_path
from .plotting import (
    GRID,
    INK,
    INK_SECONDARY,
    POLICY_STYLE,
    SERIES_COLORS,
    SURFACE,
    _style_axes,
    plot_pareto,
)
from .runner import config_path, run_episode, run_suite

import matplotlib.pyplot as plt

#: Type scale for a projected figure. 1.0 is the paper size used by
#: :mod:`formation_core.plotting`; a 9 pt tick label does not survive a talk.
SCALE = 1.4

#: The matched-budget comparison at the heart of the result: three policies
#: tuned to spend the SAME number of messages, so the only difference left is
#: WHEN they spend them. Values come from the sweep; see MATCHED_RATE.
MATCHED_BUDGET = (
    ComponentConfig('event_triggered', {'delta': 0.05}),
    ComponentConfig('periodic', {'k': 50}),
    ComponentConfig('random', {'p': 0.02}),
)
MATCHED_RATE = 0.02

#: Human-readable family names, so a slide never shows a Python identifier.
FAMILY_LABEL = {
    'event_triggered': 'event-triggered',
    'periodic': 'periodic',
    'random': 'random',
    'always': 'always (full rate)',
}


#: Symbols for the swept parameters, for labels with no room to spell them.
PARAM_SYMBOL = {'delta': '\u03b4'}


def _compact(component):
    """``event_triggered(delta=0.05)`` -> ``event-triggered  \u03b4=0.05``."""
    family = FAMILY_LABEL.get(component.name, component.name)
    if not component.params:
        return family
    inner = ' '.join(f'{PARAM_SYMBOL.get(k, k)}={v:g}'
                     for k, v in sorted(component.params.items()))
    return f'{family}  {inner}'


def _describe(component):
    """``periodic(k=50)`` -> ``periodic, k = 50`` for a slide."""
    family = FAMILY_LABEL.get(component.name, component.name)
    if not component.params:
        return family
    inner = ', '.join(f'{k} = {v:g}' for k, v in sorted(component.params.items()))
    return f'{family}  ({inner})'


def save_figure(fig, out_dir, name):
    """Write PNG for slides and PDF for anything that gets printed."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f'{name}.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths[0]


def _suite_at(suite, policy):
    """Run ``suite`` under one policy and return (mean, ci, messages, rate)."""
    results = run_suite(suite, label=policy.describe(), overrides={'policy': policy})
    error, error_ci = mean_ci(results.values('formation_rms'))
    rate, _ = mean_ci(results.values('comm_rate'))
    messages, _ = mean_ci(results.values('messages'))
    return error, error_ci, messages, rate


# --------------------------------------------------------------- 1. headline

def figure_matched_budget(suite, out_dir, progress=print):
    """Formation error of the three families at ONE shared message budget.

    A horizontal bar chart: the job is magnitude comparison across three named
    things, the names are long, and the ordering is the message. Every bar is
    directly labelled with its value, so the figure survives being projected
    badly.
    """
    rows = []
    for policy in MATCHED_BUDGET:
        error, error_ci, messages, rate = _suite_at(suite, policy)
        rows.append({'policy': policy, 'error': error, 'ci': error_ci,
                     'messages': messages, 'rate': rate})
        progress(f'  {policy.describe():28s} {error:.4f} m  '
                 f'({messages:.0f} messages, rate {rate:.3f})')
    rows.sort(key=lambda r: r['error'])

    fig, ax = plt.subplots(figsize=(11, 5.0))
    y = np.arange(len(rows))
    colors = [POLICY_STYLE[r['policy'].name][0] for r in rows]
    errors = [r['error'] for r in rows]

    ax.barh(y, errors, height=0.62, color=colors, zorder=3,
            edgecolor=SURFACE, linewidth=2.0)          # 2 px surface gap
    ax.errorbar(errors, y, xerr=[r['ci'] for r in rows], fmt='none',
                ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=4)

    best = rows[0]['error']
    for i, row in enumerate(rows):
        ax.annotate(f"{row['error']:.3f} m", xy=(row['error'], i), xytext=(8, 0),
                    textcoords='offset points', va='center',
                    color=INK, fontsize=11 * SCALE, fontweight='bold')
        if i:   # how much worse than the best, the comparison people remember
            ax.annotate(f"{row['error'] / best:.1f}x worse", xy=(row['error'], i),
                        xytext=(8, -15 * SCALE), textcoords='offset points',
                        va='center', color=INK_SECONDARY, fontsize=9 * SCALE)

    ax.set_yticks(y)
    ax.set_yticklabels([_describe(r['policy']) for r in rows], fontsize=10 * SCALE)
    ax.invert_yaxis()
    low = min(r['messages'] for r in rows)
    high = max(r['messages'] for r in rows)
    budget = f'{high:.0f}' if high - low < 1 else f'{low:.0f}-{high:.0f}'
    _style_axes(
        ax, 'formation RMS error (m)', '',
        f'Same bandwidth, {budget} messages per 60 s episode: only the TIMING differs',
        scale=SCALE)
    # Room for the widest CI whisker and its label: random's CI is wider than
    # periodic's entire bar, and clipping it would hide exactly that.
    reach = max(r['error'] + r['ci'] for r in rows)
    ax.set_xlim(0, max(max(errors), reach) * 1.20)
    ax.grid(axis='y', visible=False)
    fig.text(0.0, -0.04,
             f'{suite.name} suite, {len(suite.seeds)} seeds, mean ± 95% CI — '
             "random's CI is wider than periodic's whole bar",
             color=INK_SECONDARY, fontsize=9 * SCALE)
    return save_figure(fig, out_dir, '01_matched_budget')


# ---------------------------------------------------------------- 2. pareto

def figure_pareto(sweep_dir, out_dir):
    """The whole trade-off curve, with the matched budget marked on it.

    Reuses :func:`formation_core.plotting.plot_pareto` so the talk figure and
    the paper figure cannot drift apart, then adds the budget guide.
    """
    with open(os.path.join(sweep_dir, 'pareto.json')) as handle:
        data = json.load(handle)
    series = {}
    for point in data['points']:
        series.setdefault(point['family'], []).append(point)

    for ext in ('png', 'pdf'):
        path = plot_pareto(
            series, os.path.join(out_dir, f'02_pareto.{ext}'),
            title='Less bandwidth costs accuracy \u2014 but not equally for every policy\n'
                  f"{data['suite']} suite, {len(data['seeds'])} seeds, mean \u00b1 95% CI",
            scale=SCALE, family_labels=FAMILY_LABEL, figsize=(11, 6.2))
    return os.path.join(out_dir, '02_pareto.png')


# ------------------------------------------------------------- 3. mechanism

def figure_mechanism(config, out_dir):
    """WHY event-triggering wins: where each policy spends its messages.

    Two panels, shared axes, same budget. The trace is the error the follower
    would carry if the leader stayed silent; the rug marks actual messages.
    Periodic spends on a blind metronome; event-triggered spends exactly where
    the constant-velocity prediction breaks down -- the curves.
    """
    pairs = [
        ComponentConfig('periodic', {'k': 50}),
        ComponentConfig('event_triggered', {'delta': 0.05}),
    ]
    results = [run_episode(config.with_overrides(policy=p), label=p.describe())
               for p in pairs]

    fig, axes = plt.subplots(len(pairs), 1, figsize=(11, 6.4), sharex=True, sharey=True)
    # At t=0 the leader accelerates from rest while the constant-velocity
    # predictor still holds v=0, so the first inter-message gap carries a
    # metre-scale spike. Scaling to it would flatten the steady state that is
    # the actual subject, so the axis is set from the steady state and the
    # transient is called out where it clips.
    # The cutoff is a fraction of the episode, not a fixed 5 s: a short run
    # has no samples past 5 s at all, and an empty slice makes the axis NaN.
    cutoff = min(5.0, 0.25 * config.duration)
    steady = np.concatenate([
        r.recorder.column('prediction_error')[r.recorder.column('time') > cutoff]
        for r in results])
    if not steady.size or not np.isfinite(np.nanmax(steady)):
        steady = np.concatenate(
            [r.recorder.column('prediction_error') for r in results])
    ceiling = float(np.nanpercentile(steady, 99.5)) * 1.6
    peak = max(float(np.nanmax(r.recorder.column('prediction_error'))) for r in results)

    for ax, policy, result in zip(axes, pairs, results):
        recorder = result.recorder
        time = recorder.column('time')
        error = recorder.column('prediction_error')
        color = POLICY_STYLE[policy.name][0]
        events = time[recorder.column('transmitted') > 0.5]

        ax.plot(time, error, color=color, linewidth=2.0, zorder=3)
        ax.vlines(events, 0, ceiling * 1.10, color=color, alpha=0.45,
                  linewidth=1.4, zorder=1)
        _style_axes(ax, '', 'error if silent (m)', scale=SCALE)
        ax.set_ylim(0, ceiling * 1.10)
        ax.annotate(
            f'{_describe(policy)}   —   {len(events)} messages',
            xy=(0.008, 0.87), xycoords='axes fraction',
            color=INK, fontsize=11 * SCALE, fontweight='bold')

    axes[0].annotate(
        'messages on a fixed beat, wherever the error happens to be',
        xy=(0.008, 0.72), xycoords='axes fraction',
        color=INK_SECONDARY, fontsize=9.5 * SCALE)
    axes[1].annotate(
        'messages only when the prediction is about to fail',
        xy=(0.008, 0.72), xycoords='axes fraction',
        color=INK_SECONDARY, fontsize=9.5 * SCALE)
    _style_axes(axes[-1], 'time (s)', 'error if silent (m)', scale=SCALE)
    axes[0].annotate(
        f'start-up transient peaks at {peak:.2f} m, above this axis',
        xy=(0.008, 0.58), xycoords='axes fraction',
        color=INK_SECONDARY, fontsize=8.5 * SCALE, style='italic')
    axes[0].set_title(
        'A comparable number of messages, spent differently',
        color=INK, fontsize=12 * SCALE, loc='left', pad=12)
    fig.tight_layout()
    return save_figure(fig, out_dir, '03_mechanism')


# ----------------------------------------------------------- 4. trajectories

def figure_trajectories(config, out_dir):
    """What the failure looks like on the ground, at the matched budget.

    Plain leader/follower tracks do NOT show this result: most of the error is
    along-track, so a 0.05 m and a 0.30 m episode trace nearly the same oval.
    The follower's path is therefore coloured by its instantaneous formation
    error on a single light-to-dark ramp, shared across the three panels, so
    the eye compares magnitude rather than shape. Where a policy fails is then
    visible as a dark stretch -- and it falls on the curves.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    reference = make_path(config.path.name, **config.path.params)
    results = [run_episode(config.with_overrides(policy=p), label=p.describe())
               for p in MATCHED_BUDGET]

    # Sequential ramp: ONE hue, light to dark. Magnitude is not identity, so
    # it must not reuse the categorical per-family colours.
    ramp = LinearSegmentedColormap.from_list(
        'formation_error', ['#dfe8f5', '#8fb3e0', '#3f6fb5', '#1b3566'])
    # One scale across all three panels, or the eye compares nothing. It is set
    # from the 90th percentile of the pooled error and the bar is marked as
    # extending, so random's worst excursion saturates instead of washing the
    # other two panels out at the pale end of the ramp.
    pooled = np.concatenate([r.recorder.column('error_euclidean') for r in results])
    ceiling = float(np.nanpercentile(pooled, 90))
    norm = Normalize(vmin=0.0, vmax=ceiling)

    fig, axes = plt.subplots(1, len(results), figsize=(14, 3.9))
    points = reference.points
    closed = np.vstack([points, points[:1]])

    for ax, policy, result in zip(axes, MATCHED_BUDGET, results):
        recorder = result.recorder
        ax.plot(closed[:, 0], closed[:, 1], color=GRID, linewidth=6.0,
                zorder=1, solid_capstyle='round')
        track = np.column_stack(
            [recorder.column('follower_x'), recorder.column('follower_y')])
        segments = np.stack([track[:-1], track[1:]], axis=1)
        collection = LineCollection(
            segments, cmap=ramp, norm=norm, linewidth=2.6, zorder=3,
            capstyle='round')
        collection.set_array(recorder.column('error_euclidean')[:-1])
        ax.add_collection(collection)
        ax.set_aspect('equal')
        # The axes share one frame and one unit, so labelling every panel just
        # repeats the same two words three times.
        _style_axes(ax, 'x (m)' if ax is axes[len(axes) // 2] else '',
                    'y (m)' if ax is axes[0] else '', scale=SCALE)
        ax.annotate(
            f"{_compact(policy)}\nRMS {result.metrics['formation_rms']:.3f} m",
            xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom',
            color=INK, fontsize=10 * SCALE)

    # One frame for all three panels: with per-panel limits a bigger excursion
    # is drawn smaller, which is exactly backwards.
    span = max(np.abs(np.column_stack(
        [r.recorder.column('follower_x'), r.recorder.column('follower_y')])).max()
        for r in results) * 1.08
    for ax in axes:
        ax.set_xlim(-span, span)
        ax.set_ylim(-span * 0.46, span * 0.46)

    # Equal aspect on a wide, short scene leaves the axes box much shorter than
    # its slot, so an unshrunk bar towers over the panels and pads the figure.
    bar = fig.colorbar(collection, ax=axes, fraction=0.025, pad=0.02,
                       extend='max', shrink=0.52)
    bar.set_label('formation error (m)', color=INK_SECONDARY, fontsize=10 * SCALE)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=9 * SCALE)
    bar.outline.set_visible(False)
    fig.suptitle(
        f'Follower track at a matched message budget (seed {config.seed})',
        color=INK, fontsize=12 * SCALE, x=0.02, ha='left', y=1.04)
    return save_figure(fig, out_dir, '04_trajectories')


# -------------------------------------------------------------- 5. transfer

def _gazebo_rows(results_dir):
    """Every Gazebo run under ``results_dir``, with the config it actually ran.

    Each run directory holds the EFFECTIVE config beside its metrics, so the
    twin can be given the identical episode -- same duration, same policy --
    instead of the package default, which would compare two different tasks.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*', 'metrics.csv'))):
        with open(path) as handle:
            row = next(csv.DictReader(handle))
        config_path_ = os.path.join(os.path.dirname(path), 'config.yaml')
        if not os.path.exists(config_path_):
            continue
        rows.append({
            'config': EpisodeConfig.from_yaml(config_path_),
            'error': float(row['formation_rms']),
            'rate': float(row['comm_rate']),
            'messages': int(row['messages']),
        })
    return rows


def figure_sim_to_sim(gazebo_dir, gazebo_config, out_dir, progress=print):
    """Fast twin vs Gazebo on the SAME episode config and the same policies.

    A paired dot plot, not grouped bars: the Gazebo side is a single seed, and
    bars would imply a precision that one episode does not have. What the
    figure claims is only what it can support -- the two backends land in the
    same place.
    """
    gazebo = _gazebo_rows(gazebo_dir)
    if not gazebo:
        return None

    rows = []
    for entry in gazebo:
        config = entry['config']
        result = run_episode(config, label=config.policy.describe())
        rows.append({'policy': config.policy,
                     'twin': result.metrics['formation_rms'],
                     'gazebo': entry['error'],
                     'messages': entry['messages'],
                     'duration': config.duration})
        progress(f"  {config.policy.describe():28s} twin {rows[-1]['twin']:.4f} m  "
                 f"gazebo {entry['error']:.4f} m  ({config.duration:.0f} s)")
    rows.sort(key=lambda r: r['gazebo'])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(rows))
    for i, row in enumerate(rows):
        ax.plot([row['twin'], row['gazebo']], [i, i],
                color=GRID, linewidth=2.5, zorder=1, solid_capstyle='round')
    ax.scatter([r['twin'] for r in rows], y, s=150, color=SERIES_COLORS[0],
               marker='o', zorder=3, edgecolor=SURFACE, linewidth=2, label='fast twin')
    ax.scatter([r['gazebo'] for r in rows], y, s=150, color=SERIES_COLORS[1],
               marker='s', zorder=3, edgecolor=SURFACE, linewidth=2, label='Gazebo')

    for i, row in enumerate(rows):
        ax.annotate(f"{row['messages']} msgs", xy=(max(row['twin'], row['gazebo']), i),
                    xytext=(14, 0), textcoords='offset points', va='center',
                    color=INK_SECONDARY, fontsize=9.5 * SCALE)

    ax.set_yticks(y)
    ax.set_yticklabels([_describe(r['policy']) for r in rows], fontsize=10 * SCALE)
    ax.invert_yaxis()
    _style_axes(ax, 'formation RMS error (m)', '',
                'The same code, measured in both backends', scale=SCALE)
    ax.grid(axis='y', visible=False)
    # A dot plot encodes position, not length, so it needs no zero baseline;
    # anchoring at 0 would squeeze every point into the right-hand third.
    lo = min(min(r['twin'], r['gazebo']) for r in rows)
    hi = max(max(r['twin'], r['gazebo']) for r in rows)
    pad = (hi - lo) * 0.35
    ax.set_xlim(lo - pad, hi + pad * 2.2)
    # Centre-right: the only region of this plot with no marks or labels in it.
    ax.legend(frameon=False, fontsize=10 * SCALE, labelcolor=INK_SECONDARY,
              loc='center right')
    duration = rows[0]['duration']
    fig.text(0.0, -0.05,
             f'Gazebo: one {duration:.0f} s episode per policy, single seed — '
             'this shows the two backends agree, it does not rank the policies',
             color=INK_SECONDARY, fontsize=9 * SCALE)
    return save_figure(fig, out_dir, '05_sim_to_sim')


# -------------------------------------------------------------------- entry

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--suite', default=config_path('eval_suite.yaml'), help='suite YAML')
    parser.add_argument('--out', default='results/figures', help='output directory')
    parser.add_argument(
        '--sweep', default='results/sweep',
        help='directory holding pareto.json (from: formation_core sweep)')
    parser.add_argument(
        '--gazebo-results', default='',
        help='directory of Gazebo run directories, each with a metrics.csv')
    parser.add_argument(
        '--gazebo-config', default='',
        help='the episode YAML those Gazebo runs used')
    parser.add_argument(
        '--only', nargs='+', default=None,
        help='subset of: budget pareto mechanism trajectories transfer')
    args = parser.parse_args(argv)

    suite = SuiteConfig.from_yaml(args.suite)
    config = suite.episode
    os.makedirs(args.out, exist_ok=True)
    wanted = set(args.only) if args.only else None

    def want(name):
        return wanted is None or name in wanted

    written = []
    if want('budget'):
        print('1/5 matched budget')
        written.append(figure_matched_budget(suite, args.out))
    if want('pareto'):
        print('2/5 pareto')
        if os.path.exists(os.path.join(args.sweep, 'pareto.json')):
            written.append(figure_pareto(args.sweep, args.out))
        else:
            print(f'    skipped: no pareto.json in {args.sweep} '
                  '(run: python3 -m formation_core sweep)')
    if want('mechanism'):
        print('3/5 mechanism')
        written.append(figure_mechanism(config, args.out))
    if want('trajectories'):
        print('4/5 trajectories')
        written.append(figure_trajectories(config, args.out))
    if want('transfer'):
        print('5/5 sim-to-sim')
        if args.gazebo_results and args.gazebo_config:
            written.append(figure_sim_to_sim(
                args.gazebo_results, EpisodeConfig.from_yaml(args.gazebo_config),
                args.out))
        else:
            print('    skipped: needs --gazebo-results and --gazebo-config')

    print('\nwrote:')
    for path in written:
        if path:
            print(f'  {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
