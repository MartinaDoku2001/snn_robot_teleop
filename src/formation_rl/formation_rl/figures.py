"""Figures for the CURRENT state of the work: Phase 2, one centralized controller.

These are not the Phase 1 figures. Those explain the transmission-policy result
(``formation_core.figures``, and they were measured under the v1.x split
controller, so they are stale). These explain what the system IS now and how
the two controllers that drive it compare:

    1. architecture   what the system is: one controller, two uplinks
    2. controllers    analytic vs learned, on both suites
    3. tradeoff       the honest version: what the learned policy gives up
    4. trajectories   why it gives that up -- it cuts corners
    5. training       PPO converging past the analytic baseline
    6. sim_to_sim     the same two controllers, measured in Gazebo (optional)

    python3 -m formation_rl figures --out results/figures_phase2

Figures 2-4 run the episodes they need on the fast twin. Figure 5 reads the
training history; figure 6 reads Gazebo run directories and is skipped if they
are not given.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from formation_core.config import ComponentConfig, SuiteConfig
from formation_core.figures import SCALE, save_figure
from formation_core.metrics import mean_ci
from formation_core.paths import make_path
from formation_core.plotting import (
    GRID,
    INK,
    INK_SECONDARY,
    SERIES_COLORS,
    SURFACE,
    _style_axes,
)
from formation_core.runner import config_path, run_episode, run_suite

import matplotlib.pyplot as plt

#: The two entities every figure here compares. Fixed colour and marker, never
#: cycled, so a controller keeps its identity across the whole set. The pair is
#: validated for colour-vision deficiency against the light surface.
CONTROLLER_STYLE = {
    'analytic': (SERIES_COLORS[0], 'o'),
    'rl': (SERIES_COLORS[1], 's'),
}
CONTROLLER_LABEL = {
    'analytic': 'analytic',
    'rl': 'learned (PPO)',
}
SUITES = ('eval', 'stress')


def _suite(name):
    path = name if os.path.exists(name) else config_path(f'{name}_suite.yaml')
    return SuiteConfig.from_yaml(path)


def _controller(name, weights):
    params = {'weights': weights} if name == 'rl' else {}
    return ComponentConfig(name, params)


def measure(suites, weights, progress=print):
    """Run both controllers over every suite and return one row per pair."""
    import formation_rl  # noqa: F401  -- registers 'rl'

    rows = []
    for suite_name in suites:
        suite = _suite(suite_name)
        for controller in ('analytic', 'rl'):
            results = run_suite(
                suite, label=controller,
                overrides={'controller': _controller(controller, weights)})
            row = {'suite': suite.name, 'controller': controller,
                   'seeds': len(results.episodes)}
            for key in ('formation_rms', 'formation_max', 'path_rms',
                        'lateral_rms', 'heading_rms', 'distance'):
                row[key], row[f'{key}_ci'] = mean_ci(results.values(key))
            rows.append(row)
            progress(f"  {suite.name:7s} {controller:9s} "
                     f"formation {row['formation_rms']:.4f} "
                     f"path {row['path_rms']:.4f}")
    return rows


# ------------------------------------------------------------ 1. architecture

def figure_architecture(out_dir):
    """What the system is now. No data -- this is the structural change.

    Laid out on an explicit grid with reserved lanes for the arrows, because
    the first version routed them through the boxes.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.6)
    ax.axis('off')
    fig.patch.set_facecolor(SURFACE)

    controller_c, comm_c, robot_c = SERIES_COLORS[1], SERIES_COLORS[2], SERIES_COLORS[0]

    # Rows: robots 0.4-1.5, comm 2.6-3.8, controller 4.9-6.2.
    # Columns: robot boxes 0.4-5.2 and 7.3-12.1, comm boxes inset so the
    # command arrows have a clear lane down the outside.
    ROBOT_Y, COMM_Y, CTRL_Y = (0.4, 1.5), (2.6, 3.8), (4.9, 6.2)
    LEFT, RIGHT = (0.4, 5.2), (7.3, 12.1)
    COMM_L, COMM_R = (1.7, 5.2), (7.3, 10.8)
    UP_LANE = {'left': 3.4, 'right': 9.1}
    DOWN_LANE = {'left': 1.05, 'right': 11.45}

    def box(span, yspan, title, lines, color):
        x0, x1 = span
        y0, y1 = yspan
        for fill, alpha in ((color, 0.10), ('none', 1.0)):
            ax.add_patch(FancyBboxPatch(
                (x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.08',
                linewidth=2.0, edgecolor=color, facecolor=fill, alpha=alpha,
                zorder=3))
        mid = (x0 + x1) / 2
        ax.text(mid, y1 - 0.30, title, ha='center', va='top', color=INK,
                fontsize=11 * SCALE, fontweight='bold', zorder=5)
        for i, line in enumerate(lines):
            ax.text(mid, y1 - 0.66 - 0.32 * i, line, ha='center', va='top',
                    color=INK_SECONDARY, fontsize=8.5 * SCALE, zorder=5)

    def arrow(x0, y0, x1, y1, color, dashed=False):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=17,
            linewidth=2.0, color=color, zorder=2,
            linestyle=(0, (5, 3)) if dashed else '-'))

    box((0.9, 11.6), CTRL_Y, 'ONE centralized controller',
        ['observation 16 (both robots)   \u2192   action 4 (both robots)',
         'analytic   |   learned (PPO)   |   spiking actor (next)'], controller_c)
    for side, span, label in (('left', COMM_L, 'robot1'), ('right', COMM_R, 'robot2')):
        box(span, COMM_Y, f'{label} comm interface',
            ['own policy, predictor, estimate',
             'Channel hook \u2014 Phase 3 fills it'], comm_c)
    box(LEFT, ROBOT_Y, 'robot1  (leader)', ['/robot1/odom    /robot1/cmd_vel'], robot_c)
    box(RIGHT, ROBOT_Y, 'robot2  (follower)', ['/robot2/odom    /robot2/cmd_vel'], robot_c)

    for side in ('left', 'right'):
        x = UP_LANE[side]
        arrow(x, ROBOT_Y[1], x, COMM_Y[0], comm_c)              # state in
        arrow(x, COMM_Y[1], x, CTRL_Y[0], comm_c)               # estimate up
        # Low in the lane; the caption below owns the upper half of it.
        ax.text(x + 0.16, COMM_Y[1] + 0.30, 'estimate + age',
                ha='left', va='center', color=INK_SECONDARY,
                fontsize=9 * SCALE, zorder=5)
        d = DOWN_LANE[side]
        arrow(d, CTRL_Y[0], d, ROBOT_Y[1], controller_c, dashed=True)
        ax.text(d + (-0.22 if side == 'left' else 0.22),
                (ROBOT_Y[1] + CTRL_Y[0]) / 2, 'cmd_vel\n(reliable)',
                ha='right' if side == 'left' else 'left', va='center',
                rotation=90, color=INK_SECONDARY, fontsize=8.5 * SCALE, zorder=5)

    ax.text(6.25, CTRL_Y[0] - 0.32,
            'each robot decides WHEN to report  \u2014  '
            'Phase 4 learns that decision',
            ha='center', va='center', color=INK_SECONDARY,
            fontsize=9.5 * SCALE, style='italic', zorder=5)
    ax.set_title('Phase 2: one controller drives both robots \u2014 the '
                 'transmission policy stays per-robot',
                 color=INK, fontsize=12 * SCALE, loc='left', pad=12)
    return save_figure(fig, out_dir, '01_architecture')


# ------------------------------------------------------------- 2. controllers

def figure_controllers(rows, out_dir):
    """Analytic vs learned on the two measures that matter, side by side.

    Two panels rather than one chart with two y axes: formation error and path
    error are different measures and a shared axis would misrepresent both.
    """
    panels = (('formation_rms', 'formation RMS error (m)', 'holding the slot'),
              ('path_rms', 'leader path RMS error (m)', 'staying on the route'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    suites = [s for s in SUITES if any(r['suite'].startswith(s[:4]) for r in rows)]
    x = np.arange(len(suites))
    width = 0.34

    for ax, (key, ylabel, subtitle) in zip(axes, panels):
        for i, controller in enumerate(('analytic', 'rl')):
            color, _ = CONTROLLER_STYLE[controller]
            values, errors = [], []
            for suite in suites:
                row = next(r for r in rows
                           if r['controller'] == controller and r['suite'].startswith(suite[:4]))
                values.append(row[key])
                errors.append(row[f'{key}_ci'])
            offset = (i - 0.5) * width
            ax.bar(x + offset, values, width * 0.92, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=2.0,
                   label=CONTROLLER_LABEL[controller])
            ax.errorbar(x + offset, values, yerr=errors, fmt='none',
                        ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=4)
            for xi, value in zip(x + offset, values):
                ax.annotate(f'{value:.3f}', xy=(xi, value), xytext=(0, 6),
                            textcoords='offset points', ha='center',
                            color=INK, fontsize=9.5 * SCALE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s} suite' for s in suites], fontsize=10 * SCALE)
        _style_axes(ax, '', ylabel, subtitle, scale=SCALE)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, max(ax.get_ylim()[1], 1e-9) * 1.18)

    axes[0].legend(frameon=False, fontsize=10 * SCALE, labelcolor=INK_SECONDARY,
                   loc='upper left')
    fig.suptitle('The learned controller holds formation better and tracks the '
                 'path worse', color=INK, fontsize=12 * SCALE, x=0.02, ha='left',
                 y=1.02)
    fig.text(0.0, -0.04, '8 fixed seeds per bar, mean ± 95% CI',
             color=INK_SECONDARY, fontsize=9 * SCALE)
    fig.tight_layout()
    return save_figure(fig, out_dir, '02_controllers')


# ---------------------------------------------------------------- 3. tradeoff

def figure_tradeoff(rows, out_dir):
    """Both measures on one plane, so the trade is a direction, not two numbers."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    suites = [s for s in SUITES if any(r['suite'].startswith(s[:4]) for r in rows)]

    for suite in suites:
        pair = {r['controller']: r for r in rows if r['suite'].startswith(suite[:4])}
        if len(pair) != 2:
            continue
        start = (pair['analytic']['path_rms'], pair['analytic']['formation_rms'])
        end = (pair['rl']['path_rms'], pair['rl']['formation_rms'])
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='-|>', color=GRID, linewidth=2.4,
                                    mutation_scale=18, shrinkA=10, shrinkB=10))
        ax.annotate(f'{suite} suite', xy=((start[0] + end[0]) / 2,
                                          (start[1] + end[1]) / 2),
                    xytext=(0, 10), textcoords='offset points', ha='center',
                    color=INK_SECONDARY, fontsize=9.5 * SCALE)

    for controller in ('analytic', 'rl'):
        color, marker = CONTROLLER_STYLE[controller]
        points = [r for r in rows if r['controller'] == controller]
        ax.errorbar([r['path_rms'] for r in points],
                    [r['formation_rms'] for r in points],
                    xerr=[r['path_rms_ci'] for r in points],
                    yerr=[r['formation_rms_ci'] for r in points],
                    fmt=marker, markersize=13, color=color, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=2,
                    elinewidth=1.2, capsize=3, linestyle='none',
                    label=CONTROLLER_LABEL[controller])

    _style_axes(ax, 'leader path RMS error (m)  —  staying on the route',
                'formation RMS error (m)  —  holding the slot',
                'What the learned controller trades away', scale=SCALE)
    ax.annotate('better  ↙', xy=(0.02, 0.04), xycoords='axes fraction',
                color=INK_SECONDARY, fontsize=10 * SCALE, style='italic')
    ax.legend(frameon=False, fontsize=10 * SCALE, labelcolor=INK_SECONDARY,
              loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.text(0.0, -0.03,
             'Down is a better formation; left is a better racing line. The '
             'learned policy moves down and right: it cuts corners.',
             color=INK_SECONDARY, fontsize=9 * SCALE)
    return save_figure(fig, out_dir, '03_tradeoff')


# ------------------------------------------------------------ 4. trajectories

def figure_trajectories(weights, out_dir, suite='eval'):
    """Why it trades: the learned leader takes a tighter line through the caps."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    import formation_rl  # noqa: F401

    config = _suite(suite).episode
    results = {
        name: run_episode(
            config.with_overrides(controller=_controller(name, weights)), label=name)
        for name in ('analytic', 'rl')
    }
    reference = make_path(config.path.name, **config.path.params)
    ramp = LinearSegmentedColormap.from_list(
        'formation_error', ['#dfe8f5', '#8fb3e0', '#3f6fb5', '#1b3566'])
    pooled = np.concatenate(
        [r.recorder.column('error_euclidean') for r in results.values()])
    norm = Normalize(vmin=0.0, vmax=float(np.nanpercentile(pooled, 90)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    points = reference.points
    closed = np.vstack([points, points[:1]])

    for ax, (name, result) in zip(axes, results.items()):
        recorder = result.recorder
        ax.plot(closed[:, 0], closed[:, 1], color=GRID, linewidth=6.0, zorder=1,
                solid_capstyle='round')
        ax.plot(recorder.column('leader_x'), recorder.column('leader_y'),
                color=INK_SECONDARY, linewidth=1.2, alpha=0.55, zorder=2)
        track = np.column_stack(
            [recorder.column('follower_x'), recorder.column('follower_y')])
        segments = np.stack([track[:-1], track[1:]], axis=1)
        collection = LineCollection(segments, cmap=ramp, norm=norm, linewidth=2.6,
                                    zorder=3, capstyle='round')
        collection.set_array(recorder.column('error_euclidean')[:-1])
        ax.add_collection(collection)
        ax.set_aspect('equal')
        _style_axes(ax, 'x (m)', 'y (m)' if ax is axes[0] else '', scale=SCALE)
        ax.annotate(
            f"{CONTROLLER_LABEL[name]}\n"
            f"formation {result.metrics['formation_rms']:.3f} m   "
            f"path {result.metrics['path_rms']:.3f} m",
            xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom',
            color=INK, fontsize=10 * SCALE)

    span = max(np.abs(np.column_stack(
        [r.recorder.column('follower_x'), r.recorder.column('follower_y')])).max()
        for r in results.values()) * 1.08
    for ax in axes:
        ax.set_xlim(-span, span)
        ax.set_ylim(-span * 0.46, span * 0.46)

    bar = fig.colorbar(collection, ax=axes, fraction=0.03, pad=0.02,
                       extend='max', shrink=0.62)
    bar.set_label('formation error (m)', color=INK_SECONDARY, fontsize=10 * SCALE)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=9 * SCALE)
    bar.outline.set_visible(False)
    fig.suptitle(f'Follower track coloured by formation error; grey band is the '
                 f'reference path ({suite} suite, seed {config.seed})',
                 color=INK, fontsize=12 * SCALE, x=0.02, ha='left', y=1.06)
    return save_figure(fig, out_dir, '04_trajectories')


# ---------------------------------------------------------------- 5. training

def figure_training(history_path, baseline, out_dir, window=9):
    """PPO converging past the hand-written controller it has to beat."""
    with open(history_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    steps = np.array([float(r['steps']) for r in rows])
    reward = np.array([float(r['mean_reward']) for r in rows])
    smooth = np.convolve(reward, np.ones(window) / window, mode='valid')
    smooth_steps = steps[window - 1:]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    color = CONTROLLER_STYLE['rl'][0]
    ax.plot(steps, reward, color=color, linewidth=1.0, alpha=0.30, zorder=2)
    ax.plot(smooth_steps, smooth, color=color, linewidth=2.4, zorder=3,
            label=CONTROLLER_LABEL['rl'])
    ax.axhline(baseline, color=CONTROLLER_STYLE['analytic'][0], linewidth=2.0,
               linestyle='--', zorder=3, label=CONTROLLER_LABEL['analytic'])
    ax.annotate(f'analytic baseline  {baseline:+.3f}',
                xy=(steps[-1] * 0.99, baseline), xytext=(-8, 10),
                textcoords='offset points', ha='right',
                color=CONTROLLER_STYLE['analytic'][0], fontsize=9.5 * SCALE,
                fontweight='bold')
    crossing = smooth_steps[smooth >= baseline]
    if crossing.size:
        ax.axvline(crossing[0], color=GRID, linewidth=1.6, zorder=1)
        ax.annotate(f'passes it at {crossing[0] / 1000:.0f}k steps',
                    xy=(crossing[0], baseline), xytext=(10, -26),
                    textcoords='offset points', color=INK_SECONDARY,
                    fontsize=9.5 * SCALE)

    _style_axes(ax, 'environment steps', 'mean reward per step',
                'Training the centralized controller', scale=SCALE)
    # Lower right: the curve has climbed away from it by then, and the
    # crossing annotation now lives up beside the baseline.
    ax.legend(frameon=False, fontsize=10 * SCALE, labelcolor=INK_SECONDARY,
              loc='lower right')
    fig.text(0.0, -0.04,
             f'Rolling mean over {window} updates; the faint line is per-update.',
             color=INK_SECONDARY, fontsize=9 * SCALE)
    return save_figure(fig, out_dir, '05_training')


# -------------------------------------------------------------- 6. sim-to-sim

def _gazebo_metrics(run_dir):
    with open(os.path.join(run_dir, 'metrics.csv')) as handle:
        return next(csv.DictReader(handle))


def figure_sim_to_sim(gazebo_dirs, rows, out_dir):
    """The same two controllers, measured on the real simulator.

    Paired dots, not bars: each Gazebo point is a single episode, and bars
    would imply a precision one run does not have.
    """
    entries = []
    for controller, run_dir in gazebo_dirs.items():
        if not run_dir or not os.path.exists(os.path.join(run_dir, 'metrics.csv')):
            continue
        metrics = _gazebo_metrics(run_dir)
        twin = next((r for r in rows if r['controller'] == controller
                     and r['suite'].startswith('eval')), None)
        if twin is None:
            continue
        entries.append({'controller': controller,
                        'twin': twin['formation_rms'],
                        'gazebo': float(metrics['formation_rms'])})
    if not entries:
        return None

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    y = np.arange(len(entries))
    for i, entry in enumerate(entries):
        color = CONTROLLER_STYLE[entry['controller']][0]
        ax.plot([entry['twin'], entry['gazebo']], [i, i], color=GRID,
                linewidth=2.5, zorder=1, solid_capstyle='round')
        ax.scatter([entry['twin']], [i], s=150, marker='o', zorder=3,
                   color=color, edgecolor=SURFACE, linewidth=2)
        ax.scatter([entry['gazebo']], [i], s=150, marker='D', zorder=3,
                   facecolor=SURFACE, edgecolor=color, linewidth=2.4)
        # One label per row, clear to the right of both markers: stacking them
        # above and below put the lower row's text through the x axis.
        ax.annotate(f"fast twin {entry['twin']:.3f}      "
                    f"Gazebo {entry['gazebo']:.3f}",
                    xy=(max(entry['twin'], entry['gazebo']), i), xytext=(16, 0),
                    textcoords='offset points', va='center', ha='left',
                    color=INK_SECONDARY, fontsize=9.5 * SCALE)

    ax.set_yticks(y)
    ax.set_yticklabels([CONTROLLER_LABEL[e['controller']] for e in entries],
                       fontsize=10 * SCALE)
    ax.set_ylim(len(entries) - 0.45, -0.55)
    _style_axes(ax, 'formation RMS error (m)', '',
                'The same controllers, measured in Gazebo', scale=SCALE)
    ax.grid(axis='y', visible=False)
    low = min(min(e['twin'], e['gazebo']) for e in entries)
    high = max(max(e['twin'], e['gazebo']) for e in entries)
    pad = max(high - low, 1e-3)
    ax.set_xlim(max(low - pad * 0.35, 0.0), high + pad * 1.5)
    ax.legend(handles=[
        Line2D([], [], marker='o', linestyle='none', markersize=11,
               color=INK_SECONDARY, label='fast twin (8 seeds)'),
        Line2D([], [], marker='D', linestyle='none', markersize=11,
               markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY,
               markeredgewidth=2, color=INK_SECONDARY,
               label='Gazebo (one 25 s episode)'),
    ], frameon=False, fontsize=9.5 * SCALE, labelcolor=INK_SECONDARY,
        loc='lower right', ncol=2)
    fig.text(0.0, -0.12,
             'Perfect communications in both. The backends disagree by about '
             '10%, and rank the two controllers the same way.',
             color=INK_SECONDARY, fontsize=9 * SCALE)
    return save_figure(fig, out_dir, '06_sim_to_sim')


# -------------------------------------------------------------------- entry

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='results/figures_phase2')
    parser.add_argument('--weights', default='results/rl/actor.pt')
    parser.add_argument('--history', default='',
                        help='training_history.csv (default: beside --weights)')
    parser.add_argument('--suites', nargs='+', default=list(SUITES))
    parser.add_argument('--gazebo-analytic', default='',
                        help='a Gazebo run directory for the analytic controller')
    parser.add_argument('--gazebo-rl', default='',
                        help='a Gazebo run directory for the learned controller')
    parser.add_argument('--only', nargs='+', default=None,
                        help='subset of: architecture controllers tradeoff '
                             'trajectories training sim_to_sim')
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    wanted = set(args.only) if args.only else None

    def want(name):
        return wanted is None or name in wanted

    written = []
    rows = None
    needs_rows = any(want(n) for n in ('controllers', 'tradeoff', 'sim_to_sim'))
    if needs_rows:
        print('measuring both controllers on', ', '.join(args.suites))
        rows = measure(args.suites, args.weights)

    if want('architecture'):
        print('1/6 architecture')
        written.append(figure_architecture(args.out))
    if want('controllers'):
        print('2/6 controllers')
        written.append(figure_controllers(rows, args.out))
    if want('tradeoff'):
        print('3/6 tradeoff')
        written.append(figure_tradeoff(rows, args.out))
    if want('trajectories'):
        print('4/6 trajectories')
        written.append(figure_trajectories(args.weights, args.out))
    if want('training'):
        print('5/6 training')
        history = args.history or os.path.join(
            os.path.dirname(args.weights), 'training_history.csv')
        if os.path.exists(history):
            baseline = _analytic_reward()
            written.append(figure_training(history, baseline, args.out))
        else:
            print(f'    skipped: no training history at {history}')
    if want('sim_to_sim'):
        print('6/6 sim-to-sim')
        path = figure_sim_to_sim(
            {'analytic': args.gazebo_analytic, 'rl': args.gazebo_rl}, rows, args.out)
        if path is None:
            print('    skipped: needs --gazebo-analytic and --gazebo-rl')
        else:
            written.append(path)

    print('\nwrote:')
    for path in written:
        if path:
            print(f'  {path}')
    return 0


def _analytic_reward():
    """Mean per-step reward of the analytic controller: what PPO has to beat."""
    result = run_episode(_suite('eval').episode)
    return float(np.nanmean(result.recorder.column('reward')))


if __name__ == '__main__':
    raise SystemExit(main())
