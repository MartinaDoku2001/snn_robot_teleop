"""Command-line entry points for single episodes and seeded suites."""

from __future__ import annotations

import argparse
import os

from .config import ComponentConfig, EpisodeConfig, SuiteConfig
from .metrics import write_rows_csv
from .runner import config_path, run_episode, run_suite


def _policy_override(args):
    if not args.policy:
        return {}
    params = {}
    if args.k is not None:
        params['k'] = args.k
    if args.p is not None:
        params['p'] = args.p
    if args.delta is not None:
        params['delta'] = args.delta
    return {'policy': ComponentConfig(args.policy, params)}


def _add_policy_args(parser):
    parser.add_argument(
        '--policy', choices=['always', 'periodic', 'random', 'event_triggered'],
        help='override the transmission policy')
    parser.add_argument('--k', type=int, help='periodic: transmit every k-th step')
    parser.add_argument('--p', type=float, help='random: transmit probability')
    parser.add_argument('--delta', type=float, help='event_triggered: threshold (m)')


def main_run(argv=None):
    """Run ONE episode, write its CSV and (optionally) its figures."""
    parser = argparse.ArgumentParser(description=main_run.__doc__)
    parser.add_argument(
        '--config', default=config_path('default.yaml'), help='episode YAML')
    parser.add_argument('--seed', type=int, help='override the seed')
    parser.add_argument('--duration', type=float, help='override the duration (s)')
    parser.add_argument('--out', default='results/episode', help='output directory')
    parser.add_argument('--plot', action='store_true', help='write figures too')
    _add_policy_args(parser)
    args = parser.parse_args(argv)

    config = EpisodeConfig.from_yaml(args.config)
    overrides = _policy_override(args)
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.duration is not None:
        overrides['duration'] = args.duration
    if overrides:
        config = config.with_overrides(**overrides)

    os.makedirs(args.out, exist_ok=True)
    result = run_episode(config)
    csv_path = result.write_csv(os.path.join(args.out, 'episode.csv'))
    config.to_yaml(os.path.join(args.out, 'config.yaml'))

    metrics = result.metrics
    print(f"policy            {metrics['policy']}")
    print(f"controller        {metrics['controller']}")
    print(f"steps             {metrics['steps']} ({metrics['duration']:.1f} s)")
    print(f"formation RMS     {metrics['formation_rms']:.4f} m")
    print(f"formation max     {metrics['formation_max']:.4f} m")
    print(f"lateral RMS       {metrics['lateral_rms']:.4f} m")
    print(f"heading RMS       {metrics['heading_rms']:.4f} rad")
    print(f"leader path RMS   {metrics['path_rms']:.4f} m")
    print(f"comm rate         {metrics['comm_rate']:.4f} ({metrics['messages']} messages)")
    print(f"AoI mean/max      {metrics['aoi_mean']:.2f} / {metrics['aoi_max']:.0f} steps")
    print(f"wrote             {csv_path}")

    if args.plot:
        from .plotting import plot_demand_profile, plot_error_timeseries, plot_trajectory
        from .paths import make_path
        print('wrote            ', plot_error_timeseries(
            result, os.path.join(args.out, 'errors.png')))
        print('wrote            ', plot_trajectory(
            result, os.path.join(args.out, 'trajectory.png'),
            reference_path=make_path(config.path.name, **config.path.params)))
        print('wrote            ', plot_demand_profile(
            result, os.path.join(args.out, 'demand.png')))
    return 0


def main_suite(argv=None):
    """Run a seeded suite and report mean +- 95% CI across seeds."""
    parser = argparse.ArgumentParser(description=main_suite.__doc__)
    parser.add_argument(
        '--suite', default=config_path('eval_suite.yaml'), help='suite YAML')
    parser.add_argument('--out', default='results/suite', help='output directory')
    _add_policy_args(parser)
    args = parser.parse_args(argv)

    suite = SuiteConfig.from_yaml(args.suite)
    overrides = _policy_override(args)
    os.makedirs(args.out, exist_ok=True)

    label = overrides['policy'].describe() if overrides else suite.episode.policy.describe()
    print(f'suite {suite.name!r}: {len(suite.seeds)} seeds, policy {label}')
    results = run_suite(
        suite, label=label, overrides=overrides,
        progress=lambda i, n, r: print(
            f"  seed {r.config.seed}: formation RMS {r.metrics['formation_rms']:.4f} m, "
            f"comm rate {r.metrics['comm_rate']:.4f}"))

    summary = results.write_summary_csv(os.path.join(args.out, f'{suite.name}_episodes.csv'))
    aggregate = results.aggregate()
    write_rows_csv(os.path.join(args.out, f'{suite.name}_aggregate.csv'), [aggregate])

    print('\nacross seeds (mean +- 95% CI):')
    for key in ('formation_rms', 'formation_max', 'lateral_rms', 'heading_rms',
                'path_rms', 'comm_rate', 'aoi_mean'):
        print(f"  {key:16s} {aggregate[f'{key}_mean']:.4f} +- {aggregate[f'{key}_ci95']:.4f}")
    print(f'\nwrote {summary}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main_run())
