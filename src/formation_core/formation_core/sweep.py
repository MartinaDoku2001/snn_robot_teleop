"""Pareto sweep: formation error against communication rate.

Runs the analytic controller over the baseline transmission policies on a
seeded suite and produces the Pareto plot. It also CHECKS the premise the whole
project rests on, before any learning exists:

1. error rises as the communication rate falls (the trade-off is real);
2. event-triggered beats periodic AND random at a matched rate (timing
   matters, so there is something for a learned scheduler to learn).

If either check fails the task is mis-specified, and ``--check`` makes that a
non-zero exit code rather than a figure nobody reads.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import ComponentConfig, SuiteConfig
from .metrics import mean_ci, write_rows_csv
from .runner import config_path, run_suite

#: Swept parameter ranges. Chosen so all three families span roughly the same
#: range of communication rates (about 0.01 to 1.0), which is what makes a
#: matched-rate comparison possible.
PERIODIC_K = (1, 2, 4, 8, 16, 32, 64, 100)
RANDOM_P = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.01)
EVENT_DELTA = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3)


def sweep_family(suite, family, values, error_key='formation_rms', progress=None):
    """Run one policy family across ``values`` and return Pareto points."""
    points = []
    for value in values:
        if family == 'periodic':
            policy = ComponentConfig('periodic', {'k': int(value)})
            label = f'k={int(value)}'
        elif family == 'random':
            policy = ComponentConfig('random', {'p': float(value)})
            label = f'p={value:g}'
        elif family == 'event_triggered':
            policy = ComponentConfig('event_triggered', {'delta': float(value)})
            label = f'd={value:g}'
        elif family == 'always':
            policy = ComponentConfig('always')
            label = ''
        else:
            raise ValueError(f'unknown policy family {family!r}')

        results = run_suite(suite, label=policy.describe(), overrides={'policy': policy})
        rate_mean, rate_ci = mean_ci(results.values('comm_rate'))
        error_mean, error_ci = mean_ci(results.values(error_key))
        aoi_mean, _ = mean_ci(results.values('aoi_mean'))
        max_mean, _ = mean_ci(results.values('formation_max'))
        point = {
            'family': family,
            'value': float(value),
            'label': label,
            'policy': policy.describe(),
            'rate': rate_mean,
            'rate_ci': rate_ci,
            'error': error_mean,
            'error_ci': error_ci,
            'formation_max': max_mean,
            'aoi_mean': aoi_mean,
            'episodes': len(results.episodes),
            'messages': float(np.mean(results.values('messages'))),
        }
        points.append(point)
        if progress is not None:
            progress(point)
    return points


def interpolate_error_at_rate(points, rate):
    """Error of a family at a given communication rate (log-rate interpolation).

    Used for matched-rate comparisons: periodic and event-triggered never land
    on exactly the same rates, so compare them on a common one.
    """
    usable = sorted(
        [p for p in points if p['rate'] > 0 and np.isfinite(p['error'])],
        key=lambda p: p['rate'])
    if len(usable) < 2:
        return float('nan')
    rates = np.log([p['rate'] for p in usable])
    errors = [p['error'] for p in usable]
    if not (rates[0] <= np.log(rate) <= rates[-1]):
        return float('nan')  # never extrapolate
    return float(np.interp(np.log(rate), rates, errors))


def check_premise(series, verbose=True):
    """Validate the two claims the task depends on. Returns (ok, list of lines)."""
    lines = []
    ok = True

    # --- 1. the trade-off exists: less communication => more error
    for family in ('periodic', 'random', 'event_triggered'):
        points = sorted(series.get(family, []), key=lambda p: p['rate'])
        if len(points) < 3:
            continue
        rates = np.array([p['rate'] for p in points])
        errors = np.array([p['error'] for p in points])
        # Spearman-style monotonicity: error should fall as rate rises.
        correlation = float(np.corrcoef(np.log(rates), errors)[0, 1])
        worst, best = errors[0], errors[-1]
        passed = correlation < -0.3 and worst > best
        ok &= passed
        lines.append(
            f"[{'PASS' if passed else 'FAIL'}] {family}: error falls as rate rises "
            f'(corr(log rate, error) = {correlation:+.2f}, '
            f'{worst:.3f} m at rate {rates[0]:.3f} -> {best:.3f} m at rate {rates[-1]:.3f})')

    # --- 2. event-triggered beats the open-loop baselines at matched rate
    event = series.get('event_triggered', [])
    for baseline_name in ('periodic', 'random'):
        baseline = series.get(baseline_name, [])
        if not event or not baseline:
            continue
        shared_low = max(
            min(p['rate'] for p in event if p['rate'] > 0),
            min(p['rate'] for p in baseline if p['rate'] > 0))
        shared_high = min(
            max(p['rate'] for p in event), max(p['rate'] for p in baseline))
        if not shared_low < shared_high:
            lines.append(f'[FAIL] {baseline_name}: no overlapping rate range to compare')
            ok = False
            continue
        test_rates = np.exp(np.linspace(
            np.log(shared_low), np.log(shared_high), 7))[1:-1]
        wins = 0
        comparisons = []
        for rate in test_rates:
            e_err = interpolate_error_at_rate(event, rate)
            b_err = interpolate_error_at_rate(baseline, rate)
            if not (np.isfinite(e_err) and np.isfinite(b_err)):
                continue
            wins += int(e_err < b_err)
            comparisons.append((rate, e_err, b_err))
        passed = bool(comparisons) and wins == len(comparisons)
        ok &= passed
        lines.append(
            f"[{'PASS' if passed else 'FAIL'}] event_triggered beats {baseline_name} "
            f'at matched rate ({wins}/{len(comparisons)} rates)')
        if verbose:
            for rate, e_err, b_err in comparisons:
                lines.append(
                    f'         rate {rate:6.3f}: event {e_err:.4f} m vs '
                    f'{baseline_name} {b_err:.4f} m '
                    f'({100 * (b_err - e_err) / b_err:+.0f}% better)')
    return ok, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--suite', default=config_path('eval_suite.yaml'),
        help='suite YAML (default: the packaged evaluation suite)')
    parser.add_argument('--out', default='results/sweep', help='output directory')
    parser.add_argument(
        '--error-key', default='formation_rms',
        help='metric on the y axis (default: RMS formation error)')
    parser.add_argument(
        '--families', nargs='+',
        default=['periodic', 'random', 'event_triggered'],
        help='policy families to sweep')
    parser.add_argument(
        '--check', action='store_true',
        help='exit non-zero if the task premise does not hold')
    parser.add_argument('--no-plot', action='store_true', help='skip the figure')
    args = parser.parse_args(argv)

    suite = SuiteConfig.from_yaml(args.suite)
    os.makedirs(args.out, exist_ok=True)

    values = {'periodic': PERIODIC_K, 'random': RANDOM_P, 'event_triggered': EVENT_DELTA}
    series = {}
    print(f'Sweeping on suite {suite.name!r}: {len(suite.seeds)} seeds x '
          f'{suite.episode.steps} steps per episode')
    for family in args.families:
        def report(point, family=family):
            print(f'  {family:16s} {point["label"]:>8s}  '
                  f'rate {point["rate"]:.4f}  '
                  f'{args.error_key} {point["error"]:.4f} +- {point["error_ci"]:.4f}  '
                  f'AoI {point["aoi_mean"]:.1f}')
        series[family] = sweep_family(
            suite, family, values[family], error_key=args.error_key, progress=report)

    rows = [p for points in series.values() for p in points]
    csv_path = write_rows_csv(os.path.join(args.out, 'pareto.csv'), rows)
    with open(os.path.join(args.out, 'pareto.json'), 'w') as handle:
        json.dump({'suite': suite.name, 'seeds': suite.seeds, 'points': rows}, handle, indent=2)
    print(f'\nWrote {csv_path}')

    ok, lines = check_premise(series)
    print('\nTask premise:')
    for line in lines:
        print('  ' + line)

    if not args.no_plot:
        from .plotting import plot_pareto
        plot_path = plot_pareto(
            series, os.path.join(args.out, 'pareto.png'),
            title=f'Formation error vs communication rate ({suite.name} suite, '
                  f'{len(suite.seeds)} seeds, mean +- 95% CI)')
        print(f'\nWrote {plot_path}')

    if args.check and not ok:
        print('\nPREMISE CHECK FAILED')
        return 1
    print('\nPREMISE CHECK PASSED' if ok else '\n(premise check reported failures)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
