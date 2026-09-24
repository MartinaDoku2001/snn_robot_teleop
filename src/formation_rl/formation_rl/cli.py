"""Command line: train the actor, and benchmark it against the analytic baseline.

The benchmark runs both controllers over the SAME seeded suites the analytic
baseline was measured on, through ``formation_core``'s own runner, so the two
numbers are produced by identical code and are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os

from formation_core.config import ComponentConfig, SuiteConfig
from formation_core.metrics import mean_ci, write_rows_csv
from formation_core.runner import config_path, run_suite

from .ppo import PPOConfig, train

#: Metrics reported side by side for the two controllers.
REPORTED = ('formation_rms', 'formation_max', 'lateral_rms', 'path_rms', 'heading_rms')


def _suite_path(name):
    """Accept a bare suite name or a full path."""
    if os.path.exists(name):
        return name
    return config_path(name if name.endswith('.yaml') else f'{name}_suite.yaml')


def main_train(argv=None):
    parser = argparse.ArgumentParser(description='Train the PPO actor.')
    parser.add_argument('--suite', default='eval',
                        help='suite whose episode config to train on')
    parser.add_argument('--out', default='results/rl', help='output directory')
    parser.add_argument('--total-steps', type=int, default=None)
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--rollout-steps', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--duration', type=float, default=None,
                        help='shorten training episodes (s); evaluation is unaffected')
    args = parser.parse_args(argv)

    suite = SuiteConfig.from_yaml(_suite_path(args.suite))
    config = suite.episode
    if args.duration is not None:
        config = config.with_overrides(duration=args.duration)

    overrides = {k: v for k, v in (
        ('total_steps', args.total_steps),
        ('num_envs', args.num_envs),
        ('rollout_steps', args.rollout_steps),
        ('learning_rate', args.learning_rate),
        ('seed', args.seed),
    ) if v is not None}
    ppo = PPOConfig(**overrides)

    os.makedirs(args.out, exist_ok=True)
    print(f'training on the {suite.name!r} episode config: '
          f'{config.duration:.0f} s episodes, {ppo.total_steps} steps, '
          f'{ppo.num_envs} envs, batch {ppo.batch_size}')

    def report(record):
        print(f"  update {record['update']:4d}  "
              f"{record['steps']:8d} steps  "
              f"mean reward {record['mean_reward']:+.4f}  "
              f"log_std {record['log_std']:+.2f}  "
              f"{record['elapsed']:6.1f} s", flush=True)

    actor, history = train(config, ppo=ppo, progress=report)

    weights = actor.save(os.path.join(args.out, 'actor.pt'))
    with open(os.path.join(args.out, 'ppo_config.json'), 'w') as handle:
        json.dump({'ppo': ppo.to_dict(), 'suite': suite.name,
                   'episode_duration': config.duration}, handle, indent=2)
    write_rows_csv(os.path.join(args.out, 'training_history.csv'), history)
    config.to_yaml(os.path.join(args.out, 'train_episode_config.yaml'))
    print(f'\nwrote {weights}')
    return 0


def main_benchmark(argv=None):
    parser = argparse.ArgumentParser(
        description='Benchmark the RL controller against the analytic one.')
    parser.add_argument('--weights', default='results/rl/actor.pt')
    parser.add_argument('--suites', nargs='+', default=['eval', 'stress'])
    parser.add_argument('--out', default='results/rl', help='output directory')
    args = parser.parse_args(argv)

    import formation_rl  # noqa: F401  -- registers the 'rl' controller

    rows = []
    for name in args.suites:
        suite = SuiteConfig.from_yaml(_suite_path(name))
        print(f'\n=== {suite.name} suite, {len(suite.seeds)} seeds '
              f'(mean +- 95% CI) ===')
        for controller in (
            ComponentConfig('analytic', {}),
            ComponentConfig('rl', {'weights': args.weights}),
        ):
            results = run_suite(
                suite, label=controller.name, overrides={'controller': controller})
            row = {'suite': suite.name, 'controller': controller.name,
                   'episodes': len(results.episodes)}
            for key in REPORTED:
                mean, ci = mean_ci(results.values(key))
                row[f'{key}_mean'] = mean
                row[f'{key}_ci95'] = ci
            rows.append(row)
            print(f"  {controller.name:9s} " + '  '.join(
                f"{key.replace('_rms', '')} {row[f'{key}_mean']:.4f}"
                f" +- {row[f'{key}_ci95']:.4f}" for key in REPORTED))

    os.makedirs(args.out, exist_ok=True)
    path = write_rows_csv(os.path.join(args.out, 'benchmark.csv'), rows)
    print(f'\nwrote {path}')

    # The headline comparison, stated rather than left to the reader.
    for name in args.suites:
        pair = {r['controller']: r for r in rows if r['suite'].startswith(name[:4])}
        if len(pair) == 2:
            analytic = pair['analytic']['formation_rms_mean']
            learned = pair['rl']['formation_rms_mean']
            verdict = 'beats' if learned < analytic else 'does NOT beat'
            print(f'{name}: rl {learned:.4f} m vs analytic {analytic:.4f} m '
                  f'-- rl {verdict} analytic '
                  f'({100 * (analytic - learned) / analytic:+.1f}%)')
    return 0
