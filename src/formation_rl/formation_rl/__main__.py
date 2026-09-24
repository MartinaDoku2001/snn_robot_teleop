"""``python3 -m formation_rl <command>``: train and evaluate the RL controller.

    python3 -m formation_rl train --out results/rl
    python3 -m formation_rl train --out results/rl --total-steps 100000
    python3 -m formation_rl benchmark --weights results/rl/actor.pt
    python3 -m formation_rl figures   --out results/figures_phase2
"""

import sys

USAGE = """usage: python3 -m formation_rl <command> [options]

commands:
  train      train the PPO actor on the fast twin, write actor.pt
  benchmark  compare a trained actor against the analytic controller
  figures    the Phase 2 figure set: what the system is now, and how the two
             controllers compare

Add --help after a command for its options.
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(USAGE)
        return 0 if argv else 1

    command, rest = argv[0], argv[1:]
    if command == 'train':
        from .cli import main_train
        return main_train(rest)
    if command == 'benchmark':
        from .cli import main_benchmark
        return main_benchmark(rest)
    if command == 'figures':
        from .figures import main as figures_main
        return figures_main(rest)

    print(f'unknown command {command!r}\n\n{USAGE}', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
