"""Portable entry point: ``python3 -m formation_core <command>``.

The package is installed two ways, and they disagree about where console
scripts belong: colcon/ament puts them in ``lib/formation_core`` (so
``ros2 run formation_core formation-sweep`` works), while pip would normally
put them on PATH. This module works under both, and in a plain checkout with
nothing installed at all.

    python3 -m formation_core run    --policy event_triggered --delta 0.05 --plot
    python3 -m formation_core suite  --suite configs/eval_suite.yaml
    python3 -m formation_core sweep  --check
"""

import sys

USAGE = """usage: python3 -m formation_core <command> [options]

commands:
  run      run a single episode (CSV, optional figures)
  suite    run a seeded suite and report mean +- 95% CI
  sweep    Pareto sweep over transmission policies, with the premise check

Add --help after a command for its options.
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(USAGE)
        return 0 if argv else 1

    command, rest = argv[0], argv[1:]
    if command == 'run':
        from .cli import main_run
        return main_run(rest)
    if command == 'suite':
        from .cli import main_suite
        return main_suite(rest)
    if command == 'sweep':
        from .sweep import main as sweep_main
        return sweep_main(rest)

    print(f'unknown command {command!r}\n\n{USAGE}', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
