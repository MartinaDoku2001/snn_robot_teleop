"""Metrics: control, communication, and aggregation across seeds.

Three families are logged EVERY step and written to CSV:

* **control** -- longitudinal / lateral / heading / Euclidean formation error,
  plus the leader's own path-tracking error;
* **communication** -- transmit flag, age of information, prediction error;
* **state** -- true and estimated poses, commands, reward, so any figure in the
  paper can be rebuilt from the CSV without re-running.

The same recorder is used by the fast twin and by the ROS evaluation node, so
numbers from the two backends are directly comparable.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

#: Two-sided 95% Student-t critical values by degrees of freedom. Keeps
#: confidence intervals correct for the small seed counts used here without
#: pulling in SciPy at runtime.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t95(df):
    """Two-sided 95% t critical value for ``df`` degrees of freedom."""
    if df <= 0:
        return float('nan')
    if df in _T95:
        return _T95[df]
    return 1.96 if df > 30 else _T95[max(_T95)]


def mean_ci(values):
    """Mean and 95% confidence half-width over a sample (e.g. across seeds)."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float('nan'), float('nan')
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0
    sem = float(arr.std(ddof=1) / math.sqrt(n))
    return mean, float(t95(n - 1) * sem)


def _stats(values, prefix):
    """mean / rms / max / std for an error series."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {f'{prefix}_{k}': float('nan') for k in ('mean', 'rms', 'max', 'std')}
    return {
        f'{prefix}_mean': float(np.mean(np.abs(arr))),
        f'{prefix}_rms': float(np.sqrt(np.mean(arr ** 2))),
        f'{prefix}_max': float(np.max(np.abs(arr))),
        f'{prefix}_std': float(np.std(arr)),
    }


#: CSV columns, in order. Frozen alongside the contract so downstream analysis
#: scripts can rely on the layout.
STEP_FIELDS = (
    'step', 'time',
    'leader_x', 'leader_y', 'leader_theta', 'leader_v', 'leader_w',
    'follower_x', 'follower_y', 'follower_theta', 'follower_v', 'follower_w',
    'estimate_x', 'estimate_y', 'estimate_theta', 'estimate_v', 'estimate_w',
    'error_longitudinal', 'error_lateral', 'error_heading', 'error_euclidean',
    'path_error', 'estimate_error', 'prediction_error',
    'transmitted', 'received', 'age',
    'action_v', 'action_w', 'reward',
)


@dataclass
class EpisodeRecorder:
    """Accumulates per-step rows and derives episode metrics."""

    dt: float = 0.05
    rows: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record(self, info):
        """Append one step from an env ``info`` dict."""
        leader = info['leader_state']
        follower = info['follower_state']
        estimate = info['leader_estimate']
        errors = info['errors']
        action = np.asarray(info.get('action', (0.0, 0.0)), dtype=float).reshape(-1)
        self.rows.append({
            'step': int(info['step']),
            'time': float(info['time']),
            'leader_x': leader.x, 'leader_y': leader.y, 'leader_theta': leader.theta,
            'leader_v': leader.v, 'leader_w': leader.w,
            'follower_x': follower.x, 'follower_y': follower.y,
            'follower_theta': follower.theta,
            'follower_v': follower.v, 'follower_w': follower.w,
            'estimate_x': estimate.x, 'estimate_y': estimate.y,
            'estimate_theta': estimate.theta,
            'estimate_v': estimate.v, 'estimate_w': estimate.w,
            'error_longitudinal': errors['longitudinal'],
            'error_lateral': errors['lateral'],
            'error_heading': errors['heading'],
            'error_euclidean': errors['euclidean'],
            'path_error': float(info.get('path_error', float('nan'))),
            'estimate_error': float(info.get('estimate_error', float('nan'))),
            'prediction_error': float(info.get('prediction_error', float('nan'))),
            'transmitted': int(bool(info.get('transmitted', False))),
            'received': int(bool(info.get('received', False))),
            'age': int(info.get('age', 0)),
            'action_v': float(action[0]) if action.size > 0 else float('nan'),
            'action_w': float(action[1]) if action.size > 1 else float('nan'),
            'reward': float(info.get('reward', float('nan'))),
        })

    def __len__(self):
        return len(self.rows)

    def column(self, name):
        return np.array([row[name] for row in self.rows], dtype=float)

    def write_csv(self, path):
        """Write the full per-step time series."""
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(STEP_FIELDS))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        return path

    def metrics(self, comm_stats=None):
        """Derive the episode metrics dict (control + communication families)."""
        out = {}
        out.update(_stats(self.column('error_longitudinal'), 'longitudinal'))
        out.update(_stats(self.column('error_lateral'), 'lateral'))
        out.update(_stats(self.column('error_heading'), 'heading'))
        out.update(_stats(self.column('error_euclidean'), 'formation'))
        out.update(_stats(self.column('path_error'), 'path'))
        out.update(_stats(self.column('estimate_error'), 'estimate'))

        steps = len(self.rows)
        transmitted = self.column('transmitted') if steps else np.zeros(0)
        ages = self.column('age') if steps else np.zeros(0)
        total = int(transmitted.sum()) if steps else 0
        out['steps'] = steps
        out['duration'] = steps * self.dt
        out['messages'] = total
        out['comm_rate'] = float(total / steps) if steps else float('nan')
        out['messages_per_second'] = float(total / (steps * self.dt)) if steps else float('nan')
        out['aoi_mean'] = float(ages.mean()) if steps else float('nan')
        out['aoi_max'] = float(ages.max()) if steps else float('nan')
        out['reward_total'] = float(np.nansum(self.column('reward'))) if steps else 0.0

        transmit_steps = [int(r['step']) for r in self.rows if r['transmitted']]
        intervals = [b - a for a, b in zip(transmit_steps[:-1], transmit_steps[1:])]
        if intervals:
            arr = np.asarray(intervals, dtype=float)
            out['iti_mean'] = float(arr.mean())
            out['iti_std'] = float(arr.std())
            out['iti_min'] = float(arr.min())
            out['iti_max'] = float(arr.max())
        else:
            for key in ('iti_mean', 'iti_std', 'iti_min', 'iti_max'):
                out[key] = float('nan')

        if comm_stats is not None:
            out['comm_rate_policy'] = float(comm_stats.rate)
            out['messages_policy'] = int(comm_stats.transmissions)
        out.update(self.metadata)
        return out


@dataclass
class EpisodeResult:
    """One episode: its config, metrics, and full time series."""

    config: Any
    metrics: Dict[str, Any]
    recorder: EpisodeRecorder
    terminated: bool = False
    truncated: bool = True
    label: str = ''

    def write_csv(self, path):
        return self.recorder.write_csv(path)


@dataclass
class SuiteResult:
    """A suite: per-seed episodes plus aggregates with confidence intervals."""

    name: str
    episodes: List[EpisodeResult] = field(default_factory=list)
    label: str = ''
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def seeds(self):
        return [ep.config.seed for ep in self.episodes]

    def values(self, key):
        return [ep.metrics.get(key, float('nan')) for ep in self.episodes]

    def aggregate(self, keys=None):
        """Mean and 95% CI across seeds for each metric."""
        if not self.episodes:
            return {}
        keys = keys or [
            k for k, v in self.episodes[0].metrics.items() if isinstance(v, (int, float))]
        out = {'name': self.name, 'label': self.label, 'episodes': len(self.episodes)}
        for key in keys:
            mean, ci = mean_ci(self.values(key))
            out[f'{key}_mean'] = mean
            out[f'{key}_ci95'] = ci
        out.update(self.meta)
        return out

    def write_summary_csv(self, path):
        """One row per episode, with the aggregate appended as the last row."""
        rows = []
        for ep in self.episodes:
            row = {'suite': self.name, 'label': self.label, 'seed': ep.config.seed}
            row.update({k: v for k, v in ep.metrics.items() if isinstance(v, (int, float, str))})
            rows.append(row)
        if not rows:
            return path
        fieldnames = list(rows[0].keys())
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        return path


def write_rows_csv(path, rows, fieldnames=None):
    """Write a list of dicts to CSV (used by the sweep)."""
    rows = list(rows)
    if not rows:
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return path
