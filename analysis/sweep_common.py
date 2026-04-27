"""Shared utilities for analysing texture-sweep JSONL files.

One row per (variant_id, episode_idx) pair. Schema:
  variant_id: int in [0, 108)
  episode_idx: int
  world_seed: int
  return: float
  length: int
  hue: int (deg)
  sat: float
  bright: float
  achievements: dict[str, int]
Optional: worker_idx.
"""

import json
import pathlib
import sys

import numpy as np

_CRAFTER_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_CRAFTER_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_CRAFTER_REPO_ROOT))

from crafter.textures import (
    NUM_VARIANTS, variant_to_hsv, TextureBank)


REQUIRED_KEYS = (
    'variant_id', 'episode_idx', 'world_seed',
    'return', 'length', 'hue', 'sat', 'bright', 'achievements')


def load_sweep(path):
  path = pathlib.Path(path)
  rows = []
  with path.open() as f:
    for i, line in enumerate(f):
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      missing = [k for k in REQUIRED_KEYS if k not in row]
      if missing:
        raise ValueError(f'{path}:{i+1} missing keys {missing}')
      rows.append(row)
  return rows


def pool_of(variant_id):
  if variant_id in TextureBank.TRAIN_POOL:
    return 'train'
  if variant_id in TextureBank.TEST_POOL:
    return 'test'
  raise ValueError(f'variant_id {variant_id} is in neither pool')


def world_id(row):
  """Stable identifier for the underlying world a row was rolled out in.

  With a fixed --world-seed, worker k uses seed `world_seed + k` and crafter
  derives each episode's world from `(seed, episode_idx)`. So the
  (worker_idx, episode_idx) tuple uniquely identifies the world across all
  variants in the same sweep — and across the matched baseline/treatment
  sweeps if both used the same --world-seed and --envs.
  """
  return (row['worker_idx'], row['episode_idx'])


def returns_by_variant(rows):
  out = {v: [] for v in range(NUM_VARIANTS)}
  for row in rows:
    out[row['variant_id']].append(row['return'])
  return {v: np.asarray(xs, dtype=np.float64) for v, xs in out.items() if xs}


def lengths_by_variant(rows):
  out = {v: [] for v in range(NUM_VARIANTS)}
  for row in rows:
    out[row['variant_id']].append(row['length'])
  return {v: np.asarray(xs, dtype=np.float64) for v, xs in out.items() if xs}


def bootstrap_ci(values, n=2000, conf=0.95, rng=None):
  values = np.asarray(values, dtype=np.float64)
  if len(values) == 0:
    return (float('nan'), float('nan'))
  rng = rng or np.random.default_rng(0)
  idx = rng.integers(0, len(values), size=(n, len(values)))
  means = values[idx].mean(axis=1)
  lo = float(np.quantile(means, (1 - conf) / 2))
  hi = float(np.quantile(means, 1 - (1 - conf) / 2))
  return lo, hi


def heatmap_matrix(rows, reducer=np.mean, value_key='return'):
  """Return (hues, sats, grid) where grid[hue_idx, sat_idx] is reduced over bright.

  `value_key` selects which row field to aggregate (e.g. 'return' or 'length').
  """
  from crafter.textures import _HUE_SHIFTS, _SAT_MULTS, _BRIGHT_MULTS
  n_h, n_s, n_b = len(_HUE_SHIFTS), len(_SAT_MULTS), len(_BRIGHT_MULTS)
  accum = {}
  for row in rows:
    hue_idx, rest = divmod(row['variant_id'], n_s * n_b)
    sat_idx, _ = divmod(rest, n_b)
    accum.setdefault((hue_idx, sat_idx), []).append(row[value_key])
  grid = np.full((n_h, n_s), np.nan)
  for (hi, si), xs in accum.items():
    grid[hi, si] = float(reducer(xs))
  return list(_HUE_SHIFTS), list(_SAT_MULTS), grid


def achievement_names(rows):
  names = set()
  for row in rows:
    names.update(row['achievements'].keys())
  return sorted(names)


def achievement_success_rate(rows, name):
  hits = [1 if row['achievements'].get(name, 0) > 0 else 0 for row in rows]
  return float(np.mean(hits)) if hits else float('nan')


def crafter_score(rows):
  """Geometric mean of achievement success rates, Crafter-paper convention."""
  names = achievement_names(rows)
  rates = np.array([achievement_success_rate(rows, n) for n in names])
  # Geometric mean with +1% smoothing so zero rates don't collapse the score.
  return float(np.exp(np.mean(np.log(1 + 100 * rates))) - 1)


def denorm(variant_id):
  h, s, b = variant_to_hsv(variant_id)
  return {'hue': int(h), 'sat': float(s), 'bright': float(b)}


def paired_diff_ci(
    base_rows, treat_rows, pool, metric='return', n_boot=2000, rng=None):
  """Bootstrap 95% CI on the mean per-world paired difference (treat - base).

  For each world_id present in both conditions and in the requested pool,
  averages the metric across that world's variants under each condition,
  takes the diff, then bootstrap-resamples *worlds* (not episodes).

  Returns (mean_diff, ci_lo, ci_hi, n_worlds).
  """
  rng = rng or np.random.default_rng(0)

  def per_world_means(rows):
    buckets = {}
    for r in rows:
      if pool_of(r['variant_id']) != pool:
        continue
      buckets.setdefault(world_id(r), []).append(r[metric])
    return {w: float(np.mean(xs)) for w, xs in buckets.items()}

  base = per_world_means(base_rows)
  treat = per_world_means(treat_rows)
  common = sorted(set(base) & set(treat))
  if not common:
    return float('nan'), float('nan'), float('nan'), 0
  diffs = np.array([treat[w] - base[w] for w in common], dtype=np.float64)
  mean_diff = float(diffs.mean())
  idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
  resampled = diffs[idx].mean(axis=1)
  lo = float(np.quantile(resampled, 0.025))
  hi = float(np.quantile(resampled, 0.975))
  return mean_diff, lo, hi, len(common)


def resolve_output(raw, default_name):
  """Resolve a user-supplied --output into a concrete file path.

  - Trailing '/' (directory-style): returns <raw>/<default_name>.
  - Has a file suffix (e.g. 'foo.pdf'): returns <raw> as-is.
  - Otherwise (bare prefix like '/tmp/first_test'):
      returns <parent>/<name>_<default_name>.

  Also creates the parent directory (mkdir -p). Returns a pathlib.Path.
  """
  raw_str = str(raw)
  if raw_str.endswith('/'):
    resolved = pathlib.Path(raw_str) / default_name
  else:
    p = pathlib.Path(raw_str)
    if p.suffix:
      resolved = p
    else:
      resolved = p.parent / f'{p.name}_{default_name}'
  resolved.parent.mkdir(parents=True, exist_ok=True)
  return resolved
