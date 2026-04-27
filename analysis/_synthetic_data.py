"""Generate synthetic sweep JSONL with realistic shape.

Used only by the analysis-script tests. Never a substitute for real eval.
"""

import argparse
import json
import pathlib

import numpy as np

from crafter import constants
from crafter.textures import NUM_VARIANTS, variant_to_hsv


ACHIEVEMENTS = list(constants.achievements)


def _variant_mean_return(variant_id, condition):
  """Plausible mean return for a variant under a given condition.

  Baseline degrades on test-pool hues (>=180); treatment recovers most of
  the gap. Both are noisy.
  """
  hue, sat, bright = variant_to_hsv(variant_id)
  far_hue = min(hue, 360 - hue) / 180.0  # 0 for hue=0, 1 for hue=180
  sat_pen = abs(sat - 1.0) * 2
  bright_pen = abs(bright - 1.0) * 1.5
  penalty = far_hue * 4.0 + sat_pen + bright_pen
  if condition == 'baseline':
    return max(0.5, 10.0 - penalty)
  if condition == 'treatment':
    return max(1.0, 10.0 - 0.3 * penalty)
  raise ValueError(condition)


def generate(
    condition='baseline', n_episodes=20, world_seed=12345,
    variant_ids=None, rng_seed=0):
  if variant_ids is None:
    variant_ids = list(range(NUM_VARIANTS))
  rng = np.random.default_rng(rng_seed)
  rows = []
  for variant_id in variant_ids:
    hue, sat, bright = variant_to_hsv(variant_id)
    mean_ret = _variant_mean_return(variant_id, condition)
    for episode_idx in range(n_episodes):
      ret = float(rng.normal(mean_ret, 1.5))
      length = int(rng.integers(200, 1000))
      success_prob = np.clip(mean_ret / 12.0, 0.05, 0.95)
      achievements = {}
      for name in ACHIEVEMENTS:
        # Per-achievement hit probability; modulated by per-achievement
        # difficulty so not every cell is equally easy.
        k = hash((name, 'seed')) % 100 / 100.0
        p = np.clip(success_prob * (0.3 + 0.7 * k), 0, 1)
        achievements[name] = int(rng.binomial(3, p))
      rows.append({
          'variant_id': int(variant_id),
          'episode_idx': int(episode_idx),
          'worker_idx': int(episode_idx % 4),
          'world_seed': int(world_seed),
          'return': round(ret, 3),
          'length': length,
          'hue': int(hue),
          'sat': float(sat),
          'bright': float(bright),
          'achievements': achievements,
      })
  return rows


def write_jsonl(rows, path):
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w') as f:
    for row in rows:
      f.write(json.dumps(row) + '\n')
  return path


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--condition', choices=['baseline', 'treatment'],
                 default='baseline')
  p.add_argument('--output', required=True)
  p.add_argument('--episodes', type=int, default=20)
  p.add_argument('--world-seed', type=int, default=12345)
  p.add_argument('--rng-seed', type=int, default=0)
  args = p.parse_args()
  rows = generate(
      condition=args.condition, n_episodes=args.episodes,
      world_seed=args.world_seed, rng_seed=args.rng_seed)
  write_jsonl(rows, args.output)
  print(f'wrote {len(rows)} rows to {args.output}')


if __name__ == '__main__':
  main()
