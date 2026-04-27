"""Paired Wilcoxon signed-rank test across worlds on test-pool returns.

For each episode_idx (= world), compute each condition's mean return across
the test-pool variants, then run a signed-rank test on the N paired
differences (one per world).
"""

import argparse

import numpy as np
from scipy import stats

import sweep_common


def mean_per_world(rows, pool='test'):
  """Per-world mean return; world keyed by (worker_idx, episode_idx)."""
  buckets = {}
  for r in rows:
    if sweep_common.pool_of(r['variant_id']) != pool:
      continue
    buckets.setdefault(sweep_common.world_id(r), []).append(r['return'])
  worlds = sorted(buckets)
  means = {w: float(np.mean(buckets[w])) for w in worlds}
  return worlds, means


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--baseline', required=True)
  p.add_argument('--treatment', required=True)
  p.add_argument('--pool', default='test', choices=['train', 'test'])
  p.add_argument('--output', required=True)
  args = p.parse_args()
  args.output = sweep_common.resolve_output(
      args.output, f'paired_test_{args.pool}.txt')

  base_rows = sweep_common.load_sweep(args.baseline)
  treat_rows = sweep_common.load_sweep(args.treatment)

  base_worlds, base_means = mean_per_world(base_rows, args.pool)
  treat_worlds, treat_means = mean_per_world(treat_rows, args.pool)
  common = sorted(set(base_worlds) & set(treat_worlds))
  if len(common) != len(base_worlds) or len(common) != len(treat_worlds):
    print(f'WARNING: mismatched worlds (base={len(base_worlds)}, '
          f'treat={len(treat_worlds)}), using intersection ({len(common)})')
  base_aligned = np.array([base_means[w] for w in common])
  treat_aligned = np.array([treat_means[w] for w in common])
  diffs = treat_aligned - base_aligned

  if np.all(diffs == 0):
    stat, pval = float('nan'), 1.0
    note = 'All paired differences are zero; Wilcoxon skipped.'
  else:
    result = stats.wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
    stat = float(result.statistic)
    pval = float(result.pvalue)
    note = ''

  with open(args.output, 'w') as f:
    f.write(f'Paired Wilcoxon signed-rank test ({args.pool} pool)\n')
    f.write(f'n_worlds = {len(common)}\n')
    f.write(f'baseline mean = {base_aligned.mean():.4f}\n')
    f.write(f'treatment mean = {treat_aligned.mean():.4f}\n')
    f.write(f'mean paired diff (T - B) = {diffs.mean():.4f}\n')
    f.write(f'median paired diff = {np.median(diffs):.4f}\n')
    f.write(f'W statistic = {stat}\n')
    f.write(f'p-value = {pval}\n')
    if note:
      f.write(note + '\n')
  print(f'wrote {args.output}')


if __name__ == '__main__':
  main()
