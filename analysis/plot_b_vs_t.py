"""Baseline vs treatment line plot: x = pool, y = mean return, with 95% CI."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import sweep_common


def pool_stats(rows, pool, metric):
  if pool == 'clean':
    subset = [r for r in rows if r['variant_id'] == 0]
  else:
    subset = [r for r in rows if sweep_common.pool_of(r['variant_id']) == pool]
  values = np.asarray([r[metric] for r in subset], dtype=np.float64)
  lo, hi = sweep_common.bootstrap_ci(values)
  return float(values.mean()) if len(values) else float('nan'), lo, hi


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--baseline', required=True)
  p.add_argument('--treatment', required=True)
  p.add_argument('--random', default=None,
                 help='Optional sweep_results_r.jsonl. Draws a horizontal '
                      'dashed line at the random-policy mean as a floor.')
  p.add_argument('--metric', default='return', choices=['return', 'length'])
  p.add_argument('--output', required=True)
  args = p.parse_args()
  args.output = sweep_common.resolve_output(args.output, 'b_vs_t.pdf')

  baseline = sweep_common.load_sweep(args.baseline)
  treatment = sweep_common.load_sweep(args.treatment)
  random_rows = sweep_common.load_sweep(args.random) if args.random else None

  pools = ['clean', 'train', 'test']
  xs = np.arange(len(pools))
  fig, ax = plt.subplots(figsize=(5, 3.5))
  for label, rows, color in (
      ('baseline', baseline, 'C0'),
      ('treatment', treatment, 'C1')):
    means, los, his = [], [], []
    for pool in pools:
      m, lo, hi = pool_stats(rows, pool, args.metric)
      means.append(m)
      los.append(lo)
      his.append(hi)
    means = np.array(means)
    los = np.array(los)
    his = np.array(his)
    ax.plot(xs, means, marker='o', color=color, label=label)
    ax.fill_between(xs, los, his, alpha=0.2, color=color)

  if random_rows is not None:
    test_mean = pool_stats(random_rows, 'test', args.metric)[0]
    train_mean = pool_stats(random_rows, 'train', args.metric)[0]
    if abs(train_mean - test_mean) > 0.5:
      y, suffix = train_mean, ' (train-pool mean)'
    else:
      y, suffix = test_mean, ''
    ax.axhline(y, linestyle='--', color='gray', alpha=0.6,
               label=f'random policy{suffix}')

  ax.set_xticks(xs)
  ax.set_xticklabels(pools)
  ax.set_xlabel('Condition')
  ax.set_ylabel(
      'Mean episode length' if args.metric == 'length'
      else 'Mean episode return')
  ax.legend()
  ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(args.output)
  print(f'wrote {args.output}')


if __name__ == '__main__':
  main()
