"""Per-(condition, pool) aggregates: mean metric ± 95% bootstrap CI.

Pools enumerated: variant_0 (single-cell ceiling), train, test. Note
variant_0 is also a member of train; the variant_0 row is reported
separately so a variant_0-vs-test gap is the difference of two CSV rows.

When --treatment is given, two extra rows are appended with
condition='paired_diff' (one per pool ∈ {train, test}); their
mean_return is the per-world mean (treat - base) and ci_lo/ci_hi are
bootstrap-resampled across worlds. n_episodes for these rows is the
number of paired worlds (= |worlds in both runs|), not episodes.

The 'mean_return' column name is preserved even under --metric length,
so downstream consumers reading CSV by column name don't break.
"""

import argparse
import csv

import numpy as np

import sweep_common


def aggregate(rows, pool, metric):
  if pool == 'variant_0':
    subset = [r for r in rows if r['variant_id'] == 0]
  else:
    subset = [r for r in rows if sweep_common.pool_of(r['variant_id']) == pool]
  values = np.asarray([r[metric] for r in subset], dtype=np.float64)
  lo, hi = sweep_common.bootstrap_ci(values)
  return {
      'n_episodes': int(len(values)),
      'mean_return': float(values.mean()) if len(values) else float('nan'),
      'ci_lo': lo,
      'ci_hi': hi,
  }


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--baseline', required=True)
  p.add_argument('--treatment', required=False, default=None)
  p.add_argument('--random', required=False, default=None,
                 help='Optional sweep_results_r.jsonl from a uniform-random '
                      'policy. Adds three condition=random rows.')
  p.add_argument('--metric', default='return', choices=['return', 'length'])
  p.add_argument('--output', required=True)
  args = p.parse_args()
  args.output = sweep_common.resolve_output(args.output, 'pool_aggregates.csv')

  conditions = [('baseline', args.baseline)]
  if args.treatment:
    conditions.append(('treatment', args.treatment))
  if args.random:
    conditions.append(('random', args.random))

  loaded = {cond: sweep_common.load_sweep(path) for cond, path in conditions}

  fields = ['condition', 'pool', 'n_episodes',
            'mean_return', 'ci_lo', 'ci_hi']
  with open(args.output, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for cond, _ in conditions:
      for pool in ('variant_0', 'train', 'test'):
        agg = aggregate(loaded[cond], pool, args.metric)
        w.writerow({'condition': cond, 'pool': pool, **agg})
    if args.treatment:
      base_rows = loaded['baseline']
      treat_rows = loaded['treatment']
      for pool in ('train', 'test'):
        mean_diff, lo, hi, n_worlds = sweep_common.paired_diff_ci(
            base_rows, treat_rows, pool, metric=args.metric)
        w.writerow({
            'condition': 'paired_diff',
            'pool': pool,
            'n_episodes': n_worlds,
            'mean_return': mean_diff,
            'ci_lo': lo,
            'ci_hi': hi,
        })
  print(f'wrote {args.output}')


if __name__ == '__main__':
  main()
