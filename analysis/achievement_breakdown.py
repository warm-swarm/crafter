"""Per-achievement success rate, broken down by (condition, pool)."""

import argparse
import csv

import sweep_common


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--baseline', required=True)
  p.add_argument('--treatment', required=False, default=None)
  p.add_argument('--output', required=True)
  args = p.parse_args()
  args.output = sweep_common.resolve_output(
      args.output, 'achievement_breakdown.csv')

  conditions = [('baseline', args.baseline)]
  if args.treatment:
    conditions.append(('treatment', args.treatment))

  loaded = {cond: sweep_common.load_sweep(path) for cond, path in conditions}
  all_achievements = set()
  for rows in loaded.values():
    all_achievements.update(sweep_common.achievement_names(rows))
  achievements = sorted(all_achievements)

  def pool_subset(rows, pool):
    if pool == 'clean':
      return [r for r in rows if r['variant_id'] == 0]
    return [r for r in rows if sweep_common.pool_of(r['variant_id']) == pool]

  fields = ['achievement', 'condition', 'pool', 'n_episodes', 'success_rate']
  with open(args.output, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for ach in achievements:
      for cond, rows in loaded.items():
        for pool in ('clean', 'train', 'test'):
          pool_rows = pool_subset(rows, pool)
          if not pool_rows:
            continue
          rate = sweep_common.achievement_success_rate(pool_rows, ach)
          w.writerow({
              'achievement': ach,
              'condition': cond,
              'pool': pool,
              'n_episodes': len(pool_rows),
              'success_rate': round(rate, 4),
          })
    # Crafter score: geometric mean of achievement success rates per cell.
    # Leading underscore makes these easy to filter from achievement rows.
    for cond, rows in loaded.items():
      for pool in ('clean', 'train', 'test'):
        pool_rows = pool_subset(rows, pool)
        if not pool_rows:
          continue
        w.writerow({
            'achievement': '_crafter_score',
            'condition': cond,
            'pool': pool,
            'n_episodes': len(pool_rows),
            'success_rate': round(sweep_common.crafter_score(pool_rows), 4),
        })
  print(f'wrote {args.output}')


if __name__ == '__main__':
  main()
