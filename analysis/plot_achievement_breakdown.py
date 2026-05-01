"""Per-achievement grouped bar chart, three variants from one script.

For each achievement we draw six bars: clean/train/test under each of
baseline/treatment. Achievements are sorted by treatment-train rate
descending. Three PDF variants are emitted:
  *_filtered.pdf — drop achievements where all six cells are 0
  *_faded.pdf    — keep all 22, fade all-zero rows to alpha=0.3
  *_all.pdf      — show all 22 at full alpha
Pool is encoded by hatch (clean='', train='//', test='xx'); condition by
the existing C0 (baseline) / C1 (treatment) palette.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import sweep_common


_POOLS = ('clean', 'train', 'test')
_HATCH = {'clean': '', 'train': '//', 'test': 'xx'}
_COLOR = {'baseline': 'C0', 'treatment': 'C1'}


def _pool_subset(rows, pool):
  if pool == 'clean':
    return [r for r in rows if r['variant_id'] == 0]
  return [r for r in rows if sweep_common.pool_of(r['variant_id']) == pool]


def _rate_table(loaded, achievements):
  """Returns dict: rates[ach][cond][pool] -> float."""
  rates = {}
  for ach in achievements:
    rates[ach] = {}
    for cond, rows in loaded.items():
      rates[ach][cond] = {}
      for pool in _POOLS:
        sub = _pool_subset(rows, pool)
        rates[ach][cond][pool] = (
            sweep_common.achievement_success_rate(sub, ach) if sub else 0.0)
  return rates


def _draw(ax, achievements, rates, conditions, faded_set=None):
  faded_set = faded_set or set()
  width = 0.13
  offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * width
  bars = []
  for cond_idx, cond in enumerate(conditions):
    for pool_idx, pool in enumerate(_POOLS):
      slot = cond_idx * len(_POOLS) + pool_idx
      offset = offsets[slot]
      values = [rates[a][cond][pool] for a in achievements]
      alphas = [0.3 if a in faded_set else 1.0 for a in achievements]
      xs = np.arange(len(achievements)) + offset
      # Per-bar alpha needs a per-bar call; use a single call with the
      # constant alpha when no fading is in effect.
      if any(a < 1.0 for a in alphas):
        for x, v, a in zip(xs, values, alphas):
          ax.bar(x, v, width=width, color=_COLOR[cond], alpha=a,
                 hatch=_HATCH[pool], edgecolor='black', linewidth=0.3)
        # Sentinel for legend.
        bars.append(ax.bar(
            [], [], width=width, color=_COLOR[cond],
            hatch=_HATCH[pool], edgecolor='black', linewidth=0.3,
            label=f'{cond} / {pool}'))
      else:
        bars.append(ax.bar(
            xs, values, width=width, color=_COLOR[cond],
            hatch=_HATCH[pool], edgecolor='black', linewidth=0.3,
            label=f'{cond} / {pool}'))
  ax.set_xticks(np.arange(len(achievements)))
  ax.set_xticklabels(achievements, rotation=30, ha='right', fontsize=8)
  ax.set_ylabel('Success rate')
  ax.set_xlim(-0.6, len(achievements) - 0.4)
  ax.grid(alpha=0.3, axis='y')
  ax.legend(fontsize=7, ncol=3, loc='upper right')


def _save(path, achievements, rates, conditions, faded_set=None):
  fig, ax = plt.subplots(figsize=(11, 4))
  _draw(ax, achievements, rates, conditions, faded_set=faded_set)
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)
  print(f'wrote {path}')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--baseline', required=True)
  p.add_argument('--treatment', required=True)
  p.add_argument('--output-prefix', required=True,
                 help='Files written: <prefix>_{filtered,faded,all}.pdf')
  args = p.parse_args()

  loaded = {
      'baseline': sweep_common.load_sweep(args.baseline),
      'treatment': sweep_common.load_sweep(args.treatment),
  }
  conditions = ('baseline', 'treatment')
  achievements = sorted(
      set().union(*(sweep_common.achievement_names(r) for r in loaded.values())))
  rates = _rate_table(loaded, achievements)

  achievements.sort(
      key=lambda a: -rates[a]['treatment']['train'])

  all_zero = {
      a for a in achievements
      if all(rates[a][c][p] == 0.0
             for c in conditions for p in _POOLS)}

  prefix = args.output_prefix
  _save(f'{prefix}_filtered.pdf',
        [a for a in achievements if a not in all_zero],
        rates, conditions)
  _save(f'{prefix}_faded.pdf', achievements, rates, conditions,
        faded_set=all_zero)
  _save(f'{prefix}_all.pdf', achievements, rates, conditions)


if __name__ == '__main__':
  main()
