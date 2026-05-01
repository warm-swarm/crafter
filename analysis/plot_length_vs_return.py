"""Length-vs-return scatter, three layouts from one script.

Visualises the "alive but misperceiving" claim: episodes that survive close
to the horizon but earn little reward.

Emits:
  *_overlay.pdf  — single panel; per-episode dots at low alpha, plus
                   per-variant-mean markers at high alpha for readability.
  *_simple.pdf   — single panel, per-episode dots only.
  *_faceted.pdf  — two panels (baseline | treatment), shared axes.

Encoding: colour = condition (C0 baseline, C1 treatment),
          marker  = pool ('o' clean, 's' train, '^' test).
"""

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import sweep_common


_POOL_MARKER = {'clean': 'o', 'train': 's', 'test': '^'}
_COND_COLOR = {'baseline': 'C0', 'treatment': 'C1'}


def _pool_label(variant_id):
  if variant_id == 0:
    return 'clean'
  return sweep_common.pool_of(variant_id)


def _split_by_pool(rows):
  by_pool = defaultdict(list)
  for r in rows:
    by_pool[_pool_label(r['variant_id'])].append(r)
  return by_pool


def _per_variant_means(rows):
  by_variant = defaultdict(lambda: ([], []))
  for r in rows:
    lengths, returns = by_variant[r['variant_id']]
    lengths.append(r['length'])
    returns.append(r['return'])
  out = []
  for vid, (lengths, returns) in by_variant.items():
    out.append((vid, float(np.mean(lengths)), float(np.mean(returns))))
  return out


def _scatter_episodes(ax, rows, cond, alpha):
  by_pool = _split_by_pool(rows)
  for pool, pool_rows in by_pool.items():
    xs = [r['length'] for r in pool_rows]
    ys = [r['return'] for r in pool_rows]
    ax.scatter(xs, ys, marker=_POOL_MARKER[pool], color=_COND_COLOR[cond],
               alpha=alpha, s=10, linewidths=0,
               label=f'{cond} / {pool}')


def _scatter_variant_means(ax, rows, cond):
  for vid, mean_len, mean_ret in _per_variant_means(rows):
    pool = _pool_label(vid)
    ax.scatter([mean_len], [mean_ret], marker=_POOL_MARKER[pool],
               color=_COND_COLOR[cond], alpha=0.9, s=80,
               edgecolors='black', linewidths=0.6)


def _set_axes(ax):
  ax.set_xlabel('Episode length')
  ax.set_ylabel('Episode return')
  ax.grid(alpha=0.3)


def _save_overlay(path, loaded):
  fig, ax = plt.subplots(figsize=(6, 4))
  for cond, rows in loaded.items():
    _scatter_episodes(ax, rows, cond, alpha=0.15)
  for cond, rows in loaded.items():
    _scatter_variant_means(ax, rows, cond)
  _set_axes(ax)
  ax.legend(fontsize=7, loc='lower right', ncol=2)
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)
  print(f'wrote {path}')


def _save_simple(path, loaded):
  fig, ax = plt.subplots(figsize=(6, 4))
  for cond, rows in loaded.items():
    _scatter_episodes(ax, rows, cond, alpha=0.3)
  _set_axes(ax)
  ax.legend(fontsize=7, loc='lower right', ncol=2)
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)
  print(f'wrote {path}')


def _save_faceted(path, loaded):
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, sharex=True)
  for ax, (cond, rows) in zip(axes, loaded.items()):
    _scatter_episodes(ax, rows, cond, alpha=0.3)
    _set_axes(ax)
    ax.set_title(cond)
    ax.legend(fontsize=7, loc='lower right')
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)
  print(f'wrote {path}')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--baseline', required=True)
  p.add_argument('--treatment', required=True)
  p.add_argument('--output-prefix', required=True,
                 help='Files written: <prefix>_{overlay,simple,faceted}.pdf')
  args = p.parse_args()
  loaded = {
      'baseline': sweep_common.load_sweep(args.baseline),
      'treatment': sweep_common.load_sweep(args.treatment),
  }
  prefix = args.output_prefix
  _save_overlay(f'{prefix}_overlay.pdf', loaded)
  _save_simple(f'{prefix}_simple.pdf', loaded)
  _save_faceted(f'{prefix}_faceted.pdf', loaded)


if __name__ == '__main__':
  main()
