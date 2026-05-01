"""Heatmap: hue (x) x saturation (y) x mean return, averaged over brightness."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import sweep_common


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--input', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--title', default='')
  p.add_argument('--metric', default='return', choices=['return', 'length'])
  args = p.parse_args()
  args.output = sweep_common.resolve_output(args.output, 'heatmap.pdf')

  rows = sweep_common.load_sweep(args.input)
  hues, sats, grid = sweep_common.heatmap_matrix(
      rows, reducer=np.mean, value_key=args.metric)

  # _SAT_MULTS is identity-first (1.0, 0.7, 1.3); reorder so the y-axis
  # reads monotonically (0.7, 1.0, 1.3) bottom-to-top.
  order = np.argsort(sats)
  sats = [sats[i] for i in order]
  grid = grid[:, order]

  fig, ax = plt.subplots(figsize=(7, 3))
  # Transpose so y=sat, x=hue.
  display = grid.T
  im = ax.imshow(display, aspect='auto', origin='lower', cmap='viridis')
  # Per-cell numeric overlay. Viridis luminance increases monotonically with
  # value, so flip text white when the *normalised* value is below 0.5.
  vmin = float(np.nanmin(display))
  vmax = float(np.nanmax(display))
  span = vmax - vmin if vmax > vmin else 1.0
  for (yi, xi), val in np.ndenumerate(display):
    if np.isnan(val):
      continue
    color = 'white' if (val - vmin) / span < 0.5 else 'black'
    ax.text(xi, yi, f'{val:.1f}', ha='center', va='center',
            color=color, fontsize=7)
  ax.set_xticks(range(len(hues)))
  ax.set_xticklabels([f'{h}' for h in hues])
  ax.set_yticks(range(len(sats)))
  ax.set_yticklabels([f'{s:.1f}' for s in sats])
  ax.set_xlabel('Hue shift (deg)')
  ax.set_ylabel('Saturation mult')
  # Train/test split at 180 deg; draw a dividing line between index 5 (150)
  # and 6 (180).
  ax.axvline(x=5.5, color='white', linestyle='--', linewidth=1)
  ax.text(2.5, len(sats) - 0.3, 'train pool', color='white',
          ha='center', va='top', fontsize=8)
  ax.text(8.5, len(sats) - 0.3, 'test pool', color='white',
          ha='center', va='top', fontsize=8)
  cbar = fig.colorbar(im, ax=ax)
  cbar.set_label(
      'Mean episode length' if args.metric == 'length' else 'Mean return')
  if args.title:
    ax.set_title(args.title)
  fig.tight_layout()
  fig.savefig(args.output)
  print(f'wrote {args.output}')


if __name__ == '__main__':
  main()
