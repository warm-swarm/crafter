"""Render a fixed Crafter scene under every texture variant as a grid PNG.

Usage:
    python scripts/render_variant_grid.py [--output grid.png] [--seed 42]
"""

import argparse
import math
import pathlib
import sys

import imageio.v3 as imageio
import numpy as np

# Make the bundled crafter package importable without installation.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import crafter  # noqa: E402
from crafter.textures import NUM_VARIANTS, variant_to_hsv  # noqa: E402


def render_variant(seed, variant_id):
  env = crafter.Env(seed=seed, texture_variant=variant_id)
  return env.reset()


def compose_grid(frames, cols):
  tile_h, tile_w = frames[0].shape[:2]
  rows = math.ceil(len(frames) / cols)
  grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
  for i, frame in enumerate(frames):
    r, c = divmod(i, cols)
    grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = frame
  return grid


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='variant_grid.png')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--cols', type=int, default=12,
                      help='Number of columns in the grid (default: 12 = hues).')
  args = parser.parse_args()

  frames = []
  for v in range(NUM_VARIANTS):
    hue, sat, bright = variant_to_hsv(v)
    print(f'[{v:3d}/{NUM_VARIANTS}] hue={hue:3d}° sat={sat} bright={bright}')
    frames.append(render_variant(args.seed, v))

  grid = compose_grid(frames, cols=args.cols)
  imageio.imwrite(args.output, grid)
  print(f'Wrote {grid.shape[1]}x{grid.shape[0]} grid to {args.output}')


if __name__ == '__main__':
  main()
