"""Acceptance tests for the Crafter texture randomisation system."""

import pathlib
import shutil
import sys
import tempfile

import numpy as np
import pytest

# Make `dreamerv3.main` importable for the per-worker texture-seed test
# (the dreamerv3/ package is a sibling of crafter/ under the repo root,
# not on the default import path when pytest is invoked from crafter/).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import crafter
from crafter import constants
from crafter.textures import TextureBank, NUM_VARIANTS


WORLD_SEED = 42
TEXTURE_SEED = 123

# Small but non-degenerate world: keeps worldgen/rollout fast while still
# exercising every code path the default (64, 64) world would. The texture
# pipeline is agnostic to world size, so shrinking the area only affects how
# expensive ``env.reset()`` is.
SMALL_AREA = (16, 16)
TINY_AREA = (4, 4)


def _roll_episode(env, n_steps=25, action_seed=0):
  """Run a fixed-length episode, recording obs/reward/done/state per step."""
  rng = np.random.RandomState(action_seed)
  n_actions = env.action_space.n
  obs = env.reset()
  frames = [obs.copy()]
  rewards, dones, inventories, positions, facings = [], [], [], [], []
  for _ in range(n_steps):
    action = int(rng.randint(0, n_actions))
    obs, reward, done, info = env.step(action)
    frames.append(obs.copy())
    rewards.append(float(reward))
    dones.append(bool(done))
    inventories.append(dict(info['inventory']))
    positions.append(tuple(info['player_pos']))
    facings.append(tuple(env._player.facing))
    if done:
      break
  return {
      'frames': frames,
      'rewards': rewards,
      'dones': dones,
      'inventories': inventories,
      'positions': positions,
      'facings': facings,
  }


def test_variant_zero_is_default():
  """Variant 0 must produce byte-identical frames to unmodified Crafter."""
  legacy_env = crafter.Env(seed=WORLD_SEED, area=SMALL_AREA)
  variant0_env = crafter.Env(seed=WORLD_SEED, area=SMALL_AREA, texture_variant=0)
  a = _roll_episode(legacy_env)
  b = _roll_episode(variant0_env)
  assert len(a['frames']) == len(b['frames'])
  for i, (fa, fb) in enumerate(zip(a['frames'], b['frames'])):
    assert np.array_equal(fa, fb), (
        f'frame {i} differs between legacy and variant 0')


def test_determinism():
  """Two envs with the same variant produce bit-identical frames."""
  env1 = crafter.Env(seed=WORLD_SEED, area=SMALL_AREA, texture_variant=37)
  env2 = crafter.Env(seed=WORLD_SEED, area=SMALL_AREA, texture_variant=37)
  a = _roll_episode(env1)
  b = _roll_episode(env2)
  for i, (fa, fb) in enumerate(zip(a['frames'], b['frames'])):
    assert np.array_equal(fa, fb), (
        f'frame {i} differs across identical-variant envs')


def test_texture_only_perturbation():
  """Perturbation alters pixels but not dynamics, rewards, or game state."""
  env0 = crafter.Env(seed=WORLD_SEED, area=SMALL_AREA, texture_variant=0)
  env_v = crafter.Env(seed=WORLD_SEED, area=SMALL_AREA, texture_variant=73)
  a = _roll_episode(env0)
  b = _roll_episode(env_v)
  assert a['rewards'] == b['rewards'], 'rewards must be identical'
  assert a['dones'] == b['dones'], 'done signals must be identical'
  assert a['inventories'] == b['inventories'], 'inventories must be identical'
  assert a['positions'] == b['positions'], 'player positions must be identical'
  assert a['facings'] == b['facings'], 'player facings must be identical'
  any_diff = any(
      not np.array_equal(fa, fb)
      for fa, fb in zip(a['frames'], b['frames']))
  assert any_diff, 'expected pixel differences between distinct variants'


def _initial_inventory_ui_region(size=(64, 64), view=(9, 9)):
  """Row range of the rendered frame occupied by the initial inventory strip.

  The default initial inventory has only UI-category items nonzero
  (health / food / drink / energy); every other slot has amount 0 and is not
  drawn. The inventory strip is therefore composed exclusively of UI-sprite
  pixels on a black background, so an equality check over the whole strip is
  a valid test of the "UI pixels untouched" property.
  """
  view = np.array(view)
  size = np.array(size)
  unit = size // view
  item_rows = int(np.ceil(len(constants.items) / view[0]))
  border = (size - (size // view) * view) // 2
  local_rows = (view[1] - item_rows) * unit[1]
  row_start = int(border[1] + local_rows)
  row_end = int(row_start + item_rows * unit[1])
  return row_start, row_end


def test_ui_pixels_untouched():
  """UI-sprite pixels must be bit-identical across every variant."""
  env = crafter.Env(seed=WORLD_SEED, area=SMALL_AREA, texture_variant=0)
  env.reset()
  r0, r1 = _initial_inventory_ui_region()
  reference = env.render()[r0:r1].copy()
  for v in range(1, NUM_VARIANTS):
    env._textures.set_variant(v)
    strip = env.render()[r0:r1]
    assert np.array_equal(strip, reference), (
        f'UI strip differs at variant {v}')


def test_pool_partition():
  train = set(TextureBank.TRAIN_POOL)
  test = set(TextureBank.TEST_POOL)
  assert train, 'TRAIN_POOL must be non-empty'
  assert test, 'TEST_POOL must be non-empty'
  assert train.isdisjoint(test), 'pools must be disjoint'
  assert train | test == set(range(NUM_VARIANTS)), (
      'pools must cover 0..107 exactly')
  assert 0 in train, 'variant 0 must belong to TRAIN_POOL'


def test_pool_sampling():
  """1000 resets with train_pool must cover >=90% of TRAIN_POOL."""
  env = crafter.Env(
      seed=WORLD_SEED,
      area=TINY_AREA,
      texture_variant='train_pool',
      texture_seed=TEXTURE_SEED,
  )
  # Pre-warm the bank for every pool variant so the loop below only pays the
  # ~worldgen cost per reset, not the one-time HSV-pipeline cost.
  for v in TextureBank.TRAIN_POOL:
    env._textures.set_variant(v)
  seen = set()
  for _ in range(1000):
    env.reset()
    seen.add(env._textures.variant_id)
  pool = set(TextureBank.TRAIN_POOL)
  assert seen.issubset(pool), 'sampled variants must all be in TRAIN_POOL'
  coverage = len(seen) / len(pool)
  assert coverage >= 0.9, (
      f'expected at least 90% coverage of TRAIN_POOL, got {coverage:.2%}')


def test_current_variant_id_reflects_state():
  """Env.current_variant_id: None in legacy, fixed for int, resamples for pool."""
  legacy = crafter.Env(seed=WORLD_SEED, area=TINY_AREA)
  assert legacy.current_variant_id is None
  fixed = crafter.Env(seed=WORLD_SEED, area=TINY_AREA, texture_variant=42)
  assert fixed.current_variant_id == 42
  fixed.reset()
  assert fixed.current_variant_id == 42
  pool = crafter.Env(
      seed=WORLD_SEED, area=TINY_AREA,
      texture_variant='train_pool', texture_seed=TEXTURE_SEED)
  seen = set()
  for _ in range(20):
    pool.reset()
    v = pool.current_variant_id
    assert v in set(TextureBank.TRAIN_POOL)
    seen.add(v)
  assert len(seen) > 1, 'expected multiple variants across 20 resets'


def test_unknown_asset_fails_loudly():
  """Unexpected PNG in assets dir causes TextureBank construction to raise."""
  with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    real_assets = constants.root / 'assets'
    for src in real_assets.glob('*.png'):
      shutil.copyfile(src, tmp / src.name)
    bogus = tmp / 'surprise-sprite.png'
    shutil.copyfile(next(real_assets.glob('*.png')), bogus)
    with pytest.raises(ValueError) as exc_info:
      TextureBank(tmp, variant_id=0)
    assert 'surprise-sprite' in str(exc_info.value)


def test_within_episode_variant_stability():
  """current_variant_id must only change on reset(), never during step()."""
  env = crafter.Env(
      seed=WORLD_SEED, area=SMALL_AREA,
      texture_variant='train_pool', texture_seed=TEXTURE_SEED)
  env.reset()
  v0 = env.current_variant_id
  assert v0 in set(TextureBank.TRAIN_POOL)
  rng = np.random.RandomState(0)
  n_actions = env.action_space.n
  for step in range(25):
    env.step(int(rng.randint(0, n_actions)))
    assert env.current_variant_id == v0, (
        f'variant changed at step {step}: {v0} -> {env.current_variant_id}')


def test_per_worker_texture_seed():
  """make_env must give each worker a distinct texture_seed when set > 0,
  so envs > 1 don't all walk the same train_pool variant sequence."""
  import elements
  from dreamerv3.main import make_env
  # No use_seed key: the texture-seed offset must work independently of
  # the dreamer world-seed pathway.
  config = elements.Config(
      task='crafter_reward',
      seed=0,
      logdir='/tmp/_test_make_env',
      env={'crafter': {
          'size': [64, 64],
          'logs': False,
          'texture_variant': 'train_pool',
          'texture_seed': 100,
      }},
  )
  env_a = make_env(config, 0)
  env_b = make_env(config, 1)

  def _raw(env):
    while hasattr(env, 'env'):
      env = env.env
    return env._env  # underlying crafter.Env beneath the Crafter adapter

  raw_a, raw_b = _raw(env_a), _raw(env_b)
  seq_a, seq_b = [], []
  for _ in range(50):
    raw_a.reset()
    raw_b.reset()
    seq_a.append(raw_a.current_variant_id)
    seq_b.append(raw_b.current_variant_id)
  diffs = sum(int(a != b) for a, b in zip(seq_a, seq_b))
  assert diffs >= 40, (
      f'expected >=40/50 differing positions, got {diffs}; '
      f'seq_a={seq_a[:5]}..., seq_b={seq_b[:5]}...')
