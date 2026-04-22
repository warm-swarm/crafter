"""Evaluation harness for sweeping a trained policy across texture variants."""

from .env import Env


def evaluate_across_variants(
    policy_fn,
    variant_ids,
    n_episodes_per_variant,
    world_seed,
    max_steps=None,
):
  """Run ``policy_fn`` across a set of texture variants with a fixed world seed.

  The world seed is identical for every variant so that episode returns are
  directly comparable: only texture appearance changes between runs.

  Args:
    policy_fn: callable mapping observation -> action.
    variant_ids: iterable of int variant IDs (each in ``[0, 108)``).
    n_episodes_per_variant: number of episodes to run per variant.
    world_seed: integer seed for the world RNG, held fixed across variants.
    max_steps: optional cap on steps per episode (None = run to natural end).

  Returns:
    Dict mapping variant_id (int) -> list[float] of per-episode returns.
  """
  results = {}
  for variant_id in variant_ids:
    variant_id = int(variant_id)
    env = Env(seed=world_seed, texture_variant=variant_id)
    returns = []
    for _ in range(n_episodes_per_variant):
      obs = env.reset()
      total = 0.0
      steps = 0
      done = False
      while not done:
        action = policy_fn(obs)
        obs, reward, done, _ = env.step(action)
        total += float(reward)
        steps += 1
        if max_steps is not None and steps >= max_steps:
          break
      returns.append(total)
    results[variant_id] = returns
  return results
