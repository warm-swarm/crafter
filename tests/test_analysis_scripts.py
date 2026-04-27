"""Smoke tests: every analysis script runs on synthetic data and outputs files."""

import csv
import json
import pathlib
import re
import subprocess
import sys

import pytest

from analysis._synthetic_data import generate, write_jsonl


ANALYSIS_DIR = pathlib.Path(__file__).resolve().parent.parent / 'analysis'


@pytest.fixture(scope='module')
def synthetic_sweeps(tmp_path_factory):
  tmp = tmp_path_factory.mktemp('sweeps')
  baseline = write_jsonl(
      generate('baseline', n_episodes=20, rng_seed=1), tmp / 'baseline.jsonl')
  treatment = write_jsonl(
      generate('treatment', n_episodes=20, rng_seed=2),
      tmp / 'treatment.jsonl')
  return tmp, str(baseline), str(treatment)


def _run(script_name, *args):
  cmd = [sys.executable, str(ANALYSIS_DIR / script_name), *args]
  env_path = f"{ANALYSIS_DIR.parent}:{ANALYSIS_DIR}"
  result = subprocess.run(
      cmd, capture_output=True, text=True,
      env={**__import__('os').environ, 'PYTHONPATH': env_path})
  if result.returncode != 0:
    raise AssertionError(
        f'{script_name} failed:\nstdout={result.stdout}\nstderr={result.stderr}')
  return result


def test_plot_heatmap_produces_pdf(synthetic_sweeps):
  tmp, baseline, _ = synthetic_sweeps
  out = tmp / 'heatmap.pdf'
  _run('plot_heatmap.py', '--input', baseline, '--output', str(out))
  assert out.exists() and out.stat().st_size > 1000


def test_compute_pool_aggregates_csv_shape(synthetic_sweeps):
  tmp, baseline, treatment = synthetic_sweeps
  out = tmp / 'agg.csv'
  _run('compute_pool_aggregates.py',
       '--baseline', baseline, '--treatment', treatment,
       '--output', str(out))
  rows = list(csv.DictReader(out.open()))
  # 2 conditions x 3 pools (variant_0, train, test) + 2 paired_diff rows.
  assert len(rows) == 8
  for r in rows:
    assert set(r) >= {'condition', 'pool', 'n_episodes',
                      'mean_return', 'ci_lo', 'ci_hi'}
    assert r['condition'] in ('baseline', 'treatment', 'paired_diff')
    assert r['pool'] in ('variant_0', 'train', 'test')
  paired = [r for r in rows if r['condition'] == 'paired_diff']
  assert {r['pool'] for r in paired} == {'train', 'test'}


def test_compute_pool_aggregates_no_treatment(synthetic_sweeps):
  tmp, baseline, _ = synthetic_sweeps
  out = tmp / 'agg_baseline_only.csv'
  _run('compute_pool_aggregates.py', '--baseline', baseline, '--output', str(out))
  rows = list(csv.DictReader(out.open()))
  # 1 condition x 3 pools, no paired_diff rows.
  assert len(rows) == 3
  assert {r['pool'] for r in rows} == {'variant_0', 'train', 'test'}
  assert all(r['condition'] == 'baseline' for r in rows)


def test_plot_b_vs_t_produces_pdf(synthetic_sweeps):
  tmp, baseline, treatment = synthetic_sweeps
  out = tmp / 'bvt.pdf'
  _run('plot_b_vs_t.py',
       '--baseline', baseline, '--treatment', treatment,
       '--output', str(out))
  assert out.exists() and out.stat().st_size > 1000


def test_achievement_breakdown_csv_shape(synthetic_sweeps):
  tmp, baseline, treatment = synthetic_sweeps
  out = tmp / 'ach.csv'
  _run('achievement_breakdown.py',
       '--baseline', baseline, '--treatment', treatment,
       '--output', str(out))
  rows = list(csv.DictReader(out.open()))
  # 22 achievements x 2 conditions x 2 pools + 2 conditions x 2 pools
  # (_crafter_score rows) = 88 + 4 = 92.
  assert len(rows) == 92
  for r in rows:
    assert set(r) == {'achievement', 'condition', 'pool',
                      'n_episodes', 'success_rate'}
    # Per-achievement rows: rate in [0,1]. Crafter score is a geometric
    # mean of (1 + 100*rate)-1 and can exceed 1 (typical 0..100).
    if r['achievement'] == '_crafter_score':
      assert 0.0 <= float(r['success_rate']) <= 100.0
    else:
      assert 0.0 <= float(r['success_rate']) <= 1.0
  scores = [r for r in rows if r['achievement'] == '_crafter_score']
  assert len(scores) == 4
  assert {(r['condition'], r['pool']) for r in scores} == {
      ('baseline', 'train'), ('baseline', 'test'),
      ('treatment', 'train'), ('treatment', 'test')}


def test_paired_test_writes_statistic(synthetic_sweeps):
  tmp, baseline, treatment = synthetic_sweeps
  out = tmp / 'paired.txt'
  _run('paired_test.py',
       '--baseline', baseline, '--treatment', treatment,
       '--output', str(out))
  text = out.read_text()
  assert 'Wilcoxon' in text
  assert 'W statistic' in text
  assert 'p-value' in text


def test_paired_diff_recovers_constant_offset(tmp_path):
  """If treatment = baseline + 1.0 per row, paired_test must recover a
  mean diff of ~1.0 with p < 0.001. Catches regressions in either world
  keying (Fix 2) or paired_diff_ci (Fix 3)."""
  base_rows = generate('baseline', n_episodes=20, rng_seed=42)
  treat_rows = []
  for r in base_rows:
    new = dict(r)
    new['return'] = round(r['return'] + 1.0, 3)
    treat_rows.append(new)
  base_path = write_jsonl(base_rows, tmp_path / 'base.jsonl')
  treat_path = write_jsonl(treat_rows, tmp_path / 'treat.jsonl')
  out = tmp_path / 'paired.txt'
  _run('paired_test.py',
       '--baseline', str(base_path), '--treatment', str(treat_path),
       '--pool', 'test', '--output', str(out))
  text = out.read_text()
  diff = float(re.search(r'mean paired diff.*=\s*(-?[\d.]+)', text).group(1))
  pval = float(re.search(r'p-value\s*=\s*([\d.eE+-]+)', text).group(1))
  assert abs(diff - 1.0) < 0.05, f'mean diff was {diff}, want ~1.0'
  assert pval < 0.001, f'p-value was {pval}, want < 0.001'
