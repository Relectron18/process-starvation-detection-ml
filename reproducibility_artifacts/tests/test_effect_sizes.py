import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multiseed_stability import cohens_d_paired, cohens_d_one_sample


def test_cohens_d_paired_nan_when_differences_have_zero_variance():
    a = np.array([0.8, 0.81, 0.79, 0.8, 0.8])
    assert np.isnan(cohens_d_paired(a, a))


def test_cohens_d_paired_zero_when_mean_diff_is_zero_but_variance_isnt():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.5, 2.5, 2.5, 2.5])  # same mean as a (2.5), but a varies
    assert cohens_d_paired(a, b) == 0.0


def test_cohens_d_paired_matches_hand_calculation():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 1.0, 1.5, 2.0])
    diff = a - b
    expected = diff.mean() / diff.std(ddof=1)
    assert abs(cohens_d_paired(a, b) - expected) < 1e-9


def test_cohens_d_paired_sign_flips_with_argument_order():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 1.0, 1.5])
    assert cohens_d_paired(a, b) > 0
    assert cohens_d_paired(b, a) < 0


def test_cohens_d_one_sample_matches_hand_calculation():
    x = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    expected = x.mean() / x.std(ddof=1)
    assert abs(cohens_d_one_sample(x) - expected) < 1e-9


def test_cohens_d_handles_zero_variance_without_crashing():
    x = np.array([0.5, 0.5, 0.5])
    result = cohens_d_one_sample(x)
    assert np.isnan(result)
