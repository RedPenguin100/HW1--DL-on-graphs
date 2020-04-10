import numpy as np
import pytest

from ts_generator import generate_ar_k, is_stationary_ar_k


def test_generate():
    alphas = np.array([0.5, 0.4])
    generated_ar = generate_ar_k(alphas, 1000, np.random.normal)
    print(generated_ar)


def test_stationary():
    assert is_stationary_ar_k([0.5, 0.4])
    assert not is_stationary_ar_k(np.array([0.5, 0.5]))


def test_stationary_2():
    assert is_stationary_ar_k([3.5, -4.85, +3.325, -1.1274, +0.1512])
