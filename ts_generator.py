import numpy as np


def get_last_n(arr: list, n: int):
    return arr[-n:]


def is_stationary_ar_k(alphas: np.array):
    """
    Function receives the alpha values that define the AR(k) time series,
    and return true iff they represent a stationary series.
    """
    r_alphas = np.array(list(reversed(alphas)))
    polynomial = np.append(r_alphas, -1)
    roots = np.roots(polynomial)
    for root in roots:
        if -1 <= root <= 1:
            return False
    return True


# noinspection PyPep8Naming
def generate_ar_k(alphas: np.array, N: int, noise_func) -> list:
    """
    AR-k formula:
    X_n = a_1 * x_n-1 + ... + a_k * x_n-k + noise_n
    :note: k is determined by the alpha list size.
    """
    k = len(alphas)
    if N < k:
        raise ValueError('N is too short!')
    # TODO: Decide initial condition. Random ? zeros ? ones?
    init_condition = np.random.uniform(-1000, 1000, k)
    ar_k_series = list(init_condition)
    while len(ar_k_series) < N:
        # The first alpha is multiplied by the last x, and so on, so we reverse the Xs
        new_value = alphas.dot(list(reversed(get_last_n(ar_k_series, k)))) + noise_func()
        ar_k_series.append(new_value)

    return ar_k_series
