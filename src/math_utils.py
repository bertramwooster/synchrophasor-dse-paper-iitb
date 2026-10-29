import numpy as np
import warnings

warnings.filterwarnings("ignore")

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

TWO_PI = 2.0 * np.pi
RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0

@njit(inline="always", fastmath=True)
def wrap_rad(x: float) -> float:
    return (x + np.pi) % TWO_PI - np.pi

@njit(inline="always", fastmath=True)
def wrap_deg(x_deg: float) -> float:
    return (x_deg + 180.0) % 360.0 - 180.0

@njit(fastmath=True)
def jit_argsort_inplace(arr: np.ndarray, order: np.ndarray, n: int) -> None:
    for i in range(n):
        order[i] = i
    for i in range(1, n):
        key_idx = order[i]
        key_val = arr[key_idx]
        j = i - 1
        while j >= 0 and arr[order[j]] > key_val:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key_idx

@njit(fastmath=True)
def fast_1d_sort_cluster_inplace(angles_deg: np.ndarray, eps_deg: float, labels: np.ndarray, order: np.ndarray) -> int:
    n = len(angles_deg)
    jit_argsort_inplace(angles_deg, order, n)
    cluster_id = 0
    labels[order[0]] = 0
    for i in range(1, n):
        if angles_deg[order[i]] - angles_deg[order[i - 1]] > eps_deg:
            cluster_id += 1
        labels[order[i]] = cluster_id
    
    n_clusters = cluster_id + 1
    
    # Theorem 1: S^1 Circular wrap-around check across the +/- 180 degree seam
    wrap_gap = (angles_deg[order[0]] + 360.0) - angles_deg[order[n - 1]]
    if n_clusters > 1 and wrap_gap <= eps_deg:
        last_c = cluster_id
        for i in range(n):
            if labels[i] == last_c:
                labels[i] = 0
            elif labels[i] == 0:
                # Unwrap the lower branch by +360 so cluster 0 is contiguous for median
                angles_deg[i] += 360.0
        n_clusters -= 1
        
    return n_clusters

@njit(fastmath=True)
def weighted_median_1d_subset(values: np.ndarray, weights: np.ndarray, order: np.ndarray, count: int) -> float:
    jit_argsort_inplace(values, order, count)
    total_w = 0.0
    for i in range(count):
        total_w += weights[i]
    half_w = 0.5 * total_w
    cum_w = 0.0
    for i in range(count):
        idx = order[i]
        cum_w += weights[idx]
        if cum_w >= half_w:
            return values[idx]
    return values[order[count - 1]]

@njit(fastmath=True)
def jit_median_1d(arr: np.ndarray, ws_copy: np.ndarray, n: int) -> float:
    for i in range(n):
        ws_copy[i] = arr[i]
    for i in range(1, n):
        key_val = ws_copy[i]
        j = i - 1
        while j >= 0 and ws_copy[j] > key_val:
            ws_copy[j + 1] = ws_copy[j]
            j -= 1
        ws_copy[j + 1] = key_val
    half = n // 2
    if n % 2 == 1:
        return ws_copy[half]
    return 0.5 * (ws_copy[half - 1] + ws_copy[half])
