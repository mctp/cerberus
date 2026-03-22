import numpy as np
import pytest
import torch

from cerberus.transform import Bin


def naive_numpy_bin(arr: np.ndarray, bin_size: int, method: str) -> np.ndarray:
    """
    Naive implementation of binning using numpy reshaping.
    arr: (C, L)
    """
    C, L = arr.shape
    # If L is not divisible by bin_size, we should truncate to match torch's pool default behavior
    new_L = L // bin_size
    truncated_L = new_L * bin_size
    
    if truncated_L == 0:
        return np.zeros((C, 0), dtype=arr.dtype)

    # Truncate
    arr_trunc = arr[:, :truncated_L]
    
    # Reshape (C, new_L, bin_size)
    reshaped = arr_trunc.reshape(C, new_L, bin_size)
    
    if method == "max":
        return reshaped.max(axis=2)
    elif method == "avg":
        return reshaped.mean(axis=2)
    elif method == "sum":
        # Torch implementation uses avg_pool1d * bin_size
        return reshaped.sum(axis=2)
    else:
        raise ValueError(f"Unknown method: {method}")

@pytest.mark.parametrize("method", ["max", "avg", "sum"])
@pytest.mark.parametrize("length, bin_size", [
    (100, 2),
    (100, 10),
    (100, 50),
    (100, 99),
    (100, 100),
    (1024, 2),
    (1024, 128),
    (1024, 1023),
    (1024, 1024),
    (1025, 128), # Test truncation with larger length
])
@pytest.mark.parametrize("channels", [1, 4])
def test_binning_equivalence(method, length, bin_size, channels):
    # Create random data
    data_np = np.random.rand(channels, length).astype(np.float32)
    data_torch = torch.from_numpy(data_np)
    
    # Torch binning
    # We apply to inputs to test the _bin method directly or just call _bin
    bin_transform = Bin(bin_size=bin_size, method=method, apply_to="inputs")
    
    # We can just call _bin directly as it's the core logic
    res_torch = bin_transform._bin(data_torch)
    res_torch_np = res_torch.numpy()
    
    # Numpy binning
    res_numpy = naive_numpy_bin(data_np, bin_size, method)
    
    # Compare
    # Tolerances might be needed for float operations, especially sum/avg
    np.testing.assert_allclose(res_torch_np, res_numpy, rtol=1e-5, atol=1e-6)
