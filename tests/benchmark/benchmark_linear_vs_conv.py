import time

import torch
import torch.nn as nn


def benchmark():
    N = 32
    C = 2048
    L = 512
    Out_C = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")

    x = torch.randn(N, C, L).to(device)

    # Conv1d
    conv = nn.Conv1d(C, Out_C, 1).to(device)

    # Linear + Transpose
    linear = nn.Linear(C, Out_C).to(device)

    # Warmup
    for _ in range(10):
        _ = conv(x)
        _ = linear(x.transpose(1, 2)).transpose(1, 2)

    iterations = 100

    start = time.time()
    for _ in range(iterations):
        conv(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    conv_time = (time.time() - start) / iterations

    start = time.time()
    for _ in range(iterations):
        # Emulate original sequence: Transpose -> Linear -> Transpose
        # Original: x (N, C, L) -> pool -> (N, C, L/2) -> Transpose -> (N, L/2, C) -> Linear -> (N, L/2, Tracks) -> Transpose -> (N, Tracks, L/2)
        # We test just the op
        linear(x.transpose(1, 2)).transpose(1, 2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    linear_time = (time.time() - start) / iterations

    print(f"Conv1d time: {conv_time * 1000:.4f} ms")
    print(f"Linear+Transpose time: {linear_time * 1000:.4f} ms")
    print(f"Speedup: {linear_time / conv_time:.2f}x")


if __name__ == "__main__":
    benchmark()
