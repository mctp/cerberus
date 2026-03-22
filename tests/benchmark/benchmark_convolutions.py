import time

import torch
import torch.nn as nn


def benchmark_op(layer, input_data, name, iterations=50):
    # Warmup
    print(f"[{name}] Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = layer(input_data)

    # Timing
    print(f"[{name}] Benchmarking {iterations} iterations...")
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            layer(input_data)
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    print(f"[{name}] Average Time: {avg_time:.4f}s")
    return avg_time


def run_test():
    # Force CPU
    device = torch.device("cpu")
    print(f"Running on device: {device}")

    # BPNet-like Parameters
    BATCH_SIZE = 64
    CHANNELS = 64
    LENGTH = 2000  # Typical BPNet input window
    KERNEL_SIZE = 21  # Your conv_kernel_size
    DILATION = 8  # A middle-layer dilation

    # ---------------------------------------------------------
    # TEST 1: Standard Conv1d (The Slow Way)
    # ---------------------------------------------------------
    conv1d = nn.Conv1d(
        CHANNELS, CHANNELS, kernel_size=KERNEL_SIZE, dilation=DILATION, padding="valid"
    ).to(device)

    # Input: (Batch, Channels, Length)
    input_1d = torch.randn(BATCH_SIZE, CHANNELS, LENGTH).to(device)

    time_1d = benchmark_op(conv1d, input_1d, "Conv1d")

    # ---------------------------------------------------------
    # TEST 2: Conv2d Hack (The Optimized Way)
    # ---------------------------------------------------------
    conv2d = nn.Conv2d(
        CHANNELS,
        CHANNELS,
        kernel_size=(1, KERNEL_SIZE),
        dilation=(1, DILATION),
        padding="valid",
    ).to(device)

    # Input: (Batch, Channels, Height=1, Length)
    input_2d = input_1d.unsqueeze(2)

    time_2d = benchmark_op(conv2d, input_2d, "Conv2d")

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    print("-" * 30)
    speedup = time_1d / time_2d
    print(f"Speedup Factor: {speedup:.2f}x")

    if speedup > 5.0:
        print("\nCONCLUSION: CONFIRMED.")
        print("Your CPU lacks optimized kernels for Dilated Conv1d.")
        print("The Conv2d hack is required.")
    else:
        print("\nCONCLUSION: INCONCLUSIVE.")
        print("Both operations ran at similar speeds.")


if __name__ == "__main__":
    run_test()
