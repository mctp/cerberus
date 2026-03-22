import os
import sys

import torch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from cerberus.models.asap import ConvNeXtDCNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # User requested: 2048bp input and 1024bp output.
    # To get 1024bp output from 2048bp input, we need a total stride of 2.
    # The model has a fixed init_pool with stride 2.
    # So the core block must have stride 1.
    # Core block stride is determined by bin_size // 2.
    # For stride 1, bin_size must be 2.
    # Note: "1bp resolution" in the prompt might refer to the fact that we want high resolution,
    # but strictly speaking 2048->1024 is 2bp resolution relative to input.
    # Or it implies the user wants us to try to force 1bp resolution if possible?
    # But current code `bin_size // 2` prevents `bin_size=1`.

    bin_size = 2
    input_len = 2048
    output_len = 1024

    try:
        model = ConvNeXtDCNN(
            input_len=input_len, output_len=output_len, output_bin_size=bin_size
        )

        print(f"Model instantiated with bin_size={bin_size}")

        # Test forward pass to verify shapes
        dummy_input = torch.randn(1, 4, input_len)
        try:
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            # Output is ProfileLogRates, need to check its log_rates or similar
            # Based on source, it returns ProfileLogRates(log_rates=x)
            # x shape should be (Batch, Tracks, Length)
            out_tensor = output.log_rates
            print(f"Output shape: {out_tensor.shape}")

            if out_tensor.shape[-1] != output_len:
                print(
                    f"WARNING: Output length {out_tensor.shape[-1]} does not match requested {output_len}"
                )

        except Exception as e:
            print(f"Forward pass failed: {e}")

        params = count_parameters(model)
        print(f"Total trainable parameters: {params}")

    except Exception as e:
        print(f"Model instantiation failed: {e}")


if __name__ == "__main__":
    main()
