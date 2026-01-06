import torch
from cerberus.models.geminet import GeminetMedium

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Instantiate GeminetMedium with dummy lengths as they don't affect parameter count
    model = GeminetMedium(input_len=2048, output_len=1000)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters in GeminetMedium: {total_params:,}")

    # Detailed breakdown
    print("\nDetailed breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")

if __name__ == "__main__":
    main()
