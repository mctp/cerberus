
import torch
import torch.nn as nn
from cerberus.models.geminet import GemiNet
import warnings

# Filter other warnings to focus on the one we care about
# warnings.simplefilter("ignore") 
# We actually want to see the warning, so we don't ignore.

def test_geminet_amp():
    print("Initializing GemiNet...")
    model = GemiNet(
        input_len=1000,
        output_len=1000,
        filters=32,
        n_dilated_layers=2
    ).cuda()
    
    # Dummy input: Batch, 4, Length
    x = torch.zeros(1, 4, 1000).cuda()
    
    print("Running forward pass with autocast...")
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(x)
    print("Forward pass complete.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_geminet_amp()
    else:
        print("CUDA not available, cannot test AMP warning reproduction properly as it requires Half execution.")
