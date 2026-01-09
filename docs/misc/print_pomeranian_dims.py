import torch
from cerberus.models.pomeranian import Pomeranian

def print_layer_dims():
    print("Initializing Pomeranian...")
    # Use defaults: 2112 -> 1024
    input_len = 2112
    output_len = 1024
    model = Pomeranian(input_len=input_len, output_len=output_len)
    
    print(f"Input Length: {input_len}")
    print(f"Output Length: {output_len}")
    print("-" * 50)
    
    x = torch.randn(1, 4, input_len)
    
    def get_shape_hook(name):
        def hook(model, input, output):
            # output might be a tuple or tensor
            if isinstance(output, tuple):
                shape = [tuple(o.shape) for o in output]
            elif isinstance(output, torch.Tensor):
                shape = tuple(output.shape)
            else:
                shape = type(output)
            print(f"{name}: {shape}")
        return hook

    # Register hooks
    model.stem.register_forward_hook(get_shape_hook("Stem"))
    
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(get_shape_hook(f"Layer {i+1} (PGC)"))
        
    model.profile_pointwise.register_forward_hook(get_shape_hook("Profile Pointwise"))
    model.profile_act.register_forward_hook(get_shape_hook("Profile Act"))
    model.profile_spatial.register_forward_hook(get_shape_hook("Profile Spatial (Logits)"))
    
    # Count head hooks
    # count_mlp is a Sequential
    model.count_mlp[0].register_forward_hook(get_shape_hook("Count MLP Layer 1"))
    model.count_mlp[1].register_forward_hook(get_shape_hook("Count MLP Act"))
    model.count_mlp[2].register_forward_hook(get_shape_hook("Count MLP Layer 2 (Log Counts)"))

    print("Running Forward Pass...")
    _ = model(x)
    print("-" * 50)

if __name__ == "__main__":
    print_layer_dims()
