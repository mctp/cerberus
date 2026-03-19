
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from cerberus.loss import MSEMultinomialLoss, PoissonMultinomialLoss, NegativeBinomialMultinomialLoss
from cerberus.output import ProfileCountOutput

def analyze_losses(output_len=1024, total_count=1000):
    # Setup inputs
    B = 1
    C = 1
    L = output_len
    N = total_count
    
    # Create random targets
    # Random distribution of counts summing to N
    # Use multinomial to generate counts
    probs_true = torch.softmax(torch.randn(B, C, L), dim=-1)
    targets = torch.distributions.Multinomial(total_count=N, probs=probs_true).sample()
    # targets shape: (B, C, L)
    
    # Create predictions
    # Case 1: Perfect prediction
    logits_perfect = torch.log(probs_true + 1e-9)
    log_counts_perfect = torch.log(torch.tensor([N], dtype=torch.float)).view(B, C) # or (B,) depending on shape expectation
    # Note: MSEMultinomialLoss expects log_counts in (B, C) or (B,) depending on config.
    # But usually (B, C) for ProfileCountOutput if coming from model.
    # The loss function flattens if needed.
    
    output_perfect = ProfileCountOutput(logits=logits_perfect, log_counts=log_counts_perfect)
    
    # Case 2: Noisy prediction
    # Perturb logits and counts
    logits_noisy = logits_perfect + 0.1 * torch.randn_like(logits_perfect)
    log_counts_noisy = torch.log(torch.tensor([N * 1.1], dtype=torch.float)).view(B, C) # 10% error in count
    
    output_noisy = ProfileCountOutput(logits=logits_noisy, log_counts=log_counts_noisy)

    # Instantiate losses
    # Both with equal weights as requested
    loss_mse_multi = MSEMultinomialLoss(count_weight=1.0, profile_weight=1.0)
    loss_poiss_multi = PoissonMultinomialLoss(count_weight=1.0, profile_weight=1.0)
    loss_nb_multi = NegativeBinomialMultinomialLoss(count_weight=1.0, profile_weight=1.0, total_count=10.0) # r=10
    
    # Helper to extract components
    def get_components(loss_fn, output, target):
        loss_fn_name = loss_fn.__class__.__name__
        
        # We need to hackily extract components because forward returns the sum
        # But we can replicate the logic or just call forward and rely on the fact that we can separate them 
        # by calling with 0 weights.
        
        loss_fn.count_weight = 1.0
        loss_fn.profile_weight = 0.0
        c_loss = loss_fn(output, target).item()
        
        loss_fn.count_weight = 0.0
        loss_fn.profile_weight = 1.0
        p_loss = loss_fn(output, target).item()
        
        loss_fn.count_weight = 1.0 # Restore
        
        return c_loss, p_loss
        
    print(f"--- Analysis for L={L}, N={N} ---")
    
    # MSE + Multinomial
    c_mse, p_multi = get_components(loss_mse_multi, output_noisy, targets)
    print(f"MSEMultinomialLoss (Noisy): Count (MSE) = {c_mse:.4f}, Profile (MultiNLL) = {p_multi:.4f}")
    
    # Poisson + Multinomial (CrossEntropy)
    c_poiss, p_ce = get_components(loss_poiss_multi, output_noisy, targets)
    print(f"PoissonMultinomialLoss (Noisy): Count (PoissNLL) = {c_poiss:.4f}, Profile (CE) = {p_ce:.4f}")

    # NB + Multinomial
    c_nb, p_ce_nb = get_components(loss_nb_multi, output_noisy, targets)
    print(f"NegativeBinomialMultinomialLoss (Noisy): Count (NB) = {c_nb:.4f}, Profile (CE) = {p_ce_nb:.4f}")

    return {
        "L": L,
        "N": N,
        "MSE_Count": c_mse,
        "Multi_Profile": p_multi,
        "Poiss_Count": c_poiss,
        "NB_Count": c_nb
    }

if __name__ == "__main__":
    results = []
    # Test different N to see scaling
    for n in [100, 1000, 10000]:
        results.append(analyze_losses(output_len=1024, total_count=n))
    
    # Also vary L to see effect
    for l in [512, 1024, 2048]:
        results.append(analyze_losses(output_len=l, total_count=1000))
    
    df = pd.DataFrame(results)
    print("\nSummary Table:")
    print(df)
