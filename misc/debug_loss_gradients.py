
import torch
import torch.nn.functional as F
import pandas as pd

def compute_gradients():
    # Targets: 10, 100, 1000, 10000
    targets = torch.tensor([10.0, 100.0, 1000.0, 10000.0])
    
    # Predictions: 10% overestimate
    # pred = target * 1.1
    # log_pred = log(target) + log(1.1)
    # We want gradient w.r.t the "logit" or "log_count" output of the network
    # Let x = log_pred.
    
    x = torch.log(targets * 1.1).clone().detach().requires_grad_(True)
    
    # 1. Poisson NLL
    # Loss = exp(x) - target * x
    loss_poisson = F.poisson_nll_loss(x, targets, log_input=True, reduction='none', full=False)
    loss_poisson.sum().backward()
    grad_poisson = x.grad.clone()
    x.grad.zero_()
    
    # 2. MSE on Log Counts
    # Loss = (x - log(target))^2
    target_log = torch.log(targets)
    loss_mse = F.mse_loss(x, target_log, reduction='none')
    loss_mse.sum().backward()
    grad_mse = x.grad.clone()
    x.grad.zero_()
    
    # 3. Negative Binomial NLL
    # Need to map mu=exp(x) to NB parameters (total_count, probs/logits)
    # Assume some dispersion parameter r (total_count)
    # Variance = mu + mu^2 / r
    # Behaving like Log-Normal means Var ~ mu^2, which dominates when mu is large.
    # NB Parameterization:
    # mu = r * (1-p)/p  => p = r / (r + mu)
    # logits = log(p/(1-p)) = log(r/mu) = log(r) - x
    
    # We treat 'r' as a fixed hyperparameter here to check the gradient properties w.r.t x (log-mean)
    r = 10.0 # High dispersion
    
    # We need to compute NLL of the TARGET k given predicted mu = exp(x)
    # PyTorch NegativeBinomial(total_count=r, logits=logits)
    # logits input to NB is log(p / (1-p)).
    # p = r / (r + exp(x))
    # p / (1-p) = r / exp(x)
    # logits_nb = log(r) - x (INCORRECT per PyTorch docs analysis)
    # PyTorch NB mean = total_count * exp(logits).
    # We want mean = exp(x).
    # exp(x) = r * exp(logits_nb)
    # exp(logits_nb) = exp(x) / r
    # logits_nb = x - log(r)
    
    logits_nb = x - torch.log(torch.tensor(r))
    
    # Note: PyTorch NegativeBinomial samples 'counts'. We want NLL of 'targets' given distribution.
    # log_prob(targets).
    nb_dist = torch.distributions.NegativeBinomial(total_count=r, logits=logits_nb)
    loss_nb = -nb_dist.log_prob(targets)
    
    loss_nb.sum().backward()
    grad_nb = x.grad.clone()
    x.grad.zero_()
    
    # 4. Log-Normal NLL
    # equivalent to MSE on logs with sigma=1/sqrt(2) ?
    # P(k) ~ (1/(k sigma sqrt(2pi))) * exp( - (ln k - mu)^2 / 2sigma^2 )
    # This is for continuous k.
    # NLL ~ log(k) + (ln k - mu)^2 / 2sigma^2.
    # If we ignore the log(k) term (constant target) and sigma, it's MSE.
    
    results = []
    for i, t in enumerate(targets):
        results.append({
            "Target": t.item(),
            "Poisson_Grad": grad_poisson[i].item(),
            "MSE_Grad": grad_mse[i].item(),
            "NB_Grad": grad_nb[i].item()
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = compute_gradients()
    print("Gradients w.r.t log_prediction for 10% relative error:")
    print(df)
