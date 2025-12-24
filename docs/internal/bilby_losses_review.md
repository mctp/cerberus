# Loss Function Review: Bilby vs Cerberus

This document identifies loss functions used in `bilby` (from `tmp/bilby_code_dump.py`) that could be re-implemented in `cerberus`.

## 1. PoissonLoss

**Bilby Implementation:**
```python
def poisson_loss (y_pred, y_true, epsilon=epsilon):
    y_pred = y_pred + epsilon
    y_true = y_true + epsilon
    return jnp.mean (y_pred - y_true * jnp.log(y_pred))
```

**Cerberus Status:** Not implemented.
**Re-implementation Notes:**
- PyTorch has `nn.PoissonNLLLoss` which computes `input - target * log(input)` (if `log_input=False`).
- This loss is useful for models that predict counts directly at each position/bin, assuming a Poisson distribution, without decoupling the total count from the profile.

## 2. PoissonMultinomialLoss

**Bilby Implementation:**
```python
def poisson_multinomial_loss (y_pred, y_true, epsilon=epsilon, total_weight=1., rescale=False):
    seq_len = y_pred.shape[-2]
    # ...
    s_pred = jnp.sum (y_pred, axis=-2, keepdims=True)
    s_true = jnp.sum (y_true, axis=-2, keepdims=True)
    p_loss = poisson_loss (s_pred, s_true, epsilon=0.) / seq_len
    m_loss = -jnp.mean (y_true * jnp.log(y_pred / s_pred))
    return (m_loss + total_weight*p_loss) * ...
```

**Cerberus Status:** `BPNetLoss` is similar but uses MSE for total counts.
**Comparison:**
- **Profile Loss:** Both use Multinomial NLL. `bilby` calculates it as `-mean(y_true * log(probs))`. `BPNetLoss` uses exact combinatorial terms or `log_softmax` + `nll_loss`.
- **Count Loss:** `bilby` uses `poisson_loss` on the total counts (`s_pred`, `s_true`). `BPNetLoss` uses `MSELoss` on `log(total_counts + 1)`.
- **Scaling:** `bilby` scales `p_loss` by `1/seq_len`.

**Re-implementation Potential:**
- A `PoissonMultinomialLoss` class could be added to `cerberus` that allows choosing between MSE and Poisson for the count loss.
- Alternatively, `BPNetLoss` could be extended to support a `count_loss_type` parameter ("mse" or "poisson").

## 3. WeightedPoissonLoss

**Bilby Implementation:**
```python
def weighted_poisson_loss (y_pred, y_true, weight, epsilon=epsilon):
    # ...
    loss = y_pred - y_true * jnp.log(y_pred)
    weighted_loss = weight * loss
    return jnp.mean (weight * (y_pred - y_true * jnp.log(y_pred)))
```

**Cerberus Status:** Not implemented.
**Re-implementation Notes:**
- Useful if we want to weight specific examples or bins differently.

## Summary of Recommendations

1.  **Implement `PoissonLoss`**: A wrapper around `nn.PoissonNLLLoss` or a custom implementation to handle specific input shapes/transforms consistent with `cerberus` conventions.
2.  **Extend `BPNetLoss` or create `PoissonMultinomialLoss`**: Add support for using Poisson loss for the total count prediction, as an alternative to the existing MSE log-count loss.
