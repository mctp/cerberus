# Bias-Factorized Model Implementation Considerations

This document outlines considerations for implementing bias-factorized models (e.g., BPNet style) within the Cerberus framework, specifically regarding `CerberusModule`.

## Current State

The `CerberusModule` has been generalized to support models that output predicted counts/rates directly (e.g., via `Softplus` activation), compatible with ASAP-style architectures. The loss function defaults to `PoissonNLLLoss` on these counts.

## Bias Correction Challenges

Bias correction typically involves subtracting the output of a bias model from the main model's predictions *before* activation (i.e., in logit space).

### BPNet Style (Logits)
In the original BPNet implementation:
1.  Main model outputs `logits` (profile) and `log_counts`.
2.  Bias model outputs `bias_logits` and `bias_log_counts`.
3.  Correction: `final_logits = logits - bias_logits`.
4.  Loss: Multinomial NLL on `final_logits` + MSE on `final_log_counts`.

### ASAP Style (Counts)
Current ASAP models (like `ConvNeXtCNN`) typically output `counts` directly (using `Softplus` or similar).
1.  Model output: `counts` (positive).
2.  Bias model output: `bias_counts`?
3.  Correction: Subtracting counts directly (`counts - bias_counts`) can lead to negative values, which invalidates `PoissonNLLLoss`.
4.  Correction in Log Space: We could compute `log(counts) - log(bias_counts)`, but this requires inverting the model's activation or changing the model to output logits.

## Future Implementation Requirements

To support bias-factorized models properly, we need to:

1.  **Model Interface**: The model should optionally return logits instead of activated counts, or the `CerberusModule` needs to know which space the model outputs live in.
2.  **Bias Model Compatibility**: The bias model must output in the same space as the main model.
3.  **Arithmetic Safety**: Ensure subtraction is performed in logit space (or additive log-space) to ensure final predictions remain valid for the loss function (e.g., applying Softmax/Softplus *after* bias subtraction).

## Moved Comments

The following comments were removed from `CerberusModule` during refactoring:

> Note: Bias correction implementation depends on whether we work with counts or logits.
> Current ASAP models output counts (Softplus).
> We skip bias subtraction for now to avoid incorrect arithmetic on counts.
