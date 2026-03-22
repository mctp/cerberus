import unittest

import torch
import torch.nn.functional as F

from cerberus.models.bpnet import BPNetLoss


def reference_multinomial_nll(logits, true_counts):
    """Compute the multinomial negative log-likelihood in PyTorch.
    From tmp/chrombpnet-pytorch_code_dump.py
    """
    # Ensure true_counts is an integer tensor
    true_counts = true_counts.to(
        torch.float
    )  # Keep as float to prevent conversion issues

    # Compute total counts per example (should already be integer-like)
    counts_per_example = true_counts.sum(dim=-1, keepdim=True)

    # Convert logits to log probabilities (Softmax + Log)
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute log-probability of the observed counts
    log_likelihood = (true_counts * log_probs).sum(dim=-1)

    # Compute multinomial coefficient (log factorial term)
    log_factorial_counts = torch.lgamma(counts_per_example + 1) - torch.lgamma(
        true_counts + 1
    ).sum(dim=-1)

    # Compute final NLL
    nll = -(log_factorial_counts + log_likelihood).mean()

    return nll


class TestBPNetLossComparison(unittest.TestCase):
    def test_loss_equivalence(self):
        batch_size = 10
        seq_len = 1000
        channels = 1

        # Random logits (B, C, L)
        logits = torch.randn(batch_size, channels, seq_len, requires_grad=True)
        # Random counts (B, C, L) - discrete-like floats
        true_counts = torch.floor(
            torch.abs(torch.randn(batch_size, channels, seq_len) * 10)
        )

        # Cerberus Loss
        # BPNetLoss inherits from MSEMultinomialLoss
        # alpha is count weight, beta is profile weight
        alpha = 1.0
        beta = 1.0
        cerberus_loss_fn = BPNetLoss(alpha=alpha, beta=beta)

        # Cerberus requires ProfileCountOutput
        from cerberus.output import ProfileCountOutput

        # Fake log_counts (B, C)
        log_counts = torch.randn(batch_size, channels, requires_grad=True)
        outputs = ProfileCountOutput(logits=logits, log_counts=log_counts)

        # Cerberus forward
        # MSEMultinomialLoss expects targets same shape as logits/counts
        # For counts, it computes internal sums if count_per_channel=False

        cerberus_loss_fn(outputs, true_counts)

        # Reference Loss
        # Reference calculates profile loss using MNLL
        # We need to reshape logits/counts for reference function which expects (B, N_classes)
        # Reference uses flatten channels implicitly if we treat (C*L) as classes?
        # No, reference takes (batch_size, num_classes).
        # In BPNet (chrombpnet), output is (B, Out_dim) if n_outputs=1.
        # But wait, if n_outputs > 1 (stranded), it might be (B, C, L).
        # The reference code `multinomial_nll` takes `logits` and `true_counts`.
        # `log_probs = F.log_softmax(logits, dim=-1)`
        # If input is (B, C, L), softmax is over L.
        # `log_likelihood = (true_counts * log_probs).sum(dim=-1)` -> (B, C)
        # `nll = - (...).mean()` -> mean over B and C.

        # So we can pass (B, C, L) directly to reference function.
        loss_profile_ref = reference_multinomial_nll(logits, true_counts)

        # Calculate Cerberus profile loss component manually to verify
        # _compute_profile_loss in MSEMultinomialLoss with flatten_channels=False, average_channels=True
        # profile_counts = targets.sum(dim=-1)
        # ...
        # profile_loss_per_channel = ...
        # return profile_loss_per_channel.mean()

        # We need to extract the profile loss part from Cerberus loss.
        # BPNetLoss: loss = beta * profile + alpha * count
        # Let's set alpha=0 to compare only profile loss.

        cerberus_loss_fn_profile = BPNetLoss(alpha=0.0, beta=1.0)
        loss_cerberus_profile = cerberus_loss_fn_profile(outputs, true_counts)

        print(f"Cerberus Profile Loss: {loss_cerberus_profile.item()}")
        print(f"Reference Profile Loss: {loss_profile_ref.item()}")

        self.assertTrue(
            torch.isclose(loss_cerberus_profile, loss_profile_ref, atol=1e-5)
        )

        # Now Check Count Loss
        # Reference:
        # pred_count = self.linear(pred_count) -> (B, 1) or (B, C)?
        # In BPNet.count_head: pred_count = self.linear(pred_count).
        # If n_outputs=1, linear outputs 1.
        # In BPNetWrapper._step:
        # count_loss = F.mse_loss(y_count, true_counts)
        # where true_counts = torch.log1p(true_profile.sum(dim=-1))
        # and y_count is squeezed (B, 1) -> (B) ? No, squeezed(-1).
        # If output is (B, 1), squeeze(-1) -> (B).
        # true_counts: true_profile.sum(dim=-1) is (B, C).
        # If C=1, (B, 1).

        # In Cerberus BPNetLoss:
        # count_per_channel=False (default in BPNetLoss)
        # target_global_count = targets.sum(dim=(1, 2)) # Sum over C and L
        # target_log_global_count = torch.log1p(target_global_count)
        # pred_log_counts = pred_log_counts.flatten()
        # count_loss = F.mse_loss(pred_log_counts, target_log_global_count)

        # Wait, difference here!
        # Reference: `true_counts = torch.log1p(true_profile.sum(dim=-1))`
        # If C=1, sum(dim=-1) sums over L. Result (B, 1).
        # Cerberus: `targets.sum(dim=(1, 2))` sums over C and L. Result (B).
        # If C=1, these are identical (squeezed).

        # Let's verify with C=1
        logits_1c = torch.randn(batch_size, 1, seq_len)
        true_counts_1c = torch.floor(
            torch.abs(torch.randn(batch_size, 1, seq_len) * 10)
        )
        log_counts_1c = torch.randn(batch_size, 1)

        outputs_1c = ProfileCountOutput(logits=logits_1c, log_counts=log_counts_1c)

        # Reference Count Loss
        target_counts_ref = torch.log1p(true_counts_1c.sum(dim=-1))  # (B, 1)
        loss_count_ref = F.mse_loss(
            log_counts_1c.squeeze(-1), target_counts_ref.squeeze(-1)
        )

        # Cerberus Count Loss (alpha=1, beta=0)
        cerberus_loss_fn_count = BPNetLoss(alpha=1.0, beta=0.0)
        loss_cerberus_count = cerberus_loss_fn_count(outputs_1c, true_counts_1c)

        print(f"Cerberus Count Loss (C=1): {loss_cerberus_count.item()}")
        print(f"Reference Count Loss (C=1): {loss_count_ref.item()}")

        self.assertTrue(torch.isclose(loss_cerberus_count, loss_count_ref, atol=1e-5))

        # What if C > 1?
        # Reference: `true_counts = torch.log1p(true_profile.sum(dim=-1))` -> (B, C)
        # `y_count` from model -> (B, C) (if n_outputs=C?)
        # In reference BPNet model:
        # `self.linear = torch.nn.Linear(n_filters+n_count_control, 1)` -> Always outputs 1 channel?
        # `pred_count = self.linear(pred_count)` -> (B, 1)
        # But `n_outputs` in reference is usually 1 (unstranded) or 2 (stranded).
        # If `n_outputs=2`, `fconv` output is 2 channels.
        # But `linear` is hardcoded to 1 output?
        # `self.linear = torch.nn.Linear(n_filters+n_count_control, 1, bias=count_output_bias)`
        # Yes, output is 1.
        # But `BPNetWrapper._step` uses `true_counts = torch.log1p(true_profile.sum(dim=-1))`.
        # If true_profile is (B, 2, L), true_counts is (B, 2).
        # y_count is (B, 1).
        # F.mse_loss(y_count, true_counts) -> broadcasting?
        # (B, 1) vs (B, 2).
        # It will broadcast (B, 1) to (B, 2).
        # It compares predicted total count (scalar) to BOTH strand counts? That seems wrong.
        # Or maybe reference BPNet model predicts 1 count which represents total count?
        # Reference docstring: "The count prediction task is predicting the total counts across both strands."

        # If reference predicts total counts, then `true_counts` should be total counts.
        # `true_profile.sum(dim=-1)` gives per-strand counts.
        # `log1p(...)` gives log per-strand counts.
        # If we compare log(total_pred) with log(strand_count), that's weird.
        # Unless... `BPNetWrapper` handles this?
        # Let's look at `BPNet.count_head` in reference:
        # `pred_count = self.linear(pred_count)` -> (B, 1).

        # Wait, checking `tmp/chrombpnet-pytorch_code_dump.py` again.
        # `BPNet` class docstring: "(4) The count prediction task is predicting the total counts across both strands."
        # `BPNetWrapper._step`:
        # `true_counts = torch.log1p(true_profile.sum(dim=-1))`
        # If `true_profile` has 2 channels, `true_counts` has 2 values per batch.
        # `y_count` has 1 value per batch (from linear(..., 1)).
        # `F.mse_loss(y_count, true_counts)`
        # If y_count is (B, 1) and true_counts is (B, 2).
        # (B, 1) expands to (B, 2).
        # Loss = 0.5 * ((y - log(c1))^2 + (y - log(c2))^2).
        # This forces `y` to be close to both `log(c1)` and `log(c2)`.
        # i.e. `y` tries to be `(log(c1) + log(c2)) / 2`.
        # This is the geometric mean of counts?
        # This contradicts "predicting the total counts".
        # Total count would be `log(c1 + c2)`.

        # However, `cerberus` BPNet implementation:
        # `num_count_outputs = 1 if self.predict_total_count else self.n_output_channels`
        # Default `predict_total_count=True`. So 1 output.
        # `BPNetLoss`:
        # `count_per_channel=False`.
        # `target_global_count = targets.sum(dim=(1, 2))` -> Sums over strands and length. -> Total count c1+c2.
        # `target_log_global_count = torch.log1p(target_global_count)` -> log(c1+c2).
        # `count_loss = F.mse_loss(pred_log_counts, target_log_global_count)`.
        # This seems CORRECT for "Total Count".

        # The reference implementation in `BPNetWrapper._step` seems to be doing `log(c1)` and `log(c2)` comparison if multiple channels.
        # If `n_outputs=1` (standard for many tasks or merged strands), then it's fine.
        # The provided example `chip_ar_mdapca2b_bpnet.py` uses:
        # `"output_channels": ["signal"]` -> 1 channel.
        # So C=1.
        # In this case, Cerberus and Reference are identical.

        pass


if __name__ == "__main__":
    unittest.main()
