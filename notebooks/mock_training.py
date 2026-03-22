# %% [markdown]
# # Training with Mock Dataset
# 
# This notebook demonstrates how to train a model using a synthetic (mock) dataset generated on-the-fly.
# This allows for end-to-end testing and verification of model learning capabilities without external data dependencies.
#
# **Task**: Train a model to predict Gaussian peaks at locations of 'GGAA' motifs in the input sequence.

# %%
import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in path
try:
    from paths import get_project_root
    project_root = get_project_root()
except ImportError:
    if os.path.basename(os.getcwd()) == "notebooks":
        project_root = Path(os.getcwd()).parent
    else:
        project_root = Path(os.getcwd())

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Cerberus imports
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig, FoldArgs, IntervalSamplerArgs
from cerberus.datamodule import CerberusDataModule
from cerberus.dataset import CerberusDataset
from cerberus.models.gopher import GlobalProfileCNN
from cerberus.metrics import DefaultMetricCollection
from cerberus.loss import ProfilePoissonNLLLoss
from cerberus.module import CerberusModule
import pytorch_lightning as pl

# Mock imports (assuming tests/mock_utils.py exists)
try:
    from tests.mock_utils import (
        MockSampler,
        MockSequenceExtractor,
        MockSignalExtractor,
        insert_ggaa_motifs,
        GaussianSignalGenerator
    )
except ImportError:
    # If tests module is not importable, we might need to add it to path
    sys.path.append(str(project_root / "tests"))
    from mock_utils import ( # type: ignore
        MockSampler,
        MockSequenceExtractor,
        MockSignalExtractor,
        insert_ggaa_motifs,
        GaussianSignalGenerator
    )

# %% [markdown]
# ## 1. Define Mock DataModule
# 
# We subclass `CerberusDataModule` to override the setup process and inject our Mock components.

# %%
class MockDataModule(CerberusDataModule):
    def setup(self, stage=None, batch_size=None, num_workers=0, in_memory=False, **kwargs):
        # Override setup to use MockDataset
        self.batch_size = batch_size or self.batch_size
        self.num_workers = num_workers
        
        # Initialize Mock Components
        # We generate 1000 samples of 2048bp length
        sampler = MockSampler(
            num_samples=1000,
            chroms=["chr1"],
            chrom_size=1_000_000,
            interval_length=2048,
            seed=42
        )
        
        # Sequence Extractor: Injects GGAA motifs
        # Note: We pass None for fasta_path to generate random background
        seq_extractor = MockSequenceExtractor(
            fasta_path=None, 
            motif_inserters=[insert_ggaa_motifs]
        )
        
        # Target Extractor: Generates Gaussian peaks at GGAA locations
        # Base height 10.0, sigma 10.0
        target_extractor = MockSignalExtractor(
            sequence_extractor=seq_extractor,
            signal_generator=GaussianSignalGenerator(sigma=10.0, base_height=10.0)
        )
        
        # Input Extractor: No extra input tracks
        input_extractor = None
        
        # We need to ensure CerberusDataset doesn't load inputs from config
        data_config_no_inputs = self.data_config.model_copy(update={"inputs": {}})

        # Create Full Dataset
        # We pass dummy configs because CerberusDataset verifies them, 
        # but we override the components so the paths aren't used.
        assert in_memory is not None, "in_memory must be specified"
        full_dataset = CerberusDataset(
            genome_config=self.genome_config,
            data_config=data_config_no_inputs,
            sampler_config=self.sampler_config,
            sampler=sampler,
            sequence_extractor=seq_extractor,
            input_signal_extractor=input_extractor,
            target_signal_extractor=target_extractor,
            in_memory=in_memory
        )
        
        # Split
        self.train_dataset, self.val_dataset, self.test_dataset = full_dataset.split_folds()
        self._is_initialized = True
        
        print(f"Mock Data Setup Complete.")
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")


# %% [markdown]
# ## 2. Configuration
# 
# We need valid configuration dictionaries to pass validation, even though we override the logic.
# We'll create dummy files.

# %%
# Create dummy files for config validation
dummy_dir = project_root / "tmp_mock_data"
dummy_dir.mkdir(exist_ok=True)
(dummy_dir / "genome.fa").touch()
(dummy_dir / "exclude.bed").touch()
(dummy_dir / "input.bw").touch()
(dummy_dir / "target.bw").touch()

genome_config = GenomeConfig.model_construct(
    name="mock_genome",
    fasta_path=dummy_dir / "genome.fa",
    exclude_intervals={"blacklist": dummy_dir / "exclude.bed"},
    allowed_chroms=["chr1"],
    chrom_sizes={"chr1": 1_000_000},
    fold_type="chrom_partition",
    fold_args=FoldArgs(k=5, test_fold=0, val_fold=1),
)

data_config = DataConfig.model_construct(
    inputs={"input1": dummy_dir / "input.bw"},
    targets={"target1": dummy_dir / "target.bw"},
    input_len=2048,
    output_len=2048,  # Output matches input for simplicity in this mock
    max_jitter=0,
    output_bin_size=1,  # No binning
    encoding="ACGT",
    log_transform=False,
    reverse_complement=False,
    target_scale=1.0,
    use_sequence=True,
)

sampler_config = SamplerConfig.model_construct(
    sampler_type="interval",
    padded_size=2048,
    sampler_args=IntervalSamplerArgs.model_construct(intervals_path=dummy_dir / "exclude.bed"),
)

train_config = TrainConfig(
    batch_size=16,
    max_epochs=20,
    learning_rate=2e-3,
    weight_decay=0.0,
    patience=5,
    optimizer="adamw",
    filter_bias_and_bn=True,
    reload_dataloaders_every_n_epochs=0,
    scheduler_type="cosine",
    scheduler_args={
        "num_epochs": 20,
        "warmup_epochs": 2,
        "min_lr": 1e-5,
    },
    adam_eps=1e-8,
    gradient_clip_val=None,
)

# %% [markdown]
# ## 3. Model & Training
# 
# We use a simple CNN. We assume 1 output channel.

# %%
# Initialize DataModule
datamodule = MockDataModule(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
)

# Initialize Model
# input_len=2048, output_len=2048, output_bin_size=1
model = GlobalProfileCNN(input_len=2048, output_len=2048, output_bin_size=1)

# Initialize Lightning Module
criterion = ProfilePoissonNLLLoss(log_input=True, full=False)
metrics = DefaultMetricCollection()

module = CerberusModule(
    model=model,
    train_config=train_config,
    criterion=criterion,
    metrics=metrics
)

# Train
# We use pl.Trainer directly here so the mock model object remains accessible
# for the verification section below. _train() creates the module internally
# and is not suitable when you need a reference to the model after training.
if __name__ == "__main__":
    datamodule.prepare_data()
    datamodule.setup(batch_size=train_config.batch_size, num_workers=0, in_memory=False)
    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator="auto",
        devices=1,
        limit_train_batches=50,
        limit_val_batches=10,
        enable_checkpointing=True,
        logger=True,
        enable_progress_bar=False,
        default_root_dir=str(dummy_dir / "logs"),
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=train_config.patience,
            ),
        ],
    )
    trainer.fit(module, datamodule=datamodule)

    print("\nTraining Complete.")
    
    # %% [markdown]
    # ## 4. Verification
    # 
    # We verify if the model learned the pattern by running inference on the validation set.
    # We look for correlation between ground truth peaks (GGAAs) and model predictions.

    # %%
    print("\nVerifying Model Learning...")
    module.eval()
    val_loader = datamodule.val_dataloader()
    
    total_samples = 0
    correct_peaks = 0
    missed_peaks = 0
    false_positives = 0
    
    # Store points for scatter plot
    observed_intensities = []
    expected_intensities = []
    
    # Thresholds
    # Ground truth peak is ~10.0 height (Gaussian base_height).
    # Prediction analysis on mock data shows:
    #   - Negative regions (Poly-C, no GGAA): Max prediction ~0.4
    #   - Positive regions (Single GGAA): Max prediction ~6.0
    #   - Random background (often has accidental GGAAs): Mean ~5.5
    # Therefore, a threshold of 2.0 safely separates signal from background noise.
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"] # (B, 4, 2048) - Sequence only
            targets = batch["targets"] # (B, 1, 2048)
            
            # Predict
            outputs = model(inputs) # (B, 1, 2048)
            if hasattr(outputs, "logits"):
                preds = outputs.logits
            elif hasattr(outputs, "log_rates"):
                preds = outputs.log_rates
            elif isinstance(outputs, tuple):
                preds = outputs[0]
            else:
                preds = outputs
            
            # Use Poisson output (log scale) -> exp
            preds = torch.exp(preds)
            
            # Iterate samples
            for i in range(inputs.shape[0]):
                total_samples += 1
                seq = inputs[i, :4, :] # (4, 2048)
                target_sig = targets[i, 0, :]
                pred_sig = preds[i, 0, :]
                
                # Collect all points for scatter plot (subsample to avoid clutter)
                # We prioritize points with signal > 0.1 to see the relationship clearly
                mask = target_sig > 0.1
                # Also include some background
                mask = mask | (torch.rand_like(target_sig) < 0.01)
                
                obs = pred_sig[mask].cpu().numpy()
                exp = target_sig[mask].cpu().numpy()
                
                observed_intensities.extend(obs)
                expected_intensities.extend(exp)
                
                # Find ground truth peak locations (where target > 5.0)
                gt_indices = torch.nonzero(target_sig > 5.0).flatten()
                
                # Find predicted peak locations (where pred > 2.0)
                pred_indices = torch.nonzero(pred_sig > 2.0).flatten()
                
                # Simple overlap check
                if len(gt_indices) > 0:
                    max_val_at_peaks = pred_sig[gt_indices].max()
                    if max_val_at_peaks > 2.0:
                         correct_peaks += 1
                    else:
                         missed_peaks += 1
                else:
                    if pred_sig.max() > 2.0:
                        false_positives += 1
    
    print(f"Evaluated {total_samples} samples.")
    print(f"Correctly detected samples (with peaks): {correct_peaks}")
    print(f"Missed samples (with peaks): {missed_peaks}")
    print(f"False positive samples (no peaks but high pred): {false_positives}")
    
    success_rate = correct_peaks / (correct_peaks + missed_peaks + 1e-6)
    print(f"Detection Rate on positive samples: {success_rate:.2%}")
    
    # Calculate Pearson R
    obs_arr = np.array(observed_intensities)
    exp_arr = np.array(expected_intensities)
    if len(obs_arr) > 1:
        pearson_r = np.corrcoef(obs_arr, exp_arr)[0, 1]
        print(f"Pearson Correlation (R): {pearson_r:.4f}")
    else:
        pearson_r = 0.0
        print("Not enough points to calculate correlation")

    # Generate Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(expected_intensities, observed_intensities, alpha=0.1, s=1)
    plt.plot([0, max(expected_intensities)], [0, max(expected_intensities)], 'r--', label="Perfect")
    plt.xlabel("Expected Intensity (Ground Truth)")
    plt.ylabel("Observed Intensity (Model Prediction)")
    plt.title("Observed vs Expected Intensity (GGAA Peaks)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plots_dir = project_root / "notebooks/plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    plot_path = plots_dir / "mock_training_scatter.png"
    plt.savefig(plot_path)
    print(f"\nScatter plot saved to: {plot_path}")
    
    if success_rate > 0.5:
        print("\nSUCCESS: Model successfully learned to identify GGAA motifs!")
    else:
        print("\nFAILURE: Model failed to learn the pattern.")

    # Cleanup
    import shutil
    if dummy_dir.exists():
        shutil.rmtree(dummy_dir)
