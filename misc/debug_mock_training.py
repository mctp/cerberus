
import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in path
project_root = Path(os.getcwd())
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Cerberus imports
from cerberus.config import GenomeConfig, DataConfig, SamplerConfig, TrainConfig
from cerberus.datamodule import CerberusDataModule
from cerberus.dataset import CerberusDataset
from cerberus.models.gopher import GlobalProfileCNN
from cerberus.metrics import DefaultMetricCollection
from cerberus.loss import ProfilePoissonNLLLoss
from cerberus.module import CerberusModule
from cerberus.train import _train as train

# Mock imports
try:
    from tests.mock_utils import (
        MockSampler,
        MockSequenceExtractor,
        MockSignalExtractor,
        insert_ggaa_motifs,
        GaussianSignalGenerator
    )
except ImportError:
    sys.path.append(str(project_root / "tests"))
    from mock_utils import ( # type: ignore
        MockSampler,
        MockSequenceExtractor,
        MockSignalExtractor,
        insert_ggaa_motifs,
        GaussianSignalGenerator
    )

class MockDataModule(CerberusDataModule):
    def setup(self, stage=None, batch_size=None, num_workers=0, in_memory=False):
        self.batch_size = batch_size or self.batch_size
        self.num_workers = num_workers
        
        # 1000 samples
        sampler = MockSampler(
            num_samples=1000,
            chroms=["chr1"],
            chrom_size=1_000_000,
            interval_length=2048,
            seed=42
        )
        
        seq_extractor = MockSequenceExtractor(
            fasta_path=None, 
            motif_inserters=[insert_ggaa_motifs]
        )
        
        target_extractor = MockSignalExtractor(
            sequence_extractor=seq_extractor,
            signal_generator=GaussianSignalGenerator(sigma=10.0, base_height=10.0)
        )
        
        input_extractor = None
        data_config_no_inputs = self.data_config.copy()
        data_config_no_inputs["inputs"] = {}

        full_dataset = CerberusDataset(
            genome_config=self.genome_config,
            data_config=data_config_no_inputs,
            sampler_config=self.sampler_config,
            sampler=sampler,
            sequence_extractor=seq_extractor,
            input_signal_extractor=input_extractor,
            target_signal_extractor=target_extractor,
            in_memory=in_memory # type: ignore
        )
        
        self.train_dataset, self.val_dataset, self.test_dataset = full_dataset.split_folds()
        self._is_initialized = True

# Dummy config setup
dummy_dir = project_root / "tmp_debug_mock"
dummy_dir.mkdir(exist_ok=True)
(dummy_dir / "genome.fa").touch()
(dummy_dir / "exclude.bed").touch()
(dummy_dir / "input.bw").touch()
(dummy_dir / "target.bw").touch()

genome_config = {
    "name": "mock_genome",
    "fasta_path": dummy_dir / "genome.fa",
    "exclude_intervals": {"blacklist": dummy_dir / "exclude.bed"},
    "allowed_chroms": ["chr1"],
    "chrom_sizes": {"chr1": 1_000_000},
    "fold_type": "chrom_partition",
    "fold_args": {"k": 5}
}

data_config = {
    "inputs": {"input1": dummy_dir / "input.bw"},
    "targets": {"target1": dummy_dir / "target.bw"},
    "input_len": 2048,
    "output_len": 2048,
    "max_jitter": 0,
    "output_bin_size": 1,
    "encoding": "ACGT",
    "log_transform": False,
    "reverse_complement": False,
    "use_sequence": True
}

sampler_config = {
    "sampler_type": "interval",
    "padded_size": 2048,
    "sampler_args": {"intervals_path": dummy_dir / "exclude.bed"}
}

train_config = {
    "batch_size": 16,
    "max_epochs": 15, # Enough to learn
    "learning_rate": 2e-3,
    "weight_decay": 0.0,
    "patience": 5,
    "optimizer": "adamw",
    "filter_bias_and_bn": True,
    "scheduler_type": "cosine",
    "scheduler_args": {
        "num_epochs": 15,
        "warmup_epochs": 2,
        "min_lr": 1e-5
    }
}

def analyze_predictions():
    datamodule = MockDataModule(
        genome_config=genome_config, # type: ignore
        data_config=data_config, # type: ignore
        sampler_config=sampler_config, # type: ignore
    )

    model = GlobalProfileCNN(input_len=2048, output_len=2048, output_bin_size=1)
    
    module = CerberusModule(
        model=model,
        train_config=train_config, # type: ignore
        criterion=ProfilePoissonNLLLoss(log_input=True, full=False),
        metrics=DefaultMetricCollection(num_channels=1)
    )

    # Train
    train(
        module=module,
        datamodule=datamodule,
        train_config=train_config, # type: ignore
        num_workers=0,
        accelerator="auto",
        devices=1,
        limit_train_batches=50,
        limit_val_batches=20,
        enable_checkpointing=True, 
        logger=True,
        default_root_dir=str(dummy_dir / "logs")
    )

    print("\n--- Analysis ---")
    module.eval()
    val_loader = datamodule.val_dataloader()
    
    pos_max_preds = []
    neg_max_preds = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"]
            targets = batch["targets"]
            
            preds = model(inputs)
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = torch.exp(preds)
            
            for i in range(inputs.shape[0]):
                target_sig = targets[i, 0, :]
                pred_sig = preds[i, 0, :]
                
                # Determine if positive or negative sample
                # If target has peaks > 1.0 (arbitrary low threshold, peak is ~10.0)
                if target_sig.max() > 1.0:
                    pos_max_preds.append(pred_sig.max().item())
                else:
                    neg_max_preds.append(pred_sig.max().item())
    
    pos_max_preds = np.array(pos_max_preds)
    neg_max_preds = np.array(neg_max_preds)
    
    print(f"Num Positive Samples: {len(pos_max_preds)}")
    print(f"Num Negative Samples: {len(neg_max_preds)}")
    
    if len(pos_max_preds) > 0:
        print("\nPositive Samples (Target > 1.0) Prediction Max Stats:")
        print(f"  Min: {pos_max_preds.min():.4f}")
        print(f"  Max: {pos_max_preds.max():.4f}")
        print(f"  Mean: {pos_max_preds.mean():.4f}")
        print(f"  Median: {np.median(pos_max_preds):.4f}")
        
    if len(neg_max_preds) > 0:
        print("\nNegative Samples (Target <= 1.0) Prediction Max Stats:")
        print(f"  Min: {neg_max_preds.min():.4f}")
        print(f"  Max: {neg_max_preds.max():.4f}")
        print(f"  Mean: {neg_max_preds.mean():.4f}")
        print(f"  Median: {np.median(neg_max_preds):.4f}")

    print("\n--- Manual Test on Controlled Sequences ---")
    # Poly-C sequence (Guaranteed No GGAA)
    # ACGT: A=0, C=1, G=2, T=3
    manual_inputs = torch.zeros(10, 4, 2048) # 10 samples
    manual_inputs[:, 1, :] = 1.0 
    
    manual_preds = model(manual_inputs)
    if isinstance(manual_preds, tuple):
        manual_preds = manual_preds[0]
    manual_preds = torch.exp(manual_preds)
    
    print("Poly-C (No GGAA) Prediction Max Stats:")
    max_vals = manual_preds.max(dim=2).values.detach().numpy().flatten()
    print(f"  Min: {max_vals.min():.4f}")
    print(f"  Max: {max_vals.max():.4f}")
    print(f"  Mean: {max_vals.mean():.4f}")
    
    # Sequence with exactly ONE GGAA in the middle.
    manual_pos_inputs = torch.zeros(10, 4, 2048)
    manual_pos_inputs[:, 1, :] = 1.0 # Poly-C background
    # Insert GGAA at 1000: G=2, A=0
    manual_pos_inputs[:, 1, 1000:1004] = 0.0 # Remove C
    manual_pos_inputs[:, 2, 1000] = 1.0 # G
    manual_pos_inputs[:, 2, 1001] = 1.0 # G
    manual_pos_inputs[:, 0, 1002] = 1.0 # A
    manual_pos_inputs[:, 0, 1003] = 1.0 # A
    
    manual_pos_preds = model(manual_pos_inputs)
    if isinstance(manual_pos_preds, tuple):
        manual_pos_preds = manual_pos_preds[0]
    manual_pos_preds = torch.exp(manual_pos_preds)
    
    print("\nPoly-C with one GGAA Prediction Max Stats:")
    pos_max_vals = manual_pos_preds.max(dim=2).values.detach().numpy().flatten()
    print(f"  Min: {pos_max_vals.min():.4f}")
    print(f"  Max: {pos_max_vals.max():.4f}")
    print(f"  Mean: {pos_max_vals.mean():.4f}")

    # Suggest threshold based on manual test
    neg_99 = np.percentile(max_vals, 99)
    pos_01 = np.percentile(pos_max_vals, 1)
    print("\nSuggested Threshold Analysis (Manual):")
    print(f"  Negative (Poly-C) 99th percentile: {neg_99:.4f}")
    print(f"  Positive (Single GGAA) 1st percentile: {pos_01:.4f}")
    if neg_99 < pos_01:
        print(f"  Clear separation possible around {(neg_99 + pos_01)/2:.4f}")
    else:
        print("  Overlapping distributions.")

    # Cleanup
    import shutil
    if dummy_dir.exists():
        shutil.rmtree(dummy_dir)

if __name__ == "__main__":
    analyze_predictions()
