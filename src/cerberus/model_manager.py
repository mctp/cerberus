from pathlib import Path
from typing import List, Dict
import torch
from torch import nn
import re

from cerberus.config import (
    GenomeConfig,
    DataConfig,
    TrainConfig,
    ModelConfig,
)
from cerberus.genome import create_genome_folds
from cerberus.entrypoints import instantiate


class ModelManager:
    """
    Manages loading and caching of models for different folds.
    """
    def __init__(
        self,
        checkpoint_path: Path | str,
        model_config: ModelConfig,
        data_config: DataConfig,
        train_config: TrainConfig,
        genome_config: GenomeConfig,
        device: torch.device,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.genome_config = genome_config
        self.device = device
        
        self.cache: Dict[str, nn.Module] = {}
        self.is_multifold = self._detect_multifold()
        
        if self.is_multifold:
            self.folds = create_genome_folds(
                genome_config["chrom_sizes"],
                genome_config["fold_type"],
                genome_config["fold_args"],
            )
        else:
            self.folds = []

    def _detect_multifold(self) -> bool:
        if self.checkpoint_path.is_dir():
            # Check for fold_0, fold_1, etc.
            return any((self.checkpoint_path / f"fold_{i}").exists() for i in range(2))
        return False

    def _select_best_checkpoint(self, checkpoints: List[Path]) -> Path:
        """
        Selects the best checkpoint from a list based on validation loss.
        Assumes filename format like '...val_loss=0.1234...' or '...val_loss-0.1234...'.
        If no metric found, sorts by name.
        """
        def get_val_loss(p: Path) -> float:
            # Pattern to match val_loss=0.1234
            # Usually PyTorch Lightning formats it as key=value
            # Use specific regex to avoid capturing trailing dot from .ckpt
            match = re.search(r"val_loss[=_](\d+\.?\d*)", p.name)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
            return float('inf')
        
        # Sort by val_loss ascending, then by name (for determinism)
        sorted_ckpts = sorted(checkpoints, key=lambda p: (get_val_loss(p), p.name))
        return sorted_ckpts[0]

    def get_models(self, chrom: str, start: int, end: int, use_folds: List[str] = ["test", "val"]) -> List[nn.Module]:
        """
        Returns list of models applicable for the interval.
        
        Args:
            chrom: Chromosome name.
            start: Start coordinate.
            end: End coordinate.
            use_folds: List of fold roles to include ('train', 'test', 'val').
                       Default is ['test', 'val'].
        """
        if not self.is_multifold:
            return [self._load_model("single", self.checkpoint_path)]
        
        # Identify which partition(s) this interval belongs to
        # In standard cross-validation, intervals usually belong to exactly one partition (fold).
        # We find the fold index 'x' where interval is in folds[x].
        
        target_partitions = set()
        for fold_idx, fold_map in enumerate(self.folds):
            if chrom in fold_map:
                if any(fold_map[chrom].find((start, end - 1))):
                     target_partitions.add(fold_idx)
        
        k = len(self.folds)
        models_to_load = set()
        
        for p_idx in target_partitions:
            # For a partition p_idx (where the data resides):
            
            # 1. 'test': The model trained with p_idx as test set is model `fold_p_idx`.
            if "test" in use_folds:
                models_to_load.add(p_idx)
            
            # 2. 'val': The model trained with p_idx as validation set.
            # In train_multi: test_fold=i, val_fold=(i+1)%k.
            # So partition p_idx is VAL for model `i` where (i+1)%k == p_idx.
            # i = (p_idx - 1) % k.
            if "val" in use_folds:
                val_model_idx = (p_idx - 1) % k
                models_to_load.add(val_model_idx)
                
            # 3. 'train': The models trained with p_idx as training set.
            # All other models except test (p_idx) and val ((p_idx-1)%k).
            if "train" in use_folds:
                test_model = p_idx
                val_model = (p_idx - 1) % k
                for i in range(k):
                    if i != test_model and i != val_model:
                        models_to_load.add(i)

        sorted_model_indices = sorted(list(models_to_load))
        
        models = []
        for fold_idx in sorted_model_indices:
            fold_dir = self.checkpoint_path / f"fold_{fold_idx}"
            checkpoints = list(fold_dir.glob("*.ckpt"))
            if not checkpoints:
                 checkpoints = list(fold_dir.rglob("*.ckpt"))
            
            if not checkpoints:
                print(f"Warning: No checkpoint found for fold {fold_idx} in {fold_dir}")
                continue
                
            ckpt_path = self._select_best_checkpoint(checkpoints)
            models.append(self._load_model(f"fold_{fold_idx}", ckpt_path))
            
        return models

    def _load_model(self, key: str, path: Path) -> nn.Module:
        if key in self.cache:
            return self.cache[key]
        
        if path.is_dir():
             checkpoints = list(path.rglob("*.ckpt"))
             if not checkpoints:
                 raise FileNotFoundError(f"No checkpoint found in {path}")
             path = self._select_best_checkpoint(checkpoints)

        print(f"Loading model from {path} for {key}...")
        module = instantiate(
            model_config=self.model_config,
            data_config=self.data_config,
            train_config=self.train_config,
        )
        checkpoint = torch.load(path, map_location=self.device)
        module.load_state_dict(checkpoint["state_dict"])
        module.to(self.device)
        module.eval()
        
        self.cache[key] = module.model
        return module.model
