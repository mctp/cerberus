import torch
from torch.utils.data import DataLoader

from cerberus.config import DataConfig, GenomeConfig, SamplerConfig
from cerberus.dataset import CerberusDataset
from tests.mock_utils import (
    GaussianSignalGenerator,
    MockSampler,
    MockSequenceExtractor,
    MockSignalExtractor,
    insert_ggaa_motifs,
)


def test_mock_dataset_end_to_end(mock_files):
    # 1. Configs
    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path=mock_files["fasta"],
        exclude_intervals={"blacklist": mock_files["exclude"]},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 10000},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": None, "val_fold": None},
    )

    data_config = DataConfig.model_construct(
        inputs={"input1": mock_files["input"]},
        targets={"target1": mock_files["target"]},
        input_len=200,
        output_len=200,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=200,
        sampler_args={"intervals_path": mock_files["exclude"]},
    )

    # 2. Components
    sampler = MockSampler(
        num_samples=100,
        chroms=["chr1"],
        chrom_size=10000,
        interval_length=200,
        seed=42
    )

    seq_extractor = MockSequenceExtractor(
        fasta_path=None, # Use synthetic
        motif_inserters=[insert_ggaa_motifs]
    )

    # Target extractor generates Gaussian peaks
    target_extractor = MockSignalExtractor(
        sequence_extractor=seq_extractor,
        signal_generator=GaussianSignalGenerator(sigma=2.0, base_height=10.0)
    )

    # Input extractor just zeros (or some noise)
    input_extractor = MockSignalExtractor(
        sequence_extractor=seq_extractor,
        signal_generator=None # Defaults to zeros
    )

    # 3. Dataset
    ds = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        sampler=sampler,
        sequence_extractor=seq_extractor,
        input_signal_extractor=input_extractor,
        target_signal_extractor=target_extractor,
        in_memory=False
    )

    # 4. Verification
    # Fetch a sample
    sample = ds[0]
    inputs = sample["inputs"] # (4+C, L) -> (4+1, 200) = (5, 200)
    targets = sample["targets"] # (1, 200)

    assert inputs.shape == (5, 200)
    assert targets.shape == (1, 200)

    # Check correlation between GGAA and peaks
    seq = inputs[:4, :]
    signal = targets[0, :]

    # Manually find GGAA in this sequence

    # Simple scan
    for i in range(200 - 4):
        if (seq[2, i] == 1 and seq[2, i+1] == 1 and
            seq[0, i+2] == 1 and seq[0, i+3] == 1):
            center_idx = i + 1
            peak_val = signal[center_idx:center_idx+2].mean()
            assert peak_val > 1.0, f"Expected peak at {center_idx} for GGAA"

    # With 100 samples and prob 0.5, we should see some peaks eventually
    total_peaks = 0
    for i in range(10):
        sample = ds[i]
        signal = sample["targets"][0]
        if signal.max() > 1.0:
            total_peaks += 1

    assert total_peaks > 0, "No peaks generated across samples"

def test_mock_sampler_splitting():
    sampler = MockSampler(num_samples=100)
    train, val, test = sampler.split_folds()

    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # Indices should be disjoint
    train_indices = set(str(i) for i in train._intervals)
    val_indices = set(str(i) for i in val._intervals)
    test_indices = set(str(i) for i in test._intervals)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

def create_mock_dataset(mock_files, num_samples=100) -> CerberusDataset:
    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path=mock_files["fasta"],
        exclude_intervals={"blacklist": mock_files["exclude"]},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 10000},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": None, "val_fold": None},
    )

    data_config = DataConfig.model_construct(
        inputs={"input1": mock_files["input"]},
        targets={"target1": mock_files["target"]},
        input_len=200,
        output_len=200,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=200,
        sampler_args={"intervals_path": mock_files["exclude"]},
    )

    sampler = MockSampler(num_samples=num_samples, chrom_size=10000, interval_length=200)

    seq_extractor = MockSequenceExtractor(
        fasta_path=None,
        motif_inserters=[insert_ggaa_motifs]
    )

    target_extractor = MockSignalExtractor(
        sequence_extractor=seq_extractor,
        signal_generator=GaussianSignalGenerator()
    )

    input_extractor = MockSignalExtractor(
        sequence_extractor=seq_extractor,
        signal_generator=None
    )

    ds = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        sampler_config=sampler_config,
        sampler=sampler,
        sequence_extractor=seq_extractor,
        input_signal_extractor=input_extractor,
        target_signal_extractor=target_extractor,
        in_memory=False
    )
    return ds

def test_dataset_api_conformance(mock_files):
    ds = create_mock_dataset(mock_files, num_samples=50)

    # 1. Length
    assert len(ds) == 50

    # 2. Item structure
    sample = ds[0]
    assert isinstance(sample, dict)
    assert "inputs" in sample
    assert "targets" in sample
    assert "intervals" in sample
    assert isinstance(sample["inputs"], torch.Tensor)
    assert isinstance(sample["targets"], torch.Tensor)
    assert isinstance(sample["intervals"], str)

    # 3. Split folds
    train_ds, val_ds, test_ds = ds.split_folds()
    assert isinstance(train_ds, CerberusDataset)
    assert isinstance(val_ds, CerberusDataset)
    assert isinstance(test_ds, CerberusDataset)

    # Check sizes match the sampler's split logic (80/10/10)
    assert len(train_ds) == 40
    assert len(val_ds) == 5
    assert len(test_ds) == 5

    # 4. Resample
    ds.resample(seed=123)

def test_dataset_dataloader_compatibility(mock_files):
    ds = create_mock_dataset(mock_files, num_samples=20)

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    batch_count = 0
    for batch in dl:
        batch_count += 1
        inputs = batch["inputs"]
        targets = batch["targets"]
        intervals = batch["intervals"]

        assert inputs.shape[0] == 4
        assert targets.shape[0] == 4
        assert len(intervals) == 4

    assert batch_count == 5
