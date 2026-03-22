
import pybigtools
import pytest
import torch

from cerberus.config import DataConfig, GenomeConfig
from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from cerberus.mask import BedMaskExtractor


@pytest.fixture
def dummy_bed(tmp_path):
    p = tmp_path / "test.bed"
    with open(p, "w") as f:
        f.write("chr1\t100\t200\n")
        f.write("chr1\t300\t400\n")
    return p

@pytest.fixture
def dummy_bigbed(tmp_path):
    path = tmp_path / "test.bb"
    chroms = {"chr1": 1000}
    entries = [("chr1", 100, 200, "."), ("chr1", 300, 400, ".")]
    bb = pybigtools.open(str(path), "w") # type: ignore
    bb.write(chroms, entries)
    return path

@pytest.fixture
def dummy_bigwig(tmp_path):
    path = tmp_path / "test.bw"
    # Create a simple bigwig
    pybigtools.open(str(path), "w") # type: ignore
    # Need to check pybigtools API for bigwig creation, but let's assume this works or skip if complex
    # Usually easier to just mock or assume it exists. 
    # For now, I'll trust existing tests know how to make bigwigs or just skip the BW part if it's hard.
    # Actually, pybigtools.open(w) creates bigbed by default? No, extension matters?
    # I'll stick to bed/bigbed for this test file as that's the focus.
    return path

def test_bed_mask_extractor(dummy_bed):
    # This class doesn't exist yet, so this test defines the requirement
    extractor = BedMaskExtractor({"test": dummy_bed})
    
    # Overlap
    val = extractor.extract(Interval("chr1", 150, 160, "+"))
    assert val.shape == (1, 10)
    assert torch.all(val == 1.0)
    
    # No overlap
    val = extractor.extract(Interval("chr1", 250, 260, "+"))
    assert torch.all(val == 0.0)
    
    # Partial overlap
    # Bed: 100-200. Query: 195-205. Overlap: 195-200 (first 5 bases)
    val = extractor.extract(Interval("chr1", 195, 205, "+"))
    assert torch.all(val[0, :5] == 1.0)
    assert torch.all(val[0, 5:] == 0.0)

def test_dataset_bed_support(dummy_bed, tmp_path):
    # Test that CerberusDataset can load BED files via config
    # This requires the dataset to automatically choose BedMaskExtractor

    fasta_path = tmp_path / "hg38.fa"

    # Create dummy fasta
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "N"*1000)

    genome_config = GenomeConfig.model_construct(
        name="hg38",
        fasta_path=fasta_path,
        exclude_intervals={},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 1000},
        fold_type="chrom_partition",
        fold_args={"k": 2},
    )

    data_config = DataConfig.model_construct(
        inputs={"my_bed": dummy_bed},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="onehot",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=False,
    )

    # This should succeed and use BedMaskExtractor internally
    ds = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        is_train=True
    )
    
    # Verify the extractor type (white-box testing)
    assert hasattr(ds, "input_signal_extractor")
    # We might wrap it in UniversalExtractor, so check underlying or behavior
    
    # Fetch data
    # Interval 100-200 is in BED.
    item = ds.get_interval(Interval("chr1", 100, 200, "+"))
    inputs = item["inputs"]
    # inputs shape: (C, L) = (1, 100). Should be all 1s.
    assert inputs.shape == (1, 100)
    assert torch.all(inputs == 1.0)

def test_dataset_bigbed_support(dummy_bigbed, tmp_path):
    # Test that CerberusDataset can load BigBed files via config

    fasta_path = tmp_path / "hg38.fa"

    # Create dummy fasta
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "N"*1000)

    genome_config = GenomeConfig.model_construct(
        name="hg38",
        fasta_path=fasta_path,
        exclude_intervals={},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 1000},
        fold_type="chrom_partition",
        fold_args={"k": 2},
    )

    data_config = DataConfig.model_construct(
        inputs={"my_bb": dummy_bigbed},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="onehot",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=False,
    )

    ds = CerberusDataset(
        genome_config=genome_config,
        data_config=data_config,
        is_train=True
    )
    
    item = ds.get_interval(Interval("chr1", 100, 200, "+"))
    inputs = item["inputs"]
    assert torch.all(inputs == 1.0)
