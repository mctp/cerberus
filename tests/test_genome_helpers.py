
from cerberus.genome import create_human_genome_config, create_mouse_genome_config


def test_create_mouse_genome_config(tmp_path):
    """Test that create_mouse_genome_config creates a valid config."""
    genome_dir = tmp_path / "mm10"
    genome_dir.mkdir()
    
    fasta_path = genome_dir / "mm10.fa"
    fasta_path.touch()
    
    # Mock index file
    fai_path = genome_dir / "mm10.fa.fai"
    with open(fai_path, "w") as f:
        # standard mouse chromosomes
        for i in range(1, 20):
            f.write(f"chr{i}\t1000\n")
        f.write("chrX\t1000\n")
        f.write("chrY\t1000\n")
        f.write("chrM\t1000\n")

    # Mock blacklist and gaps
    (genome_dir / "blacklist.bed").touch()
    (genome_dir / "gaps.bed").touch()

    config = create_mouse_genome_config(genome_dir)
    
    assert config.name == "mm10"
    assert config.fasta_path == fasta_path
    assert "blacklist" in config.exclude_intervals
    assert "unmappable" in config.exclude_intervals
    
    # Check allowed chroms
    assert "chr1" in config.allowed_chroms
    assert "chr19" in config.allowed_chroms
    assert "chrX" in config.allowed_chroms
    assert "chrM" not in config.allowed_chroms # default excluded

def test_create_human_genome_config(tmp_path):
    """Test that create_human_genome_config creates a valid config."""
    genome_dir = tmp_path / "hg38"
    genome_dir.mkdir()
    
    fasta_path = genome_dir / "hg38.fa"
    fasta_path.touch()
    
    # Mock index file
    fai_path = genome_dir / "hg38.fa.fai"
    with open(fai_path, "w") as f:
        # standard human chromosomes
        for i in range(1, 23):
            f.write(f"chr{i}\t1000\n")
        f.write("chrX\t1000\n")
        f.write("chrY\t1000\n")
        f.write("chrM\t1000\n")

    # Mock blacklist and gaps
    (genome_dir / "blacklist.bed").touch()
    (genome_dir / "gaps.bed").touch()

    config = create_human_genome_config(genome_dir)
    
    assert config.name == "hg38"
    assert config.fasta_path == fasta_path
    assert "blacklist" in config.exclude_intervals
    assert "unmappable" in config.exclude_intervals
    
    # Check allowed chroms
    assert "chr1" in config.allowed_chroms
    assert "chr22" in config.allowed_chroms
    assert "chrX" in config.allowed_chroms
    assert "chrM" not in config.allowed_chroms # default excluded
