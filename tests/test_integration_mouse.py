

def test_mouse_genome_fixture(mouse_genome):
    """
    Test that the mouse_genome fixture returns the expected dictionary
    and that the files exist.
    """
    assert "fasta" in mouse_genome
    assert mouse_genome["fasta"].exists()
    assert mouse_genome["fasta"].name == "mm10.fa"
    
    # Check other files
    assert "blacklist" in mouse_genome
    assert mouse_genome["blacklist"].exists()
    
    assert "gaps" in mouse_genome
    assert mouse_genome["gaps"].exists()
    
    assert "encode_cre" in mouse_genome
    assert mouse_genome["encode_cre"].exists()
