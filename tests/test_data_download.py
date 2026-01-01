
def test_mdapca2b_ar_dataset_download(mdapca2b_ar_dataset):
    """
    Test that the mdapca2b-ar dataset is downloaded and paths are correct.
    """
    paths = mdapca2b_ar_dataset
    
    assert "bigwig" in paths
    assert "narrowPeak" in paths
    
    assert paths["bigwig"].exists()
    assert paths["narrowPeak"].exists()
    
    # Check if files are not empty
    assert paths["bigwig"].stat().st_size > 0
    assert paths["narrowPeak"].stat().st_size > 0
    
    print(f"Verified mdapca2b-ar dataset files at: {paths}")
