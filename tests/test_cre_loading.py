
from cerberus.mask import BigBedMaskExtractor as MaskExtractor
from cerberus.interval import Interval

def test_cre_loading_real(human_genome):
    """
    Tests loading the real ENCODE cCREs BigBed file downloaded by the fixture.
    """
    cre_path = human_genome["encode_cre"]
    assert cre_path.exists()
    
    extractor = MaskExtractor({"cre": cre_path})
    
    # Positive region: overlap expected
    # chr8:127,777,977-127,782,457
    start_pos = 127777977
    end_pos = 127782457
    interval_pos = Interval("chr8", start_pos, end_pos, "+")
    
    mask_pos = extractor.extract(interval_pos)
    assert mask_pos.shape == (1, end_pos - start_pos)
    # Check if there is ANY overlap (sum > 0)
    assert mask_pos.sum() > 0, "Expected overlap in positive region"

    # Negative region: no overlap expected
    # chr8:127,789,157-127,793,637
    start_neg = 127789157
    end_neg = 127793637
    interval_neg = Interval("chr8", start_neg, end_neg, "+")
    
    mask_neg = extractor.extract(interval_neg)
    assert mask_neg.shape == (1, end_neg - start_neg)
    assert (mask_neg == 0.0).all(), "Expected no overlap in negative region"
