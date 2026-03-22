import torch

from cerberus.interval import Interval
from cerberus.sequence import encode_dna
from cerberus.transform import Compose, ReverseComplement


def test_reverse_complement_end_to_end():
    """
    Test ReverseComplement transform end-to-end:
    - Encoding of sequence (via encode_dna)
    - Reversal of signal
    - Complementation + Reversal of sequence
    - Using Compose
    """
    # 1. Setup Transform
    # Enforce RC with prob=1.0
    rc = ReverseComplement(probability=1.0)
    transform = Compose([rc])

    # 2. Setup Inputs (Sequence)
    # Sequence: 5'-AAAC-3'
    # We use a non-palindromic sequence to clearly distinguish RC
    seq_str = "AAAC"
    
    # Encode using the actual encoding function
    # Shape (4, 4)
    inputs = encode_dna(seq_str, encoding="ACGT") 
    
    # 3. Setup Targets (Signal)
    # Signal: 2 Channels, length 4
    # We use increasing values to clearly verify reversal
    targets = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]
    ])

    # 4. Apply Transform
    dummy_interval = Interval("chr1", 0, 4)
    inputs_rc, targets_rc, out_interval = transform(inputs, targets, dummy_interval)

    # 5. Verify Inputs (Sequence)
    # Expected RC of AAAC is GTTT
    # Logic: 
    # Original: 5'-AAAC-3'
    # Reverse:  3'-CAAA-5'
    # Complement: 5'-GTTT-3'
    
    expected_seq_str = "GTTT"
    expected_inputs = encode_dna(expected_seq_str, encoding="ACGT")
    
    # Debug info if fails
    print(f"\nOriginal Seq: {seq_str}")
    print(f"Original Encoded:\n{inputs}")
    print(f"Expected Seq: {expected_seq_str}")
    print(f"Expected Encoded:\n{expected_inputs}")
    print(f"Actual Transformed:\n{inputs_rc}")

    assert torch.equal(inputs_rc, expected_inputs), \
        f"Sequence mismatch. Expected {expected_seq_str} (encoded), got transformed tensor."

    # 6. Verify Targets (Signal)
    # Expected: Each channel reversed independently
    expected_targets = torch.tensor([
        [4.0, 3.0, 2.0, 1.0],
        [8.0, 7.0, 6.0, 5.0]
    ])
    
    print(f"Original Signal:\n{targets}")
    print(f"Expected Signal:\n{expected_targets}")
    print(f"Actual Signal:\n{targets_rc}")
    
    assert torch.equal(targets_rc, expected_targets), \
        "Signal mismatch. Expected reversed signals."
    
    # Verify interval strand flip
    assert out_interval.strand == "-"

def test_reverse_complement_identity():
    """
    Test that probability=0.0 leaves data unchanged.
    """
    rc = ReverseComplement(probability=0.0)
    transform = Compose([rc])

    seq_str = "AAAC"
    inputs = encode_dna(seq_str, encoding="ACGT")
    targets = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    dummy_interval = Interval("chr1", 0, 4)
    inputs_out, targets_out, out_interval = transform(inputs, targets, dummy_interval)

    assert torch.equal(inputs_out, inputs)
    assert torch.equal(targets_out, targets)
    assert out_interval.strand == "+"
