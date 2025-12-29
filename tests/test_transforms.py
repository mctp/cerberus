import pytest
import torch
from cerberus.transform import Jitter, TargetCrop, ReverseComplement, Log1p, Sqrt, Arcsinh, Bin, Compose
from cerberus.interval import Interval

@pytest.fixture
def dummy_interval():
    return Interval("chr1", 0, 1000)

def test_jitter(dummy_interval):
    # Input: (C, 20)
    inputs = torch.randn(4, 20)
    targets = torch.randn(1, 20)
    
    # Jitter to 10
    jitter = Jitter(input_len=10, max_jitter=5)
    
    out_in, out_target, out_interval = jitter(inputs, targets, dummy_interval)
    
    assert out_in.shape == (4, 10)
    assert out_target.shape == (1, 10)
    
    # Check interval update
    # Original start 0. New start should be between 0 and 10 (20-10=10 slack, max_jitter=5 around center 5 -> 0..10)
    assert 0 <= out_interval.start <= 10
    assert out_interval.end == out_interval.start + 10

def test_jitter_max_jitter_constraint(dummy_interval):
    # Input len 100. Output 50. Slack 50.
    # Center of slack is 25.
    inputs = torch.randn(1, 100)
    targets = inputs.clone()
    
    # max_jitter = 5. Range [25-5, 25+5] = [20, 30]
    jitter = Jitter(input_len=50, max_jitter=5)
    
    for _ in range(50):
        # Reset interval
        interval = Interval("chr1", 1000, 1100)
        out, _, out_int = jitter(inputs, targets, interval)
        first_val = out[0, 0]
        # Find start index
        matches = (inputs[0] == first_val).nonzero()
        assert len(matches) == 1
        start_idx = matches.item()
        
        assert 20 <= start_idx <= 30
        
        # Check interval
        assert out_int.start == 1000 + start_idx
        assert out_int.end == out_int.start + 50

def test_target_crop(dummy_interval):
    # Input: (C, 20)
    inputs = torch.arange(20).unsqueeze(0).float() # 0..19
    targets = inputs.clone()
    
    crop = TargetCrop(output_len=10)
    
    out_in, out_target, out_int = crop(inputs, targets, dummy_interval)
    
    # Inputs should NOT be cropped
    assert out_in.shape == (1, 20)
    
    # Targets should be cropped
    assert out_target.shape == (1, 10)
    
    # Center of 20 is 10. Start = (20-10)//2 = 5. End = 15.
    expected = torch.arange(5, 15).unsqueeze(0).float()
    assert torch.equal(out_target, expected)
    
    # Interval should NOT change for TargetCrop (metadata tracks Input)
    assert out_int.start == 0
    assert out_int.end == 1000

def test_reverse_complement_dna_only(dummy_interval):
    # Input: (4, 10) DNA only
    # A=1, C=0, G=0, T=0 at pos 0
    inputs = torch.zeros(4, 10)
    inputs[0, 0] = 1 # A at start
    
    rc = ReverseComplement(probability=1.0) # Always apply
    
    out_in, _, out_int = rc(inputs, inputs, dummy_interval) 
    
    # Check T at end
    assert out_in[3, 9] == 1
    assert out_in[0, 0] == 0
    
    # Check strand flip
    assert out_int.strand == "-"
    
    # Check length flip
    # If we had sequence 0, 1, 2...
    inputs = torch.arange(10).unsqueeze(0).expand(4, 10).float()
    out_in, _, _ = rc(inputs, inputs, dummy_interval)
    assert out_in[0, 0] == 9 # Reversed

def test_reverse_complement_with_signals(dummy_interval):
    # Input: (5, 10). 4 DNA + 1 Signal.
    inputs = torch.zeros(5, 10)
    inputs[0, 0] = 1 # A at start
    inputs[4, 0] = 100 # Signal at start
    
    rc = ReverseComplement(probability=1.0, dna_channels=slice(0, 4))
    
    out_in, _, out_int = rc(inputs, inputs, dummy_interval)
    
    # DNA Check: T at end
    assert out_in[3, 9] == 1
    
    # Signal Check: 100 at end (Reversed but NOT complemented)
    assert out_in[4, 9] == 100
    assert out_in[4, 0] == 0
    
    assert out_int.strand == "-"

def test_log1p(dummy_interval):
    targets = torch.tensor([0.0, 1.0, 10.0]).unsqueeze(0)
    inputs = torch.randn(4, 3)
    
    t = Log1p(apply_to='targets')
    _, out_target, out_int = t(inputs, targets, dummy_interval)
    
    expected = torch.log1p(targets)
    assert torch.allclose(out_target, expected)
    assert out_int.strand == "+"

def test_log1p_negative_crash(dummy_interval):
    targets = torch.tensor([-1.0]).unsqueeze(0)
    inputs = torch.randn(4, 3)
    
    # safe_check=False (default) -> No crash
    t = Log1p(apply_to='targets', safe_check=False)
    _, out, _ = t(inputs, targets, dummy_interval)
    assert torch.isinf(out).any() # log1p(-1) = -inf
    
    # safe_check=True -> Crash
    t = Log1p(apply_to='targets', safe_check=True)
    with pytest.raises(ValueError, match="Log1p target contains negative values"):
        t(inputs, targets, dummy_interval)

def test_sqrt(dummy_interval):
    targets = torch.tensor([0.0, 4.0, 100.0]).unsqueeze(0)
    inputs = torch.randn(4, 3)
    
    t = Sqrt(apply_to='targets')
    _, out_target, _ = t(inputs, targets, dummy_interval)
    
    expected = torch.sqrt(targets)
    assert torch.allclose(out_target, expected)

def test_sqrt_negative_crash(dummy_interval):
    targets = torch.tensor([-1.0]).unsqueeze(0)
    inputs = torch.randn(4, 3)
    
    t = Sqrt(apply_to='targets', safe_check=True)
    with pytest.raises(ValueError, match="Sqrt target contains negative values"):
        t(inputs, targets, dummy_interval)

def test_arcsinh(dummy_interval):
    targets = torch.tensor([0.0, 1.0, 100.0]).unsqueeze(0)
    inputs = torch.randn(4, 3)
    
    t = Arcsinh(apply_to='targets')
    _, out_target, _ = t(inputs, targets, dummy_interval)
    
    expected = torch.arcsinh(targets)
    assert torch.allclose(out_target, expected)

def test_bin_max(dummy_interval):
    targets = torch.arange(8).float().unsqueeze(0) # (1, 8)
    inputs = torch.randn(4, 8)
    
    t = Bin(bin_size=2, method='max')
    _, out_target, _ = t(inputs, targets, dummy_interval)
    
    assert out_target.shape == (1, 4)
    expected = torch.tensor([[1., 3., 5., 7.]])
    assert torch.equal(out_target, expected)

def test_bin_sum(dummy_interval):
    targets = torch.tensor([[1., 1., 2., 2.]])
    inputs = torch.randn(4, 4)
    
    t = Bin(bin_size=2, method='sum')
    _, out_target, _ = t(inputs, targets, dummy_interval)
    
    assert out_target.shape == (1, 2)
    expected = torch.tensor([[2., 4.]])
    assert torch.equal(out_target, expected)

def test_compose(dummy_interval):
    # Chain: Jitter -> TargetCrop
    inputs = torch.randn(4, 100)
    targets = torch.randn(1, 100)
    
    transforms = [
        Jitter(input_len=50),
        TargetCrop(output_len=20)
    ]
    
    compose = Compose(transforms)
    
    out_in, out_target, out_int = compose(inputs, targets, dummy_interval)
    
    assert out_in.shape == (4, 50) 
    assert out_target.shape == (1, 20)
    
    # Check Jitter updated interval
    assert out_int.end - out_int.start == 50
