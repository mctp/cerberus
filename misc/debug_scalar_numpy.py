import numpy as np

from cerberus.complexity import calculate_gc_content

try:
    arr = np.array("ACGT")
    print(f"Scalar array shape: {arr.shape}")
    calculate_gc_content(arr)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
