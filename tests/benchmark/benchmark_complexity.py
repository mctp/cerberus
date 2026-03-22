import time
import random
import numpy as np
from cerberus.complexity import calculate_gc_content, calculate_dust_score

def generate_random_dna(length):
    return "".join(random.choices("ACGT", k=length))

def benchmark():
    N = 5000
    L = 1000
    
    print(f"Generating {N} sequences of length {L}...")
    seqs_fixed = [generate_random_dna(L) for _ in range(N)]
    
    print(f"Generating {N} sequences of variable length (approx {L})...")
    seqs_var = [generate_random_dna(random.randint(L-100, L+100)) for _ in range(N)]
    
    print("\n--- GC Content ---")
    
    start = time.time()
    _ = [calculate_gc_content(s) for s in seqs_fixed]
    print(f"Loop (Fixed): {time.time() - start:.4f}s")
    
    start = time.time()
    _ = [calculate_gc_content(s) for s in seqs_fixed]
    print(f"Batch (Fixed): {time.time() - start:.4f}s")

    start = time.time()
    _ = [calculate_gc_content(s) for s in seqs_var]
    print(f"Batch (Var):   {time.time() - start:.4f}s")

    print("\n--- DUST Score (k=3) ---")

    start = time.time()
    _ = [calculate_dust_score(s, k=3) for s in seqs_fixed]
    print(f"Loop (Fixed): {time.time() - start:.4f}s")

    start = time.time()
    _ = [calculate_dust_score(s, k=3) for s in seqs_fixed]
    print(f"Batch (Fixed): {time.time() - start:.4f}s")

    start = time.time()
    _ = [calculate_dust_score(s, k=3) for s in seqs_var]
    print(f"Batch (Var):   {time.time() - start:.4f}s")

if __name__ == "__main__":
    benchmark()
