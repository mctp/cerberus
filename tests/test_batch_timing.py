import time
from typing import cast
from cerberus.genome import create_genome_config
from cerberus import CerberusDataModule
from cerberus.config import DataConfig, SamplerConfig

def test_batch_generation_timing(human_genome, mdapca2b_ar_dataset):
    """
    Measure and print timing for generating batches of sizes 8, 16, 32, 64, 128.
    Compares Disk-based vs In-Memory loading.
    Reuses DataModule to avoid reloading data for different batch sizes.
    """
    
    # Paths from fixtures
    fasta_path = human_genome["fasta"]
    blacklist_path = human_genome["blacklist"]
    mappability_path = human_genome["mappability"]
    
    peaks_path = mdapca2b_ar_dataset["narrowPeak"]
    signal_path = mdapca2b_ar_dataset["bigwig"]

    # 2. Genome Configuration
    genome_config = create_genome_config(
        name="hg38",
        fasta_path=fasta_path,
        species="human",
        exclude_intervals={"blacklist": blacklist_path},
        fold_type="chrom_partition",
        fold_args={"k": 5}
    )

    # 4. Sampler Configuration
    sampler_config = cast(SamplerConfig, {
        "sampler_type": "interval",
        "padded_size": 2304,
        "sampler_args": {
            "intervals_path": peaks_path
        }
    })

    batch_sizes = [8, 16, 32, 64, 128]
    num_batches_to_measure = 10 
    
    # Test both Disk and In-Memory
    loading_modes = [False, True] # False = Disk, True = Memory

    print("\n" + "="*100)
    print("Batch Generation Timing Test")
    print("="*100)

    for in_memory in loading_modes:
        mode_name = "In-Memory" if in_memory else "Disk-Based"
        print(f"\nMode: {mode_name}")
        print(f"{'Batch Size':<12} | {'Preload Time (s)':<18} | {'Avg Time/Batch (s)':<20} | {'Throughput (ex/s)':<20} | {'Est. Epoch Time':<20}")
        print("-" * 100)

        # 3. Data Configuration
        data_config = cast(DataConfig, {
            "inputs": {"mappability": mappability_path},
            "targets": {"AR": signal_path},
            "input_len": 2048,
            "output_len": 1024,
            "output_bin_size": 4,
            "encoding": "ACGT",
            "max_jitter": 128,
            "log_transform": True,
            "reverse_complement": True,
            "use_sequence": True,
        })

        # 5. Instantiate DataModule ONCE per mode
        data_module = CerberusDataModule(
            genome_config=genome_config,
            data_config=data_config,
            sampler_config=sampler_config,
            pin_memory=False
        )

        # 6. Setup Datasets & Measure Preload Time
        setup_start = time.time()
        data_module.setup(batch_size=batch_sizes[0], num_workers=0)
        setup_end = time.time()
        preload_time = setup_end - setup_start
        
        # Now iterate over batch sizes reusing the setup data_module
        for bs in batch_sizes:
            data_module.batch_size = bs
            
            train_loader = data_module.train_dataloader()
            total_batches = len(train_loader)
            iterator = iter(train_loader)
            
            # Warmup
            try:
                _ = next(iterator)
            except StopIteration:
                print(f"{bs:<12} | {preload_time:<18.4f} | {'Dataset too small':<20} | {'N/A':<20} | {'N/A':<20}")
                continue

            # Measure timing
            start_time = time.time()
            count = 0
            try:
                for _ in range(num_batches_to_measure):
                    _ = next(iterator)
                    count += 1
            except StopIteration:
                pass 
            end_time = time.time()
            
            if count > 0:
                total_time = end_time - start_time
                avg_time = total_time / count
                throughput = (count * bs) / total_time
                
                # Estimate epoch time
                est_epoch_seconds = avg_time * total_batches
                # Format as MM:SS or HH:MM:SS
                if est_epoch_seconds < 60:
                    est_epoch_str = f"{est_epoch_seconds:.2f} s"
                elif est_epoch_seconds < 3600:
                    est_epoch_str = f"{est_epoch_seconds/60:.2f} min"
                else:
                    est_epoch_str = f"{est_epoch_seconds/3600:.2f} h"

                print(f"{bs:<12} | {preload_time:<18.4f} | {avg_time:<20.4f} | {throughput:<20.2f} | {est_epoch_str:<20}")
            else:
                 print(f"{bs:<12} | {preload_time:<18.4f} | {'No data':<20} | {'N/A':<20} | {'N/A':<20}")

    print("="*100 + "\n")

if __name__ == "__main__":
    print("Please run with: pytest -s tests/test_batch_timing.py")
