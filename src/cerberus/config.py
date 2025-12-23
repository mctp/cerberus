from typing import TypedDict, Any
from pathlib import Path

# --- Configuration Schemas ---


class GenomeConfig(TypedDict):
    """
    Configuration for the genome assembly.

    Attributes:
        name: Name of the genome assembly (e.g., 'hg38').
        fasta_path: Path to the FASTA file.
        exclude_intervals: Dictionary mapping names to BED files of regions to exclude.
        allowed_chroms: List of chromosome names to include.
        chrom_sizes: Dictionary mapping chromosome names to their lengths.
        fold_type: Strategy for creating folds (e.g., 'chrom_partition').
        fold_args: Arguments for the folding strategy (e.g., {'k': 5}).
    """

    name: str
    fasta_path: Path
    exclude_intervals: dict[str, Path]
    allowed_chroms: list[str]
    chrom_sizes: dict[str, int]
    fold_type: str
    fold_args: dict[str, Any]


class SamplerConfig(TypedDict):
    """
    Configuration for data samplers.

    Attributes:
        sampler_type: Type of sampler to use ('interval', 'sliding_window', 'multi').
        padded_size: Length of the intervals yielded by the sampler (after padding/centering).
        sampler_args: Dictionary of arguments specific to the sampler type.

    Sampler Args Requirements by Type:
    - 'interval':
        - intervals_path: Path to BED/narrowPeak file containing regions of interest.
    - 'sliding_window':
        - stride: Step size for generating sliding windows across the genome.
    - 'multi':
        - samplers: List of sub-sampler configurations. Each item must be a dict with:
            - type: Sampler type string ('interval', 'sliding_window').
            - args: Dictionary of arguments for that sub-sampler.
            - scaling: Float scaling factor for sampling frequency (ratio of total samples).
    """

    sampler_type: str
    padded_size: int
    sampler_args: dict[str, Any]


class DataConfig(TypedDict):
    """
    Configuration for input/output data handling.

    Attributes:
        inputs: Dictionary mapping input channel names to bigWig file paths.
        targets: Dictionary mapping target channel names to bigWig file paths.
        input_len: Length of the input sequence window.
        output_len: Length of the output signal window.
        max_jitter: Maximum random shift applied to the interval center during training.
        bin_size: Size of bins for signal aggregation (1 means raw signal).
        encoding: DNA encoding strategy (e.g., 'ACGT').
        log_transform: Whether to apply log(x+1) transformation to signal.
        reverse_complement: Whether to apply reverse complement augmentation.
        in_memory: Whether to load the data (genome/signals) into memory.
    """

    inputs: dict[str, Path]
    targets: dict[str, Path]
    input_len: int
    output_len: int
    max_jitter: int
    bin_size: int
    encoding: str
    log_transform: bool
    reverse_complement: bool
    in_memory: bool


# --- Validation Logic ---


def _validate_path(
    path: str | Path, description: str, check_exists: bool = True
) -> Path:
    """Validates that a path exists (optional) and returns it as a Path object."""
    p = Path(path)
    if check_exists and not p.exists():
        raise FileNotFoundError(f"{description} not found at: {p}")
    return p


def _validate_file_dict(data: dict, description: str) -> dict[str, Path]:
    """Validates a dictionary of name -> filepath mappings."""
    if not isinstance(data, dict):
        raise TypeError(f"{description} must be a dictionary")

    validated = {}
    for k, v in data.items():
        if not isinstance(k, str):
            raise TypeError(f"{description} keys must be strings")
        validated[k] = _validate_path(v, f"{description} file '{k}'")
    return validated


def validate_genome_config(config: dict | GenomeConfig) -> GenomeConfig:
    """
    Validates the genome configuration and returns a GenomeConfig object.
    
    Checks for required keys, correct types, and consistency between allowed_chroms and chrom_sizes.
    
    Args:
        config: Dictionary containing genome configuration.
        
    Returns:
        GenomeConfig: Validated and typed configuration object.
        
    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are inconsistent.
        FileNotFoundError: If specified files do not exist.
    """
    if not isinstance(config, dict):
        raise TypeError("Genome config must be a dictionary")

    # Check for required fields
    required_keys = {
        "name",
        "fasta_path",
        "allowed_chroms",
        "chrom_sizes",
        "exclude_intervals",
        "fold_type",
        "fold_args",
    }
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Genome config missing required keys: {missing}")
    
    if not isinstance(config["fold_type"], str):
        raise TypeError("fold_type must be a string")
    
    if not isinstance(config["fold_args"], dict):
        raise TypeError("fold_args must be a dictionary")

    path_val = config["fasta_path"]
    if not isinstance(path_val, (str, Path)):
        raise TypeError("fasta_path must be a string or Path")

    p = _validate_path(path_val, "Genome file")

    exclude_intervals = _validate_file_dict(
        config["exclude_intervals"], "exclude_intervals"
    )

    if not isinstance(config["allowed_chroms"], list):
        raise TypeError("allowed_chroms must be a list of strings")
    if not all(isinstance(c, str) for c in config["allowed_chroms"]):
        raise TypeError("allowed_chroms must contain only strings")

    if not isinstance(config["chrom_sizes"], dict):
        raise TypeError("chrom_sizes must be a dictionary")
    # Validate chrom_sizes contents
    for k, v in config["chrom_sizes"].items():
        if not isinstance(k, str):
            raise TypeError(f"chrom_sizes keys must be strings, got {type(k)}")
        if not isinstance(v, int):
            raise TypeError(f"chrom_sizes values must be integers, got {type(v)}")

    # Ensure chrom_sizes only contains allowed_chroms
    allowed_set = set(config["allowed_chroms"])
    filtered_sizes = {
        k: v for k, v in config["chrom_sizes"].items() if k in allowed_set
    }

    # Check if all allowed chroms have sizes
    if len(filtered_sizes) != len(allowed_set):
        missing = allowed_set - set(filtered_sizes.keys())
        raise ValueError(f"chrom_sizes missing entries for allowed_chroms: {missing}")

    # Return valid config (cast/ensure types)
    return {
        "name": str(config["name"]),
        "fasta_path": p,
        "exclude_intervals": exclude_intervals,
        "allowed_chroms": config["allowed_chroms"],
        "chrom_sizes": filtered_sizes,
        "fold_type": config["fold_type"],
        "fold_args": config["fold_args"],
    }


def validate_data_config(config: dict | DataConfig) -> DataConfig:
    """
    Validates the data configuration dictionary.
    
    Checks for required keys and correct types for data parameters.
    
    Args:
        config: Dictionary containing data configuration.
        
    Returns:
        DataConfig: Validated and typed configuration object.
        
    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid (e.g., negative lengths).
        FileNotFoundError: If specified files do not exist.
    """
    if not isinstance(config, dict):
        raise TypeError("Data config must be a dictionary")

    required_keys = {
        "inputs",
        "targets",
        "input_len",
        "output_len",
        "bin_size",
        "encoding",
        "max_jitter",
        "log_transform",
        "reverse_complement",
        "in_memory",
    }
    
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Data config missing required keys: {missing}")

    inputs = _validate_file_dict(config["inputs"], "inputs")
    targets = _validate_file_dict(config["targets"], "targets")

    # Type and Value checks
    if not isinstance(config["input_len"], int) or config["input_len"] <= 0:
        raise ValueError("input_len must be a positive integer")

    if not isinstance(config["output_len"], int) or config["output_len"] <= 0:
        raise ValueError("output_len must be a positive integer")

    if not isinstance(config["max_jitter"], int) or config["max_jitter"] < 0:
        raise ValueError("max_jitter must be a non-negative integer")

    if not isinstance(config["bin_size"], int) or config["bin_size"] <= 0:
        raise ValueError("bin_size must be a positive integer")

    if not isinstance(config["encoding"], str):
        raise TypeError("encoding must be a string")

    if not isinstance(config["log_transform"], bool):
        raise TypeError("log_transform must be a boolean")

    if not isinstance(config["reverse_complement"], bool):
        raise TypeError("reverse_complement must be a boolean")

    if not isinstance(config["in_memory"], bool):
        raise TypeError("in_memory must be a boolean")

    return {
        "inputs": inputs,
        "targets": targets,
        "input_len": config["input_len"],
        "output_len": config["output_len"],
        "max_jitter": config["max_jitter"],
        "bin_size": config["bin_size"],
        "encoding": config["encoding"],
        "log_transform": config["log_transform"],
        "reverse_complement": config["reverse_complement"],
        "in_memory": config["in_memory"],
    }


def validate_sampler_config(config: dict | SamplerConfig) -> SamplerConfig:
    """
    Validates the sampler configuration dictionary.
    
    Checks for required keys and correct types. Performs specific validation based on `sampler_type`.
    
    Args:
        config: Dictionary containing sampler configuration.
        
    Returns:
        SamplerConfig: Validated and typed configuration object.
        
    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("Sampler config must be a dictionary")

    required_keys = {
        "sampler_type",
        "padded_size",
        "sampler_args",
    }
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Sampler config missing required keys: {missing}")

    if not isinstance(config["sampler_type"], str):
        raise TypeError("sampler_type must be a string")

    if not isinstance(config["padded_size"], int) or config["padded_size"] <= 0:
        raise ValueError("padded_size must be a positive integer")

    if not isinstance(config["sampler_args"], dict):
        raise TypeError("sampler_args must be a dictionary")

    # Specialized validation based on sampler_type
    if config["sampler_type"] == "multi":
        if "samplers" not in config["sampler_args"]:
            raise ValueError("MultiSampler requires 'samplers' list in sampler_args")
        if not isinstance(config["sampler_args"]["samplers"], list):
            raise TypeError("MultiSampler 'samplers' must be a list")

        for sub in config["sampler_args"]["samplers"]:
            if not isinstance(sub, dict):
                raise TypeError("MultiSampler sub-sampler config must be a dictionary")
            required_sub = {"type", "args", "scaling"}
            if not all(k in sub for k in required_sub):
                missing = required_sub - sub.keys()
                raise ValueError(f"MultiSampler sub-sampler missing keys: {missing}")

    elif config["sampler_type"] == "interval":
        required_args = {"intervals_path"}
        if not all(k in config["sampler_args"] for k in required_args):
            missing = required_args - config["sampler_args"].keys()
            raise ValueError(f"IntervalSampler args missing required keys: {missing}")

    elif config["sampler_type"] == "sliding_window":
        required_args = {"stride"}
        if not all(k in config["sampler_args"] for k in required_args):
            missing = required_args - config["sampler_args"].keys()
            raise ValueError(
                f"SlidingWindowSampler args missing required keys: {missing}"
            )

    return {
        "sampler_type": config["sampler_type"],
        "padded_size": config["padded_size"],
        "sampler_args": config["sampler_args"],
    }
