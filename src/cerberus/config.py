from typing import TypedDict, Any
from pathlib import Path
import torch.nn as nn
from torchmetrics import MetricCollection

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
        output_bin_size: Size of bins for signal aggregation (1 means raw signal).
        encoding: DNA encoding strategy (e.g., 'ACGT').
        log_transform: Whether to apply log(x+1) transformation to signal.
        reverse_complement: Whether to apply reverse complement augmentation.
        use_sequence: Whether to use sequence input (default: True).
    """

    inputs: dict[str, Path]
    targets: dict[str, Path]
    input_len: int
    output_len: int
    max_jitter: int
    output_bin_size: int
    encoding: str
    log_transform: bool
    reverse_complement: bool
    use_sequence: bool


class TrainConfig(TypedDict):
    """
    Configuration for training hyperparameters.

    Attributes:
        batch_size: Batch size for training/validation.
        max_epochs: Maximum number of epochs to train.
        learning_rate: Base learning rate.
        weight_decay: Weight decay for optimizer.
        patience: Patience for early stopping.
        optimizer: Optimizer name (e.g., 'adamw', 'sgd').
        filter_bias_and_bn: Whether to exclude bias and batch norm parameters from weight decay.
    """

    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    optimizer: str
    scheduler_type: str
    scheduler_args: dict[str, Any]
    filter_bias_and_bn: bool


class PredictConfig(TypedDict):
    """
    Configuration for prediction parameters.

    Attributes:
        stride: Stride for sliding window predictions (default: output_len).
        use_folds: List of fold types to use models from ('train', 'test', 'val').
        aggregation: Aggregation mode ('model' or 'interval+model').
    """
    stride: int
    use_folds: list[str]
    aggregation: str


class ModelConfig(TypedDict):
    """
    Configuration for the model architecture.

    Attributes:
        name: Name of the model.
        input_channels: List of input channel names.
        output_channels: List of output channel names.
        output_type: Type of output ('signal' or 'decoupled').
    """

    name: str
    model_cls: type[nn.Module]
    loss_cls: type[nn.Module]
    loss_args: dict[str, Any]
    metrics_cls: type[MetricCollection]
    metrics_args: dict[str, Any]
    model_args: dict[str, Any]


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


def validate_genome_config(config: GenomeConfig) -> GenomeConfig:
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

    # Validate common fold_args keys if present
    for key in ["k", "val_fold", "test_fold"]:
        if key in config["fold_args"]:
            if not isinstance(config["fold_args"][key], int):
                raise TypeError(f"fold_args['{key}'] must be an integer")
            if config["fold_args"][key] < 0:
                raise ValueError(f"fold_args['{key}'] must be non-negative")

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


def validate_data_config(config: DataConfig) -> DataConfig:
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
        "output_bin_size",
        "encoding",
        "max_jitter",
        "log_transform",
        "reverse_complement",
    }
    
    # Optional with default
    use_sequence = config.get("use_sequence", True)
    
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

    if not isinstance(config["output_bin_size"], int) or config["output_bin_size"] <= 0:
        raise ValueError("output_bin_size must be a positive integer")

    if not isinstance(config["encoding"], str):
        raise TypeError("encoding must be a string")

    if not isinstance(config["log_transform"], bool):
        raise TypeError("log_transform must be a boolean")

    if not isinstance(config["reverse_complement"], bool):
        raise TypeError("reverse_complement must be a boolean")

    if not isinstance(use_sequence, bool):
        raise TypeError("use_sequence must be a boolean")

    return {
        "inputs": inputs,
        "targets": targets,
        "input_len": config["input_len"],
        "output_len": config["output_len"],
        "max_jitter": config["max_jitter"],
        "output_bin_size": config["output_bin_size"],
        "encoding": config["encoding"],
        "log_transform": config["log_transform"],
        "reverse_complement": config["reverse_complement"],
        "use_sequence": use_sequence,
    }


def validate_sampler_config(config: SamplerConfig) -> SamplerConfig:
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
            
    elif config["sampler_type"] == "dummy":
        pass

    return {
        "sampler_type": config["sampler_type"],
        "padded_size": config["padded_size"],
        "sampler_args": config["sampler_args"],
    }


def validate_data_and_sampler_compatibility(
    data_config: DataConfig, sampler_config: SamplerConfig
) -> None:
    """
    Validates compatibility between data and sampler configurations.

    Checks if the sampler's padded_size is sufficient to cover the input window
    plus the maximum jitter range.

    Args:
        data_config: Validated DataConfig.
        sampler_config: Validated SamplerConfig.

    Raises:
        ValueError: If padded_size is too small for the requested input_len and max_jitter.
    """
    input_len = data_config["input_len"]
    max_jitter = data_config["max_jitter"]
    padded_size = sampler_config["padded_size"]
    
    required_size = input_len + 2 * max_jitter
    
    if padded_size < required_size:
        raise ValueError(
            f"Sampler padded_size ({padded_size}) is smaller than required size "
            f"({required_size} = input_len {input_len} + 2 * max_jitter {max_jitter}). "
            "Please increase padded_size or decrease input_len/max_jitter."
        )


def validate_train_config(config: TrainConfig) -> TrainConfig:
    """
    Validates the training configuration dictionary.

    Args:
        config: Dictionary containing training configuration.

    Returns:
        TrainConfig: Validated and typed configuration object.

    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("Train config must be a dictionary")

    required_keys = {
        "batch_size",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "patience",
        "optimizer",
        "filter_bias_and_bn",
    }
    # Optional: scheduler_type, scheduler_args
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Train config missing required keys: {missing}")

    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")

    if not isinstance(config["max_epochs"], int) or config["max_epochs"] <= 0:
        raise ValueError("max_epochs must be a positive integer")

    if not isinstance(config["learning_rate"], float) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be a positive float")
        
    if not isinstance(config["weight_decay"], float) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be a non-negative float")

    if not isinstance(config["patience"], int) or config["patience"] < 0:
        raise ValueError("patience must be a non-negative integer")

    if not isinstance(config["optimizer"], str):
        raise TypeError("optimizer must be a string")

    if not isinstance(config["filter_bias_and_bn"], bool):
        raise TypeError("filter_bias_and_bn must be a boolean")

    scheduler_type = config.get("scheduler_type", "default")
    if not isinstance(scheduler_type, str):
        raise TypeError("scheduler_type must be a string")

    scheduler_args = config.get("scheduler_args", {})
    if not isinstance(scheduler_args, dict):
        raise TypeError("scheduler_args must be a dictionary")

    return {
        "batch_size": config["batch_size"],
        "max_epochs": config["max_epochs"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "patience": config["patience"],
        "optimizer": config["optimizer"],
        "scheduler_type": scheduler_type,
        "scheduler_args": scheduler_args,
        "filter_bias_and_bn": config["filter_bias_and_bn"],
    }


def validate_predict_config(config: PredictConfig) -> PredictConfig:
    """
    Validates the prediction configuration dictionary.

    Args:
        config: Dictionary containing prediction configuration.

    Returns:
        PredictConfig: Validated and typed configuration object.

    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("Predict config must be a dictionary")

    required_keys = {"stride", "aggregation"}
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Predict config missing required keys: {missing}")

    stride = config["stride"]
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")

    use_folds = config.get("use_folds", ["test", "val"])
    if not isinstance(use_folds, list):
        raise TypeError("use_folds must be a list of strings")
    valid_folds = {"train", "test", "val"}
    if not all(f in valid_folds for f in use_folds):
        raise ValueError(f"use_folds must be subset of {valid_folds}")

    aggregation = config["aggregation"]
    if not isinstance(aggregation, str):
        raise TypeError("aggregation must be a string")
    if aggregation not in {"model", "interval+model"}:
        raise ValueError("aggregation must be 'model' or 'interval+model'")

    return {
        "stride": stride,
        "use_folds": use_folds,
        "aggregation": aggregation,
    }


def validate_predict_and_data_compatibility(
    predict_config: PredictConfig, data_config: DataConfig
) -> None:
    """
    Validates compatibility between prediction and data configurations.

    Args:
        predict_config: Validated PredictConfig.
        data_config: Validated DataConfig.

    Raises:
        ValueError: If stride > output_len.
    """
    stride = predict_config["stride"]
    output_len = data_config["output_len"]

    if stride > output_len:
        raise ValueError(
            f"Prediction stride ({stride}) cannot be larger than model output length ({output_len})."
        )


def validate_model_config(config: ModelConfig) -> ModelConfig:
    """
    Validates the model configuration dictionary.

    Args:
        config: Dictionary containing model configuration.

    Returns:
        ModelConfig: Validated and typed configuration object.

    Raises:
        TypeError: If input is not a dictionary or contains invalid types.
        ValueError: If required keys are missing or values are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("Model config must be a dictionary")

    required_keys = {
        "name",
        "model_cls",
        "loss_cls",
        "loss_args",
        "metrics_cls",
        "metrics_args",
        "model_args",
    }
    if not all(key in config for key in required_keys):
        missing = required_keys - config.keys()
        raise ValueError(f"Model config missing required keys: {missing}")

    if not isinstance(config["name"], str):
        raise TypeError("name must be a string")

    # Validate presence of model/loss classes (context dependent)
    
    # Check for required args in model_args (now optional/convention, but we should validate if present)
    # The requirement is moving them to model_args, so we should assume they might be there.
    # We can validate them if they exist, or check if specific keys are required by policy.
    # Given the previous validation was strict, let's strictly validate them INSIDE model_args.
    
    model_args = config["model_args"]
    if not isinstance(model_args, dict):
        raise TypeError("model_args must be a dictionary")

    # Validate input_channels if present
    if "input_channels" in model_args:
        if not isinstance(model_args["input_channels"], (list, tuple)):
            raise TypeError("model_args['input_channels'] must be a list or tuple of strings")
        if not all(isinstance(c, str) for c in model_args["input_channels"]):
            raise TypeError("model_args['input_channels'] must contain only strings")
        if len(model_args["input_channels"]) == 0:
            raise ValueError("model_args['input_channels'] must not be empty")

    # Validate output_channels if present
    if "output_channels" in model_args:
        if not isinstance(model_args["output_channels"], (list, tuple)):
            raise TypeError("model_args['output_channels'] must be a list or tuple of strings")
        if not all(isinstance(c, str) for c in model_args["output_channels"]):
            raise TypeError("model_args['output_channels'] must contain only strings")
        if len(model_args["output_channels"]) == 0:
            raise ValueError("model_args['output_channels'] must not be empty")

    # Validate output_type if present
    if "output_type" in model_args:
        if not isinstance(model_args["output_type"], str):
            raise TypeError("model_args['output_type'] must be a string")
        valid_types = {"signal", "decoupled"}
        if model_args["output_type"] not in valid_types:
            raise ValueError(f"model_args['output_type'] must be one of {valid_types}")

    return {
        "name": config["name"],
        "model_cls": config["model_cls"],
        "loss_cls": config["loss_cls"],
        "loss_args": config["loss_args"],
        "metrics_cls": config["metrics_cls"],
        "metrics_args": config["metrics_args"],
        "model_args": config["model_args"],
    }


def validate_data_and_model_compatibility(
    data_config: DataConfig, model_config: ModelConfig
) -> None:
    """
    Validates compatibility between data and model configurations.

    Checks if the model's input/output channels match the data configuration.

    Args:
        data_config: Validated DataConfig.
        model_config: Validated ModelConfig.

    Raises:
        ValueError: If channel names do not match between model and data.
    """
    # Check output channels vs targets
    target_channels = set(data_config["targets"].keys())
    
    # Look in model_args for channels
    model_args = model_config["model_args"]
    
    if "output_channels" in model_args:
        model_outputs = set(model_args["output_channels"])
        if target_channels != model_outputs:
            raise ValueError(
                f"Model output channels {model_outputs} do not match data targets {target_channels}"
            )

    # Ensure all data input tracks are present in model input channels
    if "input_channels" in model_args:
        input_tracks = set(data_config["inputs"].keys())
        # Assuming model_config inputs includes both sequence and tracks?
        # Or just tracks? User said "('A','C','G,'T',"name of signal trak")". So it includes sequence.
        
        # So model inputs should be superset of data inputs keys?
        model_inputs = set(model_args["input_channels"])
        if not input_tracks.issubset(model_inputs):
            missing = input_tracks - model_inputs
            raise ValueError(f"Data inputs {missing} are not in model input channels")
