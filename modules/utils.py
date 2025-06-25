"""
Utility functions for experiment management and system monitoring.
Clean, focused utilities with proper logging.
"""

import torch
import numpy as np
import random
import os
import psutil
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time
import json

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all frameworks."""
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Set random seed to {seed}")

def get_device(prefer_mps: bool = True) -> torch.device:
    """Get the best available device for computation."""
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device

def monitor_memory() -> Dict[str, float]:
    """Monitor system and GPU memory usage."""
    
    memory_info = {}
    
    # System memory
    vm = psutil.virtual_memory()
    memory_info.update({
        'system_memory_used_gb': (vm.total - vm.available) / 1e9,
        'system_memory_available_gb': vm.available / 1e9,
        'system_memory_percent': vm.percent
    })
    
    # GPU memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            memory_info[f'gpu_{i}_memory_allocated_gb'] = allocated
            memory_info[f'gpu_{i}_memory_reserved_gb'] = reserved
    
    return memory_info

def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Get comprehensive model information."""
    
    info = {}
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info.update({
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / 1e6  # Assume float32
    })
    
    # Layer information
    layer_info = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_params = sum(p.numel() for p in module.parameters())
            layer_info[name] = {
                'type': type(module).__name__,
                'parameters': layer_params
            }
    
    info['layers'] = layer_info
    
    return info

def create_progress_callback(exp_logger: Optional[Any] = None):
    """Create a callback function for progress monitoring."""
    
    def callback(current: int, total: int, desc: str = "Progress"):
        percent = 100.0 * current / total
        logger.debug(f"{desc}: {current}/{total} ({percent:.1f}%)")
        
        if exp_logger:
            exp_logger.log_metrics({
                f'progress/{desc.lower()}': percent
            })
    
    return callback

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator."""
    return numerator / denominator if denominator != 0 else default

def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def estimate_memory_usage(
    batch_size: int,
    input_shape: Tuple[int, ...],
    model_params: int,
    dtype_size: int = 4
) -> Dict[str, float]:
    """Estimate memory usage for training."""
    
    # Input tensor memory
    input_elements = batch_size * np.prod(input_shape)
    input_memory = input_elements * dtype_size / 1e6  # MB
    
    # Model parameters
    model_memory = model_params * dtype_size / 1e6  # MB
    
    # Gradients (same size as parameters)
    gradient_memory = model_memory
    
    # Activation memory (rough estimate)
    activation_memory = input_memory * 2  # Rough estimate
    
    total_memory = input_memory + model_memory + gradient_memory + activation_memory
    
    return {
        'input_memory_mb': input_memory,
        'model_memory_mb': model_memory,
        'gradient_memory_mb': gradient_memory,
        'activation_memory_mb': activation_memory,
        'total_estimated_mb': total_memory
    }

class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Started: {self.description}")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed: {self.description} in {format_time(duration)}")
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

def cleanup_gpu_memory():
    """Clean up GPU memory."""
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache")

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as JSON file."""
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.debug(f"Saved JSON to {filepath}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file as dictionary."""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON from {filepath}")
    return data

def validate_experiment_setup(config: Dict[str, Any]) -> None:
    """Validate experiment setup and log warnings for potential issues."""
    
    # Check for common configuration issues
    warnings = []
    
    # Training configuration
    if config.get('training', {}).get('epochs', 0) > 100:
        warnings.append("Training for >100 epochs - consider early stopping")
    
    # Batch size vs memory
    batch_size = config.get('dataset', {}).get('batch_size', 32)
    if batch_size > 512:
        warnings.append(f"Large batch size ({batch_size}) may cause memory issues")
    
    # Analysis configuration
    max_samples = config.get('analysis', {}).get('activation_extraction', {}).get('max_samples')
    if max_samples and max_samples > 10000:
        warnings.append(f"Analyzing {max_samples} samples may be slow for topology computation")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"Setup warning: {warning}")
    
    if not warnings:
        logger.info("Experiment setup validation passed")

def get_git_commit() -> Optional[str]:
    """Get current git commit hash for reproducibility."""
    
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        commit_hash = result.stdout.strip()
        logger.debug(f"Git commit: {commit_hash[:8]}")
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("Could not get git commit hash")
        return None