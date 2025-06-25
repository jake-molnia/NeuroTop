"""
Neural network model definitions with analysis hooks.
Clean model implementations with built-in ablation support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """Base model class with common functionality."""
    
    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device
    
    def get_feature_layers(self) -> Dict[str, nn.Module]:
        """Return dictionary of named feature layers for analysis."""
        raise NotImplementedError("Subclasses must implement get_feature_layers")
    
    @property
    def layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return shapes of analyzable layers."""
        raise NotImplementedError("Subclasses must implement layer_shapes")

class MNIST_MLP(BaseModel):
    """
    MLP for MNIST/FashionMNIST with built-in masking and analysis hooks.
    Supports dynamic input size based on dataset.
    """
    
    def __init__(
        self, 
        input_size: int = 784,
        use_batchnorm: bool = False, 
        dropout_rate: float = 0.5,
        hidden_sizes: Tuple[int, ...] = (512, 256, 128),
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        logger.info(f"Creating MNIST_MLP: input={input_size}, hidden={hidden_sizes}, classes={num_classes}")
        logger.info(f"Regularization: batchnorm={use_batchnorm}, dropout={dropout_rate}")
        
        # Build layers dynamically
        self.layers = nn.ModuleDict()
        self.batch_norms = nn.ModuleDict()
        self.dropouts = nn.ModuleDict()
        self.taps = nn.ModuleDict()  # Analysis hooks
        
        # First layer
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layer_name = f'layer{i+1}'
            
            self.layers[layer_name] = nn.Linear(prev_size, hidden_size)
            self.batch_norms[layer_name] = nn.BatchNorm1d(hidden_size) if use_batchnorm else nn.Identity()
            self.dropouts[layer_name] = nn.Dropout(dropout_rate)
            self.taps[layer_name] = nn.Identity()  # Hook point for analysis
            
            prev_size = hidden_size
        
        # Output layer
        self.output = nn.Linear(prev_size, num_classes)
        self.relu = nn.ReLU()
        
        # Store layer shapes for ablation
        self._layer_shapes = {f'layer{i+1}': (size,) for i, size in enumerate(hidden_sizes)}
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")
    
    def get_feature_layers(self) -> Dict[str, nn.Module]:
        """Return analysis hook points."""
        return dict(self.taps)
    
    @property
    def layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return shapes of each analyzable layer."""
        return self._layer_shapes
    
    def forward(self, x: torch.Tensor, masks: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional neuron masking for ablation."""
        
        # Default masks (no ablation)
        if masks is None:
            masks = {name: torch.ones(shape, device=self.get_device()) 
                    for name, shape in self.layer_shapes.items()}
        
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Pass through hidden layers
        for i, hidden_size in enumerate(self.hidden_sizes):
            layer_name = f'layer{i+1}'
            
            # Linear transformation
            x = self.layers[layer_name](x)
            
            # Batch normalization
            x = self.batch_norms[layer_name](x)
            
            # Activation
            x = self.relu(x)
            
            # Dropout
            x = self.dropouts[layer_name](x)
            
            # Analysis tap with optional masking
            x = self.taps[layer_name](x)
            if layer_name in masks:
                x = x * masks[layer_name]
        
        # Output layer
        x = self.output(x)
        return x

class CIFAR_MLP(BaseModel):
    """
    Larger MLP for CIFAR-10 (flattened).
    Similar structure to MNIST_MLP but scaled for higher-dimensional input.
    """
    
    def __init__(
        self,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.3,
        hidden_sizes: Tuple[int, ...] = (2048, 1024, 512, 256),
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = 3072  # 32*32*3
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        logger.info(f"Creating CIFAR_MLP: hidden={hidden_sizes}, classes={num_classes}")
        
        # Build layers (same pattern as MNIST_MLP)
        self.layers = nn.ModuleDict()
        self.batch_norms = nn.ModuleDict()
        self.dropouts = nn.ModuleDict()
        self.taps = nn.ModuleDict()
        
        prev_size = self.input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layer_name = f'layer{i+1}'
            
            self.layers[layer_name] = nn.Linear(prev_size, hidden_size)
            self.batch_norms[layer_name] = nn.BatchNorm1d(hidden_size) if use_batchnorm else nn.Identity()
            self.dropouts[layer_name] = nn.Dropout(dropout_rate)
            self.taps[layer_name] = nn.Identity()
            
            prev_size = hidden_size
        
        self.output = nn.Linear(prev_size, num_classes)
        self.relu = nn.ReLU()
        
        self._layer_shapes = {f'layer{i+1}': (size,) for i, size in enumerate(hidden_sizes)}
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"CIFAR_MLP created: {total_params:,} parameters")
    
    def get_feature_layers(self) -> Dict[str, nn.Module]:
        return dict(self.taps)
    
    @property
    def layer_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._layer_shapes
    
    def forward(self, x: torch.Tensor, masks: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional masking."""
        
        if masks is None:
            masks = {name: torch.ones(shape, device=self.get_device()) 
                    for name, shape in self.layer_shapes.items()}
        
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            layer_name = f'layer{i+1}'
            
            x = self.layers[layer_name](x)
            x = self.batch_norms[layer_name](x)
            x = self.relu(x)
            x = self.dropouts[layer_name](x)
            x = self.taps[layer_name](x)
            
            if layer_name in masks:
                x = x * masks[layer_name]
        
        return self.output(x)

def get_model(model_config: Dict[str, Any]) -> BaseModel:
    """Factory function to create models from configuration."""
    
    architecture = model_config['architecture']
    params = model_config.get('params', {})
    
    logger.info(f"Creating model: {architecture}")
    logger.debug(f"Model parameters: {params}")
    
    if architecture == 'MNIST_MLP':
        # Default to MNIST input size, but allow override
        if 'input_size' not in params:
            params['input_size'] = 784
        model = MNIST_MLP(**params)
        
    elif architecture == 'CIFAR_MLP':
        model = CIFAR_MLP(**params)
        
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
    
    logger.info(f"Successfully created {architecture}")
    return model

def get_model_for_dataset(dataset_name: str, model_params: Dict[str, Any] = None) -> BaseModel:
    """Create appropriate model for dataset."""
    
    if model_params is None:
        model_params = {}
    
    if dataset_name in ['MNIST', 'FashionMNIST']:
        return MNIST_MLP(input_size=784, **model_params)
    elif dataset_name == 'CIFAR10':
        return CIFAR_MLP(**model_params)
    else:
        raise ValueError(f"No default model for dataset: {dataset_name}")