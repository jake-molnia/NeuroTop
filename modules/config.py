"""
Configuration management and validation.
Handles hierarchical configs, templates, and validation.
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Configuration validation error."""
    pass

class ConfigManager:
    """Handles configuration loading, validation, and templating."""
    
    def __init__(self):
        self.templates = {}
        self._load_templates()
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file with validation."""
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_path}: {e}")
        
        # Apply environment variable substitution
        config = self._substitute_env_vars(config)
        
        # Validate configuration
        self._validate_config(config)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def create_from_template(
        self, 
        template_name: str, 
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create configuration from template with optional overrides."""
        
        if template_name not in self.templates:
            raise ConfigError(f"Template '{template_name}' not found")
        
        config = deepcopy(self.templates[template_name])
        
        if overrides:
            config = self._merge_configs(config, overrides)
        
        self._validate_config(config)
        
        logger.info(f"Created configuration from template '{template_name}'")
        return config
    
    def merge_configs(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations with overlay taking precedence."""
        return self._merge_configs(deepcopy(base), overlay)
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and values."""
        self._validate_config(config)
    
    def _load_templates(self) -> None:
        """Load configuration templates."""
        
        # Basic MNIST template
        self.templates['mnist_baseline'] = {
            'dataset': {
                'name': 'MNIST',
                'path': './data',
                'batch_size': 128,
                'num_workers': 4
            },
            'model': {
                'architecture': 'MNIST_MLP',
                'params': {
                    'use_batchnorm': False,
                    'dropout_rate': 0.5
                }
            },
            'training': {
                'enabled': True,
                'epochs': 10,
                'optimizer': {
                    'name': 'Adam',
                    'params': {'lr': 0.001}
                }
            },
            'analysis': {
                'activation_extraction': {
                    'layers': ['layer1', 'layer2', 'layer3'],
                    'max_samples': 1000,
                    'normalize_activations': True,
                    'normalization_method': 'standard'
                },
                'visualizations': {
                    'tsne_plot': True,
                    'distance_matrix': True,
                    'persistence_diagram': True
                }
            },
            'logging': {
                'project': 'neural-topology',
                'tags': ['mnist', 'baseline']
            }
        }
        
        # Grid search template
        self.templates['grid_search'] = {
            'base_config': self.templates['mnist_baseline'],
            'models': [
                {
                    'name': 'MLP_baseline',
                    'architecture': 'MNIST_MLP',
                    'params': {'use_batchnorm': False, 'dropout_rate': 0.5}
                }
            ],
            'ablations': [
                {
                    'strategy': 'homology_degree',
                    'params': {'homology_dim': 1}
                },
                {
                    'strategy': 'random',
                    'params': {}
                }
            ]
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and values."""
        
        # Required top-level sections
        required_sections = ['dataset', 'model', 'training', 'analysis']
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"Missing required section: {section}")
        
        # Validate dataset config
        self._validate_dataset_config(config['dataset'])
        
        # Validate model config
        self._validate_model_config(config['model'])
        
        # Validate training config
        self._validate_training_config(config['training'])
        
        # Validate analysis config
        self._validate_analysis_config(config['analysis'])
        
        logger.debug("Configuration validation passed")
    
    def _validate_dataset_config(self, dataset_config: Dict[str, Any]) -> None:
        """Validate dataset configuration."""
        
        required_fields = ['name', 'batch_size']
        for field in required_fields:
            if field not in dataset_config:
                raise ConfigError(f"Missing required dataset field: {field}")
        
        # Validate dataset name
        supported_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10']
        if dataset_config['name'] not in supported_datasets:
            raise ConfigError(f"Unsupported dataset: {dataset_config['name']}")
        
        # Validate batch size
        if not isinstance(dataset_config['batch_size'], int) or dataset_config['batch_size'] <= 0:
            raise ConfigError("batch_size must be a positive integer")
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        
        if 'architecture' not in model_config:
            raise ConfigError("Missing required model field: architecture")
        
        # Validate architecture
        supported_architectures = ['MNIST_MLP']
        if model_config['architecture'] not in supported_architectures:
            raise ConfigError(f"Unsupported architecture: {model_config['architecture']}")
    
    def _validate_training_config(self, training_config: Dict[str, Any]) -> None:
        """Validate training configuration."""
        
        if 'enabled' not in training_config:
            raise ConfigError("Missing required training field: enabled")
        
        if training_config['enabled']:
            required_fields = ['epochs', 'optimizer']
            for field in required_fields:
                if field not in training_config:
                    raise ConfigError(f"Missing required training field: {field}")
            
            # Validate epochs
            if not isinstance(training_config['epochs'], int) or training_config['epochs'] <= 0:
                raise ConfigError("epochs must be a positive integer")
    
    def _validate_analysis_config(self, analysis_config: Dict[str, Any]) -> None:
        """Validate analysis configuration."""
        
        if 'activation_extraction' not in analysis_config:
            raise ConfigError("Missing required analysis field: activation_extraction")
        
        extraction_config = analysis_config['activation_extraction']
        if 'layers' not in extraction_config:
            raise ConfigError("Missing required field: activation_extraction.layers")
        
        if not isinstance(extraction_config['layers'], list):
            raise ConfigError("activation_extraction.layers must be a list")
    
    def _merge_configs(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._merge_configs(base[key], value)
            else:
                base[key] = value
        
        return base
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""
        
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('$'):
                env_var = obj[1:]
                return os.getenv(env_var, obj)
            else:
                return obj
        
        return substitute_recursive(config)

# Global config manager instance
config_manager = ConfigManager()

# Convenience functions
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file."""
    return config_manager.load_config(config_path)

def create_from_template(template_name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create configuration from template."""
    return config_manager.create_from_template(template_name, overrides)

def merge_configs(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations."""
    return config_manager.merge_configs(base, overlay)

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration."""
    config_manager.validate_config(config)

def get_experiment_name(config: Dict[str, Any]) -> str:
    """Generate experiment name from configuration."""
    
    # Extract key components for naming
    dataset = config.get('dataset', {}).get('name', 'unknown')
    model_name = config.get('model', {}).get('name', 'model')
    
    # Add timestamp if not provided
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return f"{dataset}_{model_name}_{timestamp}"

def get_wandb_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract wandb-relevant configuration."""
    
    # Flatten config for wandb
    wandb_config = {}
    
    # Dataset info
    if 'dataset' in config:
        wandb_config.update({f"dataset_{k}": v for k, v in config['dataset'].items()})
    
    # Model info
    if 'model' in config:
        wandb_config.update({f"model_{k}": v for k, v in config['model'].items()})
    
    # Training info
    if 'training' in config:
        wandb_config.update({f"training_{k}": v for k, v in config['training'].items()})
    
    return wandb_config