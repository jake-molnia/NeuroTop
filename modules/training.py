"""
Training and evaluation utilities with logging integration.
Clean training loops with comprehensive metric tracking.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def train_and_evaluate(
    config: Dict[str, Any],
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    exp_logger: Optional[Any] = None
) -> torch.nn.Module:
    """
    Complete training loop with evaluation and logging.
    
    Args:
        exp_logger: Optional experiment logger for metric tracking
    """
    
    # Setup optimizer
    opt_config = config['training']['optimizer']
    optimizer_class = getattr(optim, opt_config['name'])
    optimizer = optimizer_class(model.parameters(), **opt_config['params'])
    
    logger.info(f"Optimizer: {opt_config['name']} with params: {opt_config['params']}")
    
    # Setup scheduler if specified
    scheduler = None
    if 'lr_scheduler' in config['training']:
        sched_config = config['training']['lr_scheduler']
        scheduler_class = getattr(optim.lr_scheduler, sched_config['name'])
        scheduler = scheduler_class(optimizer, **sched_config['params'])
        logger.info(f"Scheduler: {sched_config['name']} with params: {sched_config['params']}")
    
    model.to(device)
    epochs = config['training']['epochs']
    
    logger.info(f"Starting training for {epochs} epochs on {device}")
    
    best_test_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Learning rate scheduling
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                logger.debug(f"LR updated: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Evaluation phase
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        # Track best performance
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # Logging
        logger.info(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                   f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
        
        # Log to experiment tracker if provided
        if exp_logger:
            metrics = {
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'test/loss': test_loss,
                'test/accuracy': test_acc,
                'train/learning_rate': optimizer.param_groups[0]['lr']
            }
            exp_logger.log_metrics(metrics, step=epoch)
    
    logger.info(f"Training completed. Best test accuracy: {best_test_acc:.2f}%")
    return model

def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)  # No masks during training
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
        
        # Update progress bar
        current_acc = 100.0 * correct / total_samples
        pbar.set_postfix({'loss': loss.item(), 'acc': f'{current_acc:.1f}%'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total_samples
    
    return avg_loss, accuracy

def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    desc: str = "Evaluating"
) -> Tuple[float, float]:
    """
    Evaluate model performance.
    
    Args:
        masks: Optional neuron masks for ablation testing
        desc: Description for progress bar
    """
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, leave=False)
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass with optional masking
            output = model(data, masks=masks)
            
            # Accumulate metrics
            loss = F.cross_entropy(output, target, reduction='sum')
            total_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            # Update progress bar
            current_acc = 100.0 * correct / total_samples
            pbar.set_postfix({'acc': f'{current_acc:.1f}%'})
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    
    # Log masking info if applicable
    if masks is not None:
        total_masked = sum((mask == 0).sum().item() for mask in masks.values())
        logger.debug(f"Evaluation with {total_masked} masked neurons: {accuracy:.2f}% accuracy")
    
    return avg_loss, accuracy

def quick_evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> float:
    """Quick accuracy evaluation for ablation testing."""
    
    model.eval()
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
    
    return 100.0 * correct / total_samples

def compute_gradient_importance(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    layer_name: str,
    device: torch.device,
    max_batches: int = 5
) -> torch.Tensor:
    """
    Compute gradient-based neuron importance scores.
    Used for hybrid ablation strategies.
    """
    
    model.train()  # Need gradients
    feature_layers = model.get_feature_layers()
    
    if layer_name not in feature_layers:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    importance_scores = None
    
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
            
        data, target = data.to(device), target.to(device)
        
        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Get gradients from the target layer
        layer = feature_layers[layer_name]
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            # For linear layers, sum over input dimensions
            layer_importance = layer.weight.grad.abs().sum(dim=1)
            
            if importance_scores is None:
                importance_scores = layer_importance.clone()
            else:
                importance_scores += layer_importance
    
    if importance_scores is not None:
        importance_scores = importance_scores / max_batches
        logger.info(f"Computed gradient importance for {len(importance_scores)} neurons in {layer_name}")
        return importance_scores
    else:
        raise RuntimeError(f"Could not compute gradients for layer {layer_name}")
class FlowAwareTrainer:
    """Enhanced trainer with temporal activation capture."""
    
    def __init__(self, model, config, temporal_storage=None, experiment_id=None):
        self.model = model
        self.config = config
        self.temporal_storage = temporal_storage
        self.experiment_id = experiment_id
        self.capture_epochs = config.get('temporal_analysis', {}).get('capture_epochs', [])
    
    def should_capture(self, epoch):
        return epoch in self.capture_epochs
    
    def capture_activations(self, model, data_loader, layers, epoch, max_samples, device):
        """Capture activations for temporal analysis."""
        if not self.should_capture(epoch) or not self.temporal_storage:
            return
            
        from .activations import extract_activations
        activations = extract_activations(model, data_loader, layers, max_samples, device)
        self.temporal_storage.capture_epoch(
            self.experiment_id, 
            epoch, 
            activations,
            run_flow_analysis=True
        )
        
    def train_with_flow_monitoring(self, train_loader, test_loader, device, exp_logger=None):
        """Train model with flow monitoring and temporal capture."""
        
        # Setup optimizer and scheduler
        from torch import optim
        opt_config = self.config['training']['optimizer']
        optimizer_class = getattr(optim, opt_config['name'])
        optimizer = optimizer_class(self.model.parameters(), **opt_config['params'])
        
        scheduler = None
        if 'lr_scheduler' in self.config['training']:
            sched_config = self.config['training']['lr_scheduler']
            scheduler_class = getattr(optim.lr_scheduler, sched_config['name'])
            scheduler = scheduler_class(optimizer, **sched_config['params'])
        
        self.model.to(device)
        epochs = self.config['training']['epochs']
        
        from .training import train_epoch, evaluate
        
        for epoch in range(1, epochs + 1):
            # Training phase
            train_loss, train_acc = train_epoch(self.model, train_loader, optimizer, device, epoch)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
            
            # Evaluation phase
            test_loss, test_acc = evaluate(self.model, test_loader, device)
            
            # Capture activations if this is a capture epoch
            if self.should_capture(epoch):
                analysis_config = self.config['analysis']['activation_extraction']
                self.capture_activations(
                    self.model, test_loader, analysis_config['layers'], 
                    epoch, analysis_config.get('max_samples'), device
                )
            
            # Log to experiment tracker
            if exp_logger:
                metrics = {
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'test/loss': test_loss,
                    'test/accuracy': test_acc,
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }
                exp_logger.log_metrics(metrics, step=epoch)
        
        return self.model