import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.monitors = []
        self.epoch = 0
        
    def add_monitor(self, monitor_fn: Callable, frequency: int = 1):
        self.monitors.append((monitor_fn, frequency))
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def train(self, epochs: int, verbose: bool = True) -> Dict[str, Any]:
        losses = []
        
        for epoch in range(epochs):
            self.epoch = epoch
            loss = self.train_epoch()
            losses.append(loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
            
            for monitor_fn, frequency in self.monitors:
                if (epoch + 1) % frequency == 0:
                    monitor_fn(self.model, epoch)
        
        return {'losses': losses}

def quick_train(model: nn.Module, train_loader: DataLoader, epochs: int = 10,
                lr: float = 1e-3, device: Optional[torch.device] = None) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, train_loader, optimizer, device=device)
    trainer.train(epochs, verbose=True)
    return model