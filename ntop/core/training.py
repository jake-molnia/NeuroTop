import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any, Union

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 test_loader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 enable_monitoring: bool = False):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_monitoring = enable_monitoring
        
        self.model.to(self.device)
        self.monitors = []
        self.epoch = 0
        
    def add_monitor(self, monitor, frequency: int = 1):
        if hasattr(monitor, 'capture_snapshot'):
            self.monitors.append((monitor, frequency))
        else:
            self.monitors.append((monitor, frequency))
    
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
    
    def evaluate(self) -> float:
        if self.test_loader is None:
            return 0.0
            
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return 100 * correct / total
    
    def _execute_monitors(self, epoch: int):
        if not self.enable_monitoring or not self.monitors:
            return
            
        dataloader = self.test_loader or self.train_loader
        
        for monitor, frequency in self.monitors:
            if (epoch + 1) % frequency == 0:
                try:
                    if hasattr(monitor, 'capture_snapshot'):
                        monitor.capture_snapshot(self.model, epoch, dataloader)
                    else:
                        monitor(self.model, epoch)
                except Exception as e:
                    print(f"Monitor failed at epoch {epoch}: {e}")
    
    def train(self, epochs: int, verbose: bool = True) -> Dict[str, Any]:
        losses = []
        accuracies = []
        
        if verbose and self.enable_monitoring:
            print("Training with topology monitoring enabled")
        
        for epoch in range(epochs):
            self.epoch = epoch
            loss = self.train_epoch()
            losses.append(loss)
            
            if self.test_loader is not None:
                acc = self.evaluate()
                accuracies.append(acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.2f}%")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
            
            if self.enable_monitoring:
                self._execute_monitors(epoch)
        
        results = {'losses': losses}
        if accuracies:
            results['accuracies'] = accuracies
            
        return results

def quick_train(model: nn.Module, train_loader: DataLoader, epochs: int = 10,
                lr: float = 1e-3, device: Optional[torch.device] = None) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, train_loader, optimizer, device=device)
    trainer.train(epochs, verbose=True)
    return model