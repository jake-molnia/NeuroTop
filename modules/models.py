# modules/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

# --- Base Model ---
class BaseModel(nn.Module):
    """A base model class with a helper to get the device."""
    def get_device(self):
        return next(self.parameters()).device

# --- Model Definitions ---
class MNIST_MLP(BaseModel):
    """
    The advanced MLP model with built-in masking and feature taps for analysis.
    """
    def __init__(self, use_batchnorm=False, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        ic(f"Initializing MNIST_MLP with batchnorm={use_batchnorm}, dropout={dropout_rate}")
        
        self.layer1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512) if use_batchnorm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256) if use_batchnorm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128) if use_batchnorm else nn.Identity()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.output = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
        self.tap1, self.tap2, self.tap3 = (nn.Identity() for _ in range(3))
        self._layer_shapes = {'layer1': (512,), 'layer2': (256,), 'layer3': (128,)}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ic(f"Model parameters: total={total_params}, trainable={trainable_params}")
        ic(f"Layer shapes: {self._layer_shapes}")

    def get_feature_layers(self): 
        return {'layer1': self.tap1, 'layer2': self.tap2, 'layer3': self.tap3}
    
    @property
    def layer_shapes(self): 
        return self._layer_shapes

    def forward(self, x, masks=None):
        if masks is None:
            masks = {name: torch.ones(shape, device=self.get_device()) for name, shape in self.layer_shapes.items()}
            
        batch_size = x.size(0)
        x = x.view(-1, 28 * 28)
        
        x_l1 = self.dropout1(self.relu(self.bn1(self.layer1(x))))
        x_l1_tapped = self.tap1(x_l1) * masks['layer1']
        
        x_l2 = self.dropout2(self.relu(self.bn2(self.layer2(x_l1_tapped))))
        x_l2_tapped = self.tap2(x_l2) * masks['layer2']
        
        x_l3 = self.dropout3(self.relu(self.bn3(self.layer3(x_l2_tapped))))
        x_l3_tapped = self.tap3(x_l3) * masks['layer3']
        
        return self.output(x_l3_tapped)

# --- Model Factory ---
def get_model(model_config):
    """
    Factory function to instantiate a model based on a model specification.
    FIXED: Now correctly accesses keys from the model_config dictionary it receives.
    """
    arch = model_config['architecture']
    params = model_config.get('params', {})
    ic(f"Creating model: {arch}")
    ic(f"Model parameters: {params}")
    
    if arch == 'MNIST_MLP':
        model = MNIST_MLP(**params)
    else:
        raise ValueError(f"Model architecture '{arch}' not recognized.")
        
    print(f"Instantiated model: {arch} with params: {params}")
    return model