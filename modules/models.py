# modules/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Abstract base model to ensure required methods are implemented."""
    
    @abstractmethod
    def get_feature_layers(self):
        """Return a dictionary of layers whose activations we want to collect."""
        pass
    
    @abstractmethod
    def forward(self, x):
        """The forward pass of the model."""
        pass

class MLPnet(BaseModel):
    """
    A Multi-Layer Perceptron designed for CIFAR-10, with skip connections
    and an auxiliary output for deep supervision. Now with activation taps and masking.
    """
    def __init__(self, input_size=3*32*32, num_class=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_last = nn.Linear(128, num_class)
        self.auxil = nn.Linear(512, num_class)
        
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)

        # --- New: Taps for clean activation collection ---
        self.tap1 = nn.Identity()
        self.tap2 = nn.Identity()
        self.tap3 = nn.Identity()
        self.tap4 = nn.Identity()
        self.tap5 = nn.Identity()

        self.layer_shapes = {
            'layer1': 512, 'layer2': 512, 'layer3': 256, 
            'layer4': 256, 'layer5': 128
        }
        self.reset_masks()

    def get_feature_layers(self):
        """
        Return layers for activation collection using identity taps.
        This ensures we capture the output of each conceptual block correctly.
        """
        return {
            'layer1': self.tap1,
            'layer2': self.tap2,
            'layer3': self.tap3,
            'layer4': self.tap4,
            'layer5': self.tap5,
        }

    def update_masks(self, masks_dict):
        """
        Updates the neuron masks for ablation studies.
        'masks_dict' maps layer names to boolean/binary tensors.
        """
        for name, mask in masks_dict.items():
            if name in self.layer_shapes:
                mask_buffer_name = f"mask_{name}"
                self.register_buffer(mask_buffer_name, mask.float())

    def reset_masks(self):
        """Resets all masks to be fully permissive (all ones)."""
        for name, shape in self.layer_shapes.items():
            mask_buffer_name = f"mask_{name}"
            self.register_buffer(mask_buffer_name, torch.ones(shape))
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        
        # Block 1
        h1 = self.batchnorm1(F.gelu(self.fc1(x)))
        h1 = self.tap1(h1) * getattr(self, 'mask_layer1', 1.0)
        
        # Block 2 with skip connection
        h2 = self.batchnorm1(F.gelu(self.fc2(h1))) + h1
        h2 = self.tap2(h2) * getattr(self, 'mask_layer2', 1.0)
        
        # Block 3
        h3 = self.batchnorm2(F.gelu(self.fc3(h2)))
        h3 = self.tap3(h3) * getattr(self, 'mask_layer3', 1.0)

        # Block 4 with skip connection
        h4 = self.batchnorm2(F.gelu(self.fc4(h3))) + h3
        h4 = self.tap4(h4) * getattr(self, 'mask_layer4', 1.0)
        
        # Block 5
        h5 = self.batchnorm3(F.gelu(self.fc5(h4)))
        h5 = self.tap5(h5) * getattr(self, 'mask_layer5', 1.0)
        
        # Final output
        h5_dropped = self.dropout(h5)
        out_main = self.fc_last(h5_dropped)
        
        # Auxiliary output from an intermediate layer (affected by masking on h2)
        out_auxil = self.auxil(h2)
        
        return out_main, out_auxil