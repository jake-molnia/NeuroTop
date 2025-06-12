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
    and an auxiliary output for deep supervision.
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
        
    def get_feature_layers(self):
        """
        Return layers for activation collection.
        We capture the state after each block, including after skip connections.
        """
        return {
            'layer1': self.batchnorm1, # After fc1
            'layer2': self.batchnorm1, # After fc2 + skip
            'layer3': self.batchnorm2, # After fc3
            'layer4': self.batchnorm2, # After fc4 + skip
            'layer5': self.batchnorm3, # After fc5
        }
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        
        # Block 1
        h1 = self.batchnorm1(F.gelu(self.fc1(x)))
        
        # Block 2 with skip connection
        h2 = self.batchnorm1(F.gelu(self.fc2(h1))) + h1
        
        # Block 3
        h3 = self.batchnorm2(F.gelu(self.fc3(h2)))

        # Block 4 with skip connection
        h4 = self.batchnorm2(F.gelu(self.fc4(h3))) + h3
        
        # Block 5
        h5 = self.batchnorm3(F.gelu(self.fc5(h4)))
        
        # Final output
        h5_dropped = self.dropout(h5)
        out_main = self.fc_last(h5_dropped)
        
        # Auxiliary output from an intermediate layer
        out_auxil = self.auxil(h2)
        
        return out_main, out_auxil
