# modules/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import torchvision.models as models

class BaseModel(nn.Module, ABC):
    """Abstract base model to ensure required methods are implemented."""
    
    @abstractmethod
    def get_feature_layers(self):
        """Return a dictionary of layers whose activations we want to collect."""
        pass
    
    @property
    @abstractmethod
    def layer_shapes(self):
        """Return a dictionary of layer names to their neuron counts."""
        pass
    
    @abstractmethod
    def forward(self, x, masks=None):
        """The forward pass of the model."""
        pass

    def get_device(self):
        """Helper to get the device of the model's parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

# --- MLPnet (Corrected and Final) ---
class MLPnet(BaseModel):
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
        self.tap1, self.tap2, self.tap3, self.tap4, self.tap5 = (nn.Identity() for _ in range(5))
        self._layer_shapes = {'layer1': 512, 'layer2': 512, 'layer3': 256, 'layer4': 256, 'layer5': 128}

    def get_feature_layers(self):
        return {'layer1': self.tap1, 'layer2': self.tap2, 'layer3': self.tap3, 'layer4': self.tap4, 'layer5': self.tap5}

    @property
    def layer_shapes(self):
        return self._layer_shapes

    def forward(self, x, masks=None):
        if masks is None:
            device = self.get_device()
            masks = {name: torch.ones(shape, device=device) for name, shape in self.layer_shapes.items()}

        x = torch.flatten(x, 1)
        h1 = self.tap1(self.batchnorm1(F.gelu(self.fc1(x)))) * masks['layer1']
        h2_res = self.batchnorm1(F.gelu(self.fc2(h1)))
        h2 = self.tap2(h2_res + h1) * masks['layer2']
        h3 = self.tap3(self.batchnorm2(F.gelu(self.fc3(h2)))) * masks['layer3']
        h4_res = self.batchnorm2(F.gelu(self.fc4(h3)))
        h4 = self.tap4(h4_res + h3) * masks['layer4']
        h5 = self.tap5(self.batchnorm3(F.gelu(self.fc5(h4)))) * masks['layer5']
        out_main = self.fc_last(self.dropout(h5))
        out_auxil = self.auxil(h2)
        return out_main, out_auxil

# --- ResNet Wrapper (Corrected and Final) ---
class ResNetForCifar(BaseModel):
    def __init__(self, resnet_type='resnet18', num_classes=10):
        super().__init__()
        self.resnet = getattr(models, resnet_type)(weights=None, num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.tap1, self.tap2, self.tap3, self.tap4 = (nn.Identity() for _ in range(4))
        self._feature_layers = {
            'block1_out': self.tap1, 'block2_out': self.tap2,
            'block3_out': self.tap3, 'block4_out': self.tap4,
        }
        # THIS IS THE KEY FIX: layer_shapes must store the number of channels (an int).
        self._layer_shapes = {
            'block1_out': 64,
            'block2_out': 128,
            'block3_out': 256,
            'block4_out': 512,
        }

    def get_feature_layers(self):
        return self._feature_layers

    @property
    def layer_shapes(self):
        return self._layer_shapes

    def forward(self, x, masks=None):
        if masks is None:
            device = self.get_device()
            masks = {name: torch.ones((1, shape, 1, 1), device=device) for name, shape in self.layer_shapes.items()}

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        layer1_out = self.resnet.layer1(x)
        x = self.tap1(layer1_out) * masks['block1_out']
        layer2_out = self.resnet.layer2(x)
        x = self.tap2(layer2_out) * masks['block2_out']
        layer3_out = self.resnet.layer3(x)
        x = self.tap3(layer3_out) * masks['block3_out']
        layer4_out = self.resnet.layer4(x)
        x = self.tap4(layer4_out) * masks['block4_out']
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

# --- VGG Wrapper (API consistent) ---
class VGGForCifar(BaseModel):
    def __init__(self, vgg_type='vgg11_bn', num_classes=10):
        super().__init__()
        self.vgg = getattr(models, vgg_type)(weights=None, num_classes=num_classes)
        self._feature_layers = {}
        self._layer_shapes = {}
        for i, layer in enumerate(self.vgg.features):
            if isinstance(layer, nn.Conv2d):
                name = f'conv_{len(self._feature_layers)}'
                self._feature_layers[name] = layer
                self._layer_shapes[name] = layer.out_channels
    
    def get_feature_layers(self): return self._feature_layers
    @property
    def layer_shapes(self): return self._layer_shapes
    
    def forward(self, x, masks=None):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)
        return x
