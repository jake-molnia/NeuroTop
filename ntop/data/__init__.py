from .mnist import get_mnist_loaders, create_simple_mnist_model, get_quick_mnist, MNISTConfig
from .fashion_mnist import (
    get_fashion_mnist_loaders, create_simple_fashion_mnist_model, 
    get_quick_fashion_mnist, FashionMNISTConfig, FASHION_MNIST_CLASSES, get_class_name
)

__all__ = [
    'get_mnist_loaders', 'create_simple_mnist_model', 'get_quick_mnist', 'MNISTConfig',
    'get_fashion_mnist_loaders', 'create_simple_fashion_mnist_model', 
    'get_quick_fashion_mnist', 'FashionMNISTConfig', 'FASHION_MNIST_CLASSES', 'get_class_name'
]