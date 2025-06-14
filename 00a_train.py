# 00a_train.py

import click
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from icecream import ic
import wandb

# --- Import all models ---
from modules import MLPnet, ResNetForCifar, VGGForCifar, label_smooth

def get_dataset(dataset_name, data_root='./data'):
    """Gets the specified dataset from torchvision."""
    ic(f"Loading {dataset_name} dataset...")
    
    # Normalization constants for different datasets
    DATASET_STATS = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        'fashion_mnist': ((0.2860,), (0.3530,))
    }
    
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    mean, std = DATASET_STATS[dataset_name]
    
    # Fashion-MNIST is grayscale, needs a different transform pipeline
    if dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Grayscale(num_output_channels=3), # Convert to 3 channels for CNNs
        ])
        dataset_class = torchvision.datasets.FashionMNIST
        return dataset_class(root=data_root, train=True, download=True, transform=transform)

    # Transforms for color datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if dataset_name == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == 'cifar100':
        dataset_class = torchvision.datasets.CIFAR100
    elif dataset_name == 'svhn':
        # SVHN uses 'split' instead of 'train'
        return torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform)
        
    return dataset_class(root=data_root, train=True, download=True, transform=transform)


def train_model(model, trainloader, device, epochs, lr, num_classes=10):
    """The main training loop, now handles both single and dual output models."""
    ic(f"Starting training for {epochs} epochs on {device}...")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # --- Handle different model output signatures ---
            if isinstance(outputs, tuple):
                # For MLPnet with auxiliary output
                main_out, aux_out = outputs
                smooth_labels = label_smooth(labels, num_classes).to(device)
                main_loss = F.kl_div(F.log_softmax(main_out, dim=1), smooth_labels, reduction='batchmean')
                aux_loss = F.kl_div(F.log_softmax(aux_out, dim=1), smooth_labels, reduction='batchmean')
                loss = main_loss + 0.3 * aux_loss
            else:
                # For standard models (ResNet, VGG)
                main_out = outputs
                loss = F.cross_entropy(main_out, labels) # Standard cross-entropy is fine here
            # -----------------------------------------------

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(main_out.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct_preds / total_samples
        ic(f'Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

@click.command()
@click.option('--model', type=click.Choice(['MLPnet', 'resnet18', 'resnet34', 'vgg11_bn', 'vgg16_bn']), default='resnet18', help='Model architecture to train.')
@click.option('--dataset', type=click.Choice(['cifar10', 'cifar100', 'svhn', 'fashion_mnist']), default='cifar10', help='Dataset to use for training.')
@click.option('--epochs', type=int, default=50, help='Number of training epochs.')
@click.option('--lr', type=float, default=1e-3, help='Learning rate.')
@click.option('--batch-size', type=int, default=128, help='Training batch size.')
@click.option('--save-dir', type=click.Path(path_type=Path), default='outputs/weights', help='Directory to save the trained model weights.')
def main(model, dataset, epochs, lr, batch_size, save_dir):
    """Train a specified model on a specified dataset."""
    
    # Setup
    device = torch.device('mps')
    ic(f"Using device: {device}")
    
    # Dynamic save path
    save_path = save_dir / f'{dataset}_{model}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Data Loading
    trainset = get_dataset(dataset, data_root='./data')
    num_classes = len(trainset.classes) if hasattr(trainset, 'classes') else 10 # SVHN/FashionMNIST don't have a .classes attribute
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # Model Initialization
    ic(f"Initializing model: {model} for {num_classes} classes")
    if model == 'MLPnet':
        net = MLPnet(num_class=num_classes).to(device)
    elif 'resnet' in model:
        net = ResNetForCifar(resnet_type=model, num_classes=num_classes).to(device)
    elif 'vgg' in model:
        net = VGGForCifar(vgg_type=model, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Training
    train_model(net, trainloader, device, epochs, lr, num_classes)
    
    # Save Model
    ic(f"Saving model weights to: {save_path}")
    torch.save(net.state_dict(), save_path)
    
    ic("Training complete.")
    return

if __name__ == '__main__':
    main()