# train.py

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

from modules import MLPnet

def label_smooth(y, n_class=10):
    """Applies label smoothing."""
    y_one_hot = torch.ones(len(y), n_class) * (0.1 / (n_class - 1))
    y_one_hot.scatter_(1, y.unsqueeze(1), 0.9)
    return y_one_hot

def train_model(model, trainloader, device, epochs, lr):
    """The main training loop with wandb logging."""
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
            smooth_labels = label_smooth(labels, 10).to(device)
            
            optimizer.zero_grad()
            
            main_out, aux_out = model(inputs)
            main_loss = F.kl_div(F.log_softmax(main_out, dim=1), smooth_labels, reduction='batchmean')
            aux_loss = F.kl_div(F.log_softmax(aux_out, dim=1), smooth_labels, reduction='batchmean')
            loss = main_loss + 0.3 * aux_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(main_out.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
        
        scheduler.step()
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct_preds / total_samples
        
        # Log metrics to wandb
        #wandb.log({
        #    'epoch': epoch,
        #    'train_loss': epoch_loss,
        #    'train_acc': epoch_acc,
        #    'lr': optimizer.param_groups[0]['lr']
        #})
        
        ic(f'Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

@click.command()
@click.option('--model', type=click.Choice(['MLPnet']), default='MLPnet', help='Model architecture to train.')
@click.option('--epochs', type=int, default=50, help='Number of training epochs.')
@click.option('--lr', type=float, default=1e-3, help='Learning rate.')
@click.option('--batch-size', type=int, default=128, help='Training batch size.')
@click.option('--save-path', type=click.Path(path_type=Path), default='outputs/weights/cifar10_mlp.pth', help='Path to save the trained model weights.')
@click.option('--wandb-project', default='neural-topology', help='Wandb project name.')
@click.option('--wandb-name', default=None, help='Wandb run name (defaults to a random name).')
def main(model, epochs, lr, batch_size, save_path, wandb_project, wandb_name):
    """Train a model on CIFAR-10 and log progress to Weights & Biases."""
    
    # Initialize wandb
    config = {
        'model': model,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'dataset': 'CIFAR-10'
    }
    #wandb.init(project=wandb_project, name=wandb_name, config=config)
    
    # Setup
    #device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    ic(f"Using device: {device}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Data Loading
    ic("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Model Initialization
    ic(f"Initializing model: {model}")
    if model == 'MLPnet':
        net = MLPnet().to(device)
        #wandb.watch(net) # Watch model gradients
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Training
    train_model(net, trainloader, device, epochs, lr)
    
    # Save Model
    ic(f"Saving model weights to: {save_path}")
    torch.save(net.state_dict(), save_path)
    
    #wandb.finish()
    ic("Training complete.")
    return

if __name__ == '__main__':
    main()
