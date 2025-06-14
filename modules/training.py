# modules/training.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def train_and_evaluate(config, model, train_loader, test_loader, device):
    """
    Handles the complete training and final evaluation of a model.
    """
    # Setup optimizer
    opt_config = config['training']['optimizer']
    optimizer_class = getattr(optim, opt_config['name'])
    optimizer = optimizer_class(model.parameters(), **opt_config['params'])

    # Setup scheduler (optional)
    scheduler = None
    if config['training'].get('lr_scheduler'):
        sched_config = config['training']['lr_scheduler']
        scheduler_class = getattr(optim.lr_scheduler, sched_config['name'])
        scheduler = scheduler_class(optimizer, **sched_config['params'])

    # Training loop
    model.to(device)
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # The forward pass in training does not use masks
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        if scheduler:
            scheduler.step()

        # Evaluation at the end of each epoch
        test_loss, test_acc = evaluate(model, test_loader, device, "Test")
        print(f"Epoch {epoch} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    return model

def evaluate(model, data_loader, device, name="Test", masks=None):
    """
    Evaluates the model on a given dataset.
    MODIFIED: Accepts an optional `masks` dictionary to pass to the model.
    """
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # Pass the masks to the model's forward method
            output = model(data, masks=masks)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    total_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return total_loss, accuracy
