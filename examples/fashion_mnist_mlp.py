import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ntop.analysis as analysis
import ntop.monitoring as monitoring
import ntop.plots as plots
import matplotlib.pyplot as plt

class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x): return self.layers(x.view(x.size(0), -1))


def setup_data(batch_size=128, subset_size=8000):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    if subset_size:
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        test_indices = torch.randperm(len(test_dataset))[:subset_size//4]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2), DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return total_loss / len(train_loader), 100 * correct / total

def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


def main():
    print("Fashion-MNIST MLP homology")
    print("="*50)
    output_folder = './outputs/fashion_mnist_MLP'
    import os
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_loader, test_loader = setup_data()
    print(f"Data loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    model = FashionMLP().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters")
    monitor = monitoring.ActivationMonitor(model)
    print("Monitor initialized")
    
    monitor.set_config(
        output_file=f'{output_folder}/topology_evolution.npz',
        max_samples=1500,
        distance_metric='euclidean',
        normalize_activations='none',
        max_dim=1,
        random_seed=42,
        filter_inactive_neurons=True,
        persistence_threshold=0.01,
    )
    
    print("\n" + "="*50)
    initial_state = monitor.analyze(test_loader, 0, "INITIAL STATE")
    
    print("\nGenerating initial topology visualizations...")
    figs = [
        (plots.plot_distance_matrix(initial_state), f'{output_folder}/fashion_initial_distance.png'),
        (plots.plot_persistence_diagram(initial_state), f'{output_folder}/fashion_initial_persistence.png'),
        (plots.plot_tsne_2d(initial_state), f'{output_folder}/fashion_initial_tsne.png'),
        (plots.plot_betti_numbers(initial_state), f'{output_folder}/fashion_initial_betti.png'),
        (plots.plot_layer_composition(initial_state), f'{output_folder}/fashion_layer_composition.png')
    ]
    for fig, filename in figs:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close('all')
    print("Initial plots saved")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    epochs = 5
    
    print(f"\nStarting training for {epochs} epochs...")

    best_accuracy = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()        
        print(f"Epoch {epoch:2d}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        monitor.analyze(test_loader, epoch, save=True)
        if test_acc > best_accuracy: best_accuracy = test_acc
    
    print(f"\nTraining complete! Best test accuracy: {best_accuracy:.2f}%")
    
    print("\n" + "="*50)
    final_topology = monitor.analyze(test_loader, epochs-1, "FINAL STATE")

    print("\nGenerating final topology visualizations...")
    final_figs = [
        (plots.plot_distance_matrix(final_topology), f'{output_folder}/fashion_final_distance.png'),
        (plots.plot_persistence_diagram(final_topology), f'{output_folder}/fashion_final_persistence.png'),
        (plots.plot_tsne_2d(final_topology), f'{output_folder}/fashion_final_tsne.png'),
        (plots.plot_betti_numbers(final_topology), f'{output_folder}/fashion_final_betti.png'),
        (plots.plot_tsne_3d(final_topology), f'{output_folder}/fashion_final_tsne_3d.png')
    ]
    for fig, filename in final_figs:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close('all')
    topology_states = monitor.topology_states
    if topology_states:
        rf_figs = [
            # Individual dimension heatmaps
            (plots.plot_rf_heatmap_by_layer(topology_states, 'rf_0'), f'{output_folder}/fashion_rf0_heatmap_by_layer.png'),
            # (plots.plot_rf_heatmap_by_layer(topology_states, 'rf_1'), f'{output_folder}/fashion_rf1_heatmap_by_layer.png'),
            (plots.plot_rf_heatmap_network(topology_states, 'rf_0'), f'{output_folder}/fashion_rf0_heatmap_network.png'),
            # (plots.plot_rf_heatmap_network(topology_states, 'rf_1'), f'{output_folder}/fashion_rf1_heatmap_network.png'),
            
            # Multi-dimensional comparison plots
            (plots.plot_rf_multidim_heatmap_by_layer(topology_states), f'{output_folder}/fashion_rf_multidim_heatmap.png'),
            (plots.plot_rf_box_evolution_multidim(topology_states), f'{output_folder}/fashion_rf_multidim_evolution.png'),
            
            # Original per-layer distribution plots (will work with rf_0 by default)
            (plots.plot_rf_distribution_evolution(topology_states), f'{output_folder}/fashion_rf_distribution_evolution.png'),
            (plots.plot_rf_violin_evolution(topology_states), f'{output_folder}/fashion_rf_violin_evolution.png'),
            (plots.plot_rf_box_evolution(topology_states), f'{output_folder}/fashion_rf_box_evolution.png'),
            
            # Network-wide distribution plots (combining all layers)
            (plots.plot_rf_distribution_evolution_network(topology_states, 'rf_0'), f'{output_folder}/fashion_rf0_distribution_evolution_network.png'),
            (plots.plot_rf_violin_evolution_network(topology_states, 'rf_0'), f'{output_folder}/fashion_rf0_violin_evolution_network.png'),
            (plots.plot_rf_box_evolution_network(topology_states, 'rf_0'), f'{output_folder}/fashion_rf0_box_evolution_network.png'),
            
            # Comprehensive comparison plot
            (plots.plot_rf_evolution_comparison(topology_states, 'rf_0'), f'{output_folder}/fashion_rf0_evolution_comparison.png'),
            
            # Additional network-wide plots for rf_1 (uncomment if rf_1 is available)
            # (plots.plot_rf_distribution_evolution_network(topology_states, 'rf_1'), f'{output_folder}/fashion_rf1_distribution_evolution_network.png'),
            # (plots.plot_rf_violin_evolution_network(topology_states, 'rf_1'), f'{output_folder}/fashion_rf1_violin_evolution_network.png'),
            # (plots.plot_rf_box_evolution_network(topology_states, 'rf_1'), f'{output_folder}/fashion_rf1_box_evolution_network.png'),
            # (plots.plot_rf_evolution_comparison(topology_states, 'rf_1'), f'{output_folder}/fashion_rf1_evolution_comparison.png'),
        ]
        for fig, filename in rf_figs:
            if fig:  # Check if figure was created successfully
                fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close('all')
        print("Multi-dimensional RF plots saved")
        if topology_states:
            first_rf = topology_states[0].get('rf_values', {})
            if first_rf:
                for layer_name, layer_rf in first_rf.items():
                    if isinstance(layer_rf, dict):
                        dims = sorted(layer_rf.keys())
                        print(f"Layer {layer_name}: RF dimensions {dims}")
    else:
        print("No topology states available for RF evolution plots")
    
    monitor.remove_hooks()


if __name__ == "__main__": main()