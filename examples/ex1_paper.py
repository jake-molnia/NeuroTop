import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from ntop.monitoring import ActivationMonitor
import os

class LivePlot:
    def __init__(self, title="Training Progress"):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle(title)
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.ax.set_yscale('log')
        self.ax.grid(True)
        self.lines = {}
        self.data = defaultdict(list)
        self.steps = defaultdict(list)

    def update(self, metric_name, value, step=None):
        if step is None:
            step = len(self.data[metric_name])
        self.data[metric_name].append(value)
        self.steps[metric_name].append(step)
        if metric_name not in self.lines:
            (line,) = self.ax.plot([], [], label=metric_name)
            self.lines[metric_name] = line
            self.ax.legend()
        self.lines[metric_name].set_data(self.steps[metric_name], self.data[metric_name])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def final_save(self):
        plt.ioff()
        self.fig.savefig("final_training_plot.png")
        print("Final plot saved to final_training_plot.png")

    def close(self):
        plt.close(self.fig)

class ModularArithmetic(nn.Module):
    def __init__(self, modulus, d_model=128):
        super().__init__()
        self.modulus = modulus
        self.embed = nn.Embedding(modulus + 1, d_model)
        
        self.attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.attention_output_dense = nn.Linear(d_model, d_model)
        
        self.intermediate_dense = nn.Linear(d_model, 512)
        self.activation = nn.ReLU()
        self.output_dense = nn.Linear(512, d_model)
        
        self.unembed = nn.Linear(d_model, modulus)
        
    def forward(self, x):
        x = self.embed(x)
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.attention_output_dense(attn_out)
        x = attn_out[:, -1, :]
        
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.output_dense(x)
        
        return self.unembed(x)

class ModularDataset(Dataset):
    def __init__(self, modulus):
        self.data = []
        for x in range(modulus):
            for y in range(modulus):
                result = (x + y) % modulus
                self.data.append((
                    torch.tensor([x, y, modulus]),
                    torch.tensor(result)
                ))
        np.random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def plot_rf_outliers(rf_values, epoch, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    all_rf0 = []
    all_rf1 = []
    layer_info = []
    
    for comp_name, comp_data in rf_values.items():
        if 'rf_values' in comp_data:
            for layer, rf_dict in comp_data['rf_values'].items():
                if 'rf_0' in rf_dict:
                    vals = rf_dict['rf_0']
                    all_rf0.extend(vals)
                    layer_info.extend([(v, layer, 'rf_0') for v in vals])
                if 'rf_1' in rf_dict:
                    vals = rf_dict['rf_1']
                    all_rf1.extend(vals)
    
    if not all_rf0:
        return
    
    all_rf0 = np.array(all_rf0)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution with 1% outliers highlighted
    ax = axes[0, 0]
    threshold_99 = np.percentile(all_rf0, 99)
    threshold_1 = np.percentile(all_rf0, 1)
    
    ax.hist(all_rf0, bins=50, alpha=0.7, color='blue', label='All values')
    outliers_high = all_rf0[all_rf0 >= threshold_99]
    outliers_low = all_rf0[all_rf0 <= threshold_1]
    
    ax.hist(outliers_high, bins=20, alpha=0.9, color='red', label=f'Top 1% (>{threshold_99:.4f})')
    ax.hist(outliers_low, bins=20, alpha=0.9, color='orange', label=f'Bottom 1% (<{threshold_1:.4f})')
    ax.set_xlabel('RF_0 Value')
    ax.set_ylabel('Count')
    ax.set_title(f'RF_0 Distribution (Epoch {epoch})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot showing outliers
    ax = axes[0, 1]
    bp = ax.boxplot(all_rf0, vert=True, patch_artist=True, 
                     showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.5))
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_ylabel('RF_0 Value')
    ax.set_title(f'RF_0 Box Plot with Outliers (Epoch {epoch})')
    ax.grid(True, alpha=0.3)
    
    # 3. Sorted RF values with percentile lines
    ax = axes[1, 0]
    sorted_rf = np.sort(all_rf0)
    ax.plot(sorted_rf, linewidth=1)
    ax.axhline(threshold_99, color='red', linestyle='--', label='99th percentile')
    ax.axhline(threshold_1, color='orange', linestyle='--', label='1st percentile')
    ax.axhline(np.median(all_rf0), color='green', linestyle='--', label='Median')
    ax.set_xlabel('Neuron Index (sorted)')
    ax.set_ylabel('RF_0 Value')
    ax.set_title(f'Sorted RF_0 Values (Epoch {epoch})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Epoch: {epoch}
    
    RF_0 Statistics:
    Total neurons: {len(all_rf0):,}
    Mean: {np.mean(all_rf0):.4f}
    Median: {np.median(all_rf0):.4f}
    Std: {np.std(all_rf0):.4f}
    
    Min: {np.min(all_rf0):.4f}
    Max: {np.max(all_rf0):.4f}
    Range: {np.ptp(all_rf0):.4f}
    
    99th percentile: {threshold_99:.4f}
    1st percentile: {threshold_1:.4f}
    
    Top 1% neurons: {len(outliers_high):,}
    Bottom 1% neurons: {len(outliers_low):,}
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rf_outliers_epoch_{epoch}.png', dpi=150)
    plt.close(fig)
    
    print(f"  Saved RF outlier analysis to {output_dir}/rf_outliers_epoch_{epoch}.png")

def plot_persistence_diagrams(topo_data, epoch, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect RF values for all dimensions
    rf_data = {'rf_0': [], 'rf_1': [], 'rf_2': []}
    
    if 'by_components' in topo_data:
        for comp_data in topo_data['by_components'].values():
            if 'rf_values' in comp_data:
                for layer_rf in comp_data['rf_values'].values():
                    for dim in ['rf_0', 'rf_1', 'rf_2']:
                        if dim in layer_rf:
                            rf_data[dim].extend(layer_rf[dim])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red']
    dims = ['H0 (Components)', 'H1 (Loops)', 'H2 (Voids)']
    markers = ['o', 's', '^']
    max_val = 0
    
    for rf_key, color, dim_name, marker in zip(['rf_0', 'rf_1', 'rf_2'], colors, dims, markers):
        if rf_data[rf_key]:
            values = np.array(rf_data[rf_key])
            values = values[np.isfinite(values)]
            
            if len(values) > 0:
                births = np.zeros_like(values)
                deaths = values
                
                ax.scatter(births, deaths, alpha=0.6, s=30, c=color, marker=marker, 
                          label=f'{dim_name} ({len(values)} features)')
                
                max_val = max(max_val, np.max(deaths))
                
                # Highlight top 1% for each dimension
                if len(values) > 10:
                    threshold_99 = np.percentile(deaths, 99)
                    outliers_mask = deaths >= threshold_99
                    ax.scatter(births[outliers_mask], deaths[outliers_mask], 
                              alpha=0.9, s=80, c=color, marker='*', 
                              edgecolors='black', linewidths=1)
    
    # Diagonal line
    if max_val > 0:
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='Diagonal')
    
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death (RF Value)', fontsize=12)
    ax.set_title(f'Persistence Diagram - All Dimensions (Epoch {epoch})', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/persistence_epoch_{epoch}.png', dpi=150)
    plt.close(fig)
    
    print(f"  Saved persistence diagram to {output_dir}/persistence_epoch_{epoch}.png")
        
def train_grokking_with_topology():
    modulus = 113
    epochs = 10000
    batch_size = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModularArithmetic(modulus).to(device)
    
    dataset = ModularDataset(modulus)
    split = int(len(dataset) * 0.7)
    train_data = torch.utils.data.Subset(dataset, range(split))
    test_data = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    monitor = ActivationMonitor(model, model_type='mlp')
    plot = LivePlot("Grokking: Train/Test Loss")
    results = []
    
    print(f"Training on modulus {modulus} for {epochs} epochs")
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += len(targets)
        
        train_acc = train_correct / train_total
        train_loss = train_loss / train_total
        
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * len(targets)
                test_correct += (outputs.argmax(1) == targets).sum().item()
                test_total += len(targets)
        
        test_acc = test_correct / test_total
        test_loss = test_loss / test_total
        
        plot.update('train_loss', train_loss, epoch)
        plot.update('test_loss', test_loss, epoch)
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"\nEpoch {epoch}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
                
            try:
                torch.save(model.state_dict(), f'model_step_{epoch}.pt')
                topo = monitor.analyze(test_loader, epoch, save=True,
                                        distance_metric='euclidean',
                                        max_dim=2,
                                        analyze_by_components=True,
                                        max_samples=500)
                
                all_rf0 = []
                
                if 'by_components' in topo:
                    for comp_name, comp_data in topo['by_components'].items():
                        if 'rf_values' in comp_data:
                            for layer, rf_dict in comp_data['rf_values'].items():
                                if 'rf_0' in rf_dict:
                                    all_rf0.extend(rf_dict['rf_0'])
                
                if all_rf0:
                    rf0_mean = np.mean(all_rf0)
                    rf0_max = np.max(all_rf0)
                    
                    results.append({
                        'epoch': epoch,
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'rf0_mean': rf0_mean,
                        'rf0_max': rf0_max,
                        'rf0_std': np.std(all_rf0),
                        'rf0_99th': np.percentile(all_rf0, 99),
                        'rf0_1st': np.percentile(all_rf0, 1)
                    })
                    
                    # Generate plots
                    plot_rf_outliers(topo['by_components'], epoch)
                    plot_persistence_diagrams(topo, epoch)
                    
            except Exception as e:
                print(f"Topology analysis failed at epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
    
    plot.final_save()
    monitor.save_states('grokking_topology_states.npz')
    
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('grokking_topology_results.csv', index=False)
    
    print("\n=== GROKKING ANALYSIS ===")
    if len(df) > 0 and 'test_acc' in df.columns:
        grok_rows = df[df['test_acc'] > 0.95]
        grok_point = grok_rows.iloc[0]['epoch'] if len(grok_rows) > 0 else None
        
        if grok_point:
            print(f"Grokking occurred at epoch {grok_point}")
            
            pre_grok = df[df['epoch'] < grok_point]
            
            if len(pre_grok) > 2 and 'rf0_mean' in pre_grok.columns:
                from scipy.stats import spearmanr
                
                flat_period = pre_grok[(pre_grok['test_acc'] < 0.5) & (pre_grok['train_acc'] > 0.95)]
                
                if len(flat_period) > 1:
                    print("\n=== FLAT PERIOD (memorization) ===")
                    rf0_start = flat_period['rf0_mean'].iloc[0]
                    rf0_end = flat_period['rf0_mean'].iloc[-1]
                    pct_change = ((rf0_end - rf0_start) / rf0_start) * 100
                    print(f"RF0 mean changed by {pct_change:+.1f}% during flat period")
                    print(f"Start: {rf0_start:.4f}, End: {rf0_end:.4f}")
                    
                    # Check outlier changes
                    outlier_99_start = flat_period['rf0_99th'].iloc[0]
                    outlier_99_end = flat_period['rf0_99th'].iloc[-1]
                    outlier_change = ((outlier_99_end - outlier_99_start) / outlier_99_start) * 100
                    print(f"Top 1% (99th percentile) changed by {outlier_change:+.1f}%")
                    print(f"Start: {outlier_99_start:.4f}, End: {outlier_99_end:.4f}")
                    
                    corr, pval = spearmanr(flat_period['epoch'], flat_period['rf0_mean'])
                    print(f"Correlation RF0 vs epoch during flat: r={corr:.3f}, p={pval:.3f}")
        else:
            print("No grokking observed (test accuracy never exceeded 0.95)")
    
    plot.close()
    return df, monitor.topology_states

if __name__ == "__main__":
    df, topo_states = train_grokking_with_topology()
    print("\nResults saved to grokking_topology_results.csv")
    print("Topology states saved to grokking_topology_states.npz")