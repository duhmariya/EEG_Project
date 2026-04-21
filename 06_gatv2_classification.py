"""
EpiConnectome — GATv2 Seizure Severity Classification
Graph Attention Network v2 on source-level wPLI connectivity matrices

Task:
    Binary classification: High vs Low seizure connectivity severity
    Label: median-split of mean wPLI per seizure per frequency band

Input:
    18×18 wPLI connectivity matrices from dSPM source-level analysis
    Each seizure = one graph (18 nodes, 18×18 edges)

Model:
    GATv2 — Graph Attention Network v2 (Brody et al., 2022)
    2 attention layers + global mean pooling + binary classifier

Output:
    - Classification accuracy, precision, recall, F1
    - Attention weights per ROI connection
    - Feature importance visualization

Usage:
    pip install torch torch-geometric
    python 06_gatv2_classification.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              roc_auc_score)
from sklearn.preprocessing import LabelEncoder

# ── PyTorch + PyG imports ──────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    print(f"✅ PyTorch {torch.__version__} loaded")
except ImportError:
    print("❌ Missing packages. Install with:")
    print("   pip install torch torch-geometric")
    print("   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html")
    exit(1)

# ══════════════════════════════════════════════════════════════
#  CHANGE THESE PATHS FOR YOUR MACHINE
# ══════════════════════════════════════════════════════════════
DSPM_RESULTS_DIR = r"C:\Users\mariy\Desktop\Siena\dSPM_Results"
SOURCE_ATLAS     = r"C:\Users\mariy\Desktop\Siena\dSPM_Results\SOURCE_connectivity_atlas.xlsx"
OUTPUT_DIR       = r"C:\Users\mariy\Desktop\Siena\GATv2_Results"
# ══════════════════════════════════════════════════════════════

# ── Hyperparameters ───────────────────────────────────────────
FREQ_BANDS   = ['theta', 'alpha', 'beta']
N_ROIS       = 18
HIDDEN_DIM   = 32
HEADS        = 4         # GAT attention heads
DROPOUT      = 0.3
LR           = 0.005
EPOCHS       = 100
K_FOLDS      = 5
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD CONNECTIVITY MATRICES
# ════════════════════════════════════════════════════════════════

def load_connectivity_matrices(atlas_path, results_dir):
    """
    Load 18×18 wPLI matrices for each seizure from individual Excel files.
    Returns list of dicts with keys: subject, band, matrix, mean_wpli
    """
    atlas = pd.read_excel(atlas_path)
    print(f"Atlas: {len(atlas)} rows, {atlas['Subject'].nunique()} subjects")

    dataset = []
    missing = 0

    for _, row in atlas.iterrows():
        subject = row['Subject']
        band    = row['Band']

        excel_path = os.path.join(
            results_dir, subject,
            f'{subject}_source_connectivity.xlsx')

        if not os.path.exists(excel_path):
            missing += 1
            continue

        try:
            df = pd.read_excel(
                excel_path,
                sheet_name=band.capitalize(),
                index_col=0)
            matrix = df.values.astype(np.float32)

            if matrix.shape[0] != N_ROIS:
                # Pad or trim to N_ROIS
                n = min(matrix.shape[0], N_ROIS)
                m = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
                m[:n, :n] = matrix[:n, :n]
                matrix = m

            dataset.append({
                'subject':   subject,
                'band':      band,
                'matrix':    matrix,
                'mean_wpli': float(row['Mean_wPLI']),
                'roi_names': df.index.tolist()[:N_ROIS],
            })
        except Exception as e:
            print(f"   ⚠️  {subject} {band}: {e}")
            missing += 1

    print(f"Loaded: {len(dataset)} samples, Missing: {missing}")
    return dataset


# ════════════════════════════════════════════════════════════════
# STEP 2 — CREATE GRAPH DATASET
# ════════════════════════════════════════════════════════════════

def matrix_to_graph(matrix, label, threshold=0.1):
    """
    Convert 18×18 wPLI matrix to PyG graph.

    Nodes: 18 ROIs
    Node features: row-wise mean connectivity (18-dim degree vector)
    Edges: all pairs above threshold
    Edge weights: wPLI values
    Label: 0 (low) or 1 (high) severity
    """
    n = matrix.shape[0]

    # Node features: connectivity profile of each ROI
    # Use [mean, max, std] of each row as node features
    node_features = np.stack([
        np.mean(matrix, axis=1),
        np.max(matrix, axis=1),
        np.std(matrix, axis=1),
    ], axis=1).astype(np.float32)  # (18, 3)

    # Build edges — include all connections above threshold
    edge_index = []
    edge_attr  = []

    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > threshold:
                edge_index.append([i, j])
                edge_attr.append(matrix[i, j])

    if len(edge_index) == 0:
        # Fallback: connect all nodes with low-weight edges
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_index.append([i, j])
                    edge_attr.append(0.01)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    x          = torch.tensor(node_features, dtype=torch.float)
    y          = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_graph_dataset(dataset):
    """
    Build PyG dataset with binary labels from median split.
    Separately for each frequency band and combined.
    """
    graphs_by_band = {band: [] for band in FREQ_BANDS}
    graphs_all     = []

    # Compute median per band for labeling
    medians = {}
    for band in FREQ_BANDS:
        vals = [d['mean_wpli'] for d in dataset if d['band'] == band]
        medians[band] = np.median(vals)
        print(f"   {band}: median wPLI = {medians[band]:.3f}, "
              f"n={len(vals)}, high={sum(v >= medians[band] for v in vals)}, "
              f"low={sum(v < medians[band] for v in vals)}")

    for d in dataset:
        band   = d['band']
        label  = 1 if d['mean_wpli'] >= medians[band] else 0
        graph  = matrix_to_graph(d['matrix'], label)
        graph.subject = d['subject']
        graph.band    = band
        graphs_by_band[band].append(graph)
        graphs_all.append(graph)

    return graphs_by_band, graphs_all, medians


# ════════════════════════════════════════════════════════════════
# STEP 3 — GATv2 MODEL
# ════════════════════════════════════════════════════════════════

class GATv2Classifier(nn.Module):
    """
    GATv2 for graph-level binary classification.

    Architecture:
        GATv2Conv(3 → 32, heads=4)    # Layer 1: local ROI attention
        GATv2Conv(128 → 16, heads=2)  # Layer 2: higher-order patterns
        GlobalMeanPool                 # Graph-level representation
        Linear(32 → 2)                # Binary classifier
    """

    def __init__(self, in_channels=3, hidden=HIDDEN_DIM,
                 heads=HEADS, dropout=DROPOUT):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels, hidden,
            heads=heads, dropout=dropout,
            edge_dim=1, concat=True)

        self.conv2 = GATv2Conv(
            hidden * heads, hidden // 2,
            heads=2, dropout=dropout,
            edge_dim=1, concat=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch,
                return_attention=False):

        # Layer 1
        if return_attention:
            x, (ei1, a1) = self.conv1(
                x, edge_index, edge_attr,
                return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2
        if return_attention:
            x, (ei2, a2) = self.conv2(
                x, edge_index, edge_attr,
                return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classify
        out = self.classifier(x)

        if return_attention:
            return out, (ei1, a1), (ei2, a2)
        return out


# ════════════════════════════════════════════════════════════════
# STEP 4 — TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            out  = model(batch.x, batch.edge_index,
                         batch.edge_attr, batch.batch)
            prob = F.softmax(out, dim=1)[:, 1].numpy()
            pred = out.argmax(dim=1).numpy()
            preds.extend(pred)
            labels.extend(batch.y.numpy())
            probs.extend(prob)
    return np.array(preds), np.array(labels), np.array(probs)


def run_cross_validation(graphs, band_name, k=K_FOLDS):
    """Run k-fold cross validation and return metrics."""
    print(f"\n── {band_name.upper()} band — {k}-fold CV ──────────────────")

    if len(graphs) < k * 2:
        print(f"   ⚠️  Too few samples ({len(graphs)}) for {k}-fold CV, using 3-fold")
        k = 3

    labels   = [g.y.item() for g in graphs]
    skf      = StratifiedKFold(n_splits=k, shuffle=True,
                               random_state=RANDOM_SEED)
    indices  = np.arange(len(graphs))

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(
            skf.split(indices, labels), 1):

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs  = [graphs[i] for i in test_idx]

        train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=8)

        model     = GATv2Classifier()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5)

        # Train
        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

        # Evaluate
        preds, true, probs = evaluate(model, test_loader)
        acc  = accuracy_score(true, preds)
        prec = precision_score(true, preds, zero_division=0)
        rec  = recall_score(true, preds, zero_division=0)
        f1   = f1_score(true, preds, zero_division=0)
        try:
            auc = roc_auc_score(true, probs)
        except:
            auc = float('nan')

        fold_metrics.append({
            'fold': fold, 'acc': acc, 'prec': prec,
            'rec': rec, 'f1': f1, 'auc': auc
        })
        print(f"   Fold {fold}: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")

    df = pd.DataFrame(fold_metrics)
    print(f"   Mean: acc={df['acc'].mean():.3f}±{df['acc'].std():.3f} "
          f"f1={df['f1'].mean():.3f}±{df['f1'].std():.3f} "
          f"auc={df['auc'].mean():.3f}±{df['auc'].std():.3f}")

    return df


# ════════════════════════════════════════════════════════════════
# STEP 5 — ATTENTION VISUALIZATION
# ════════════════════════════════════════════════════════════════

def get_attention_weights(model, graphs, roi_names):
    """Extract attention weights from trained model."""
    loader = DataLoader(graphs, batch_size=len(graphs))
    model.eval()

    with torch.no_grad():
        for batch in loader:
            _, (ei1, a1), (ei2, a2) = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, return_attention=True)

    # Average attention across heads
    attn = a1.mean(dim=1).numpy()  # (n_edges,)
    edge_index = ei1.numpy()

    # Aggregate attention per node (ROI)
    n_rois = len(roi_names)
    node_attn = np.zeros(n_rois)
    counts    = np.zeros(n_rois)

    for k, (src, dst) in enumerate(edge_index.T):
        src = src % n_rois
        dst = dst % n_rois
        if src < n_rois and dst < n_rois:
            node_attn[src] += attn[k]
            node_attn[dst] += attn[k]
            counts[src] += 1
            counts[dst] += 1

    counts = np.where(counts == 0, 1, counts)
    node_attn = node_attn / counts

    return node_attn


def plot_attention(node_attn, roi_names, band, output_dir):
    """Bar chart of ROI attention weights."""
    short = [r.replace('ROI', 'R').replace('_LH', '-L').replace('_RH', '-R')
             for r in roi_names]

    colors = ['#00C9A7' if a >= np.median(node_attn) else '#1E3A6A'
              for a in node_attn]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(short, node_attn, color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.axhline(np.median(node_attn), color='#FF8C42',
               linestyle='--', linewidth=1.5, label='Median attention')

    ax.set_title(f'GATv2 ROI Attention Weights — {band.capitalize()} Band',
                 fontsize=13, fontweight='bold', color='white')
    ax.set_xlabel('ROI', fontsize=11, color='white')
    ax.set_ylabel('Mean Attention Weight', fontsize=11, color='white')
    ax.tick_params(axis='x', rotation=45, colors='white', labelsize=8)
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('#0F2040')
    fig.patch.set_facecolor('#0A1628')
    ax.spines[:].set_color('#1E3A6A')
    ax.legend(facecolor='#152A52', labelcolor='white')

    high_patch = mpatches.Patch(color='#00C9A7', label='Above median')
    low_patch  = mpatches.Patch(color='#1E3A6A', label='Below median')
    ax.legend(handles=[high_patch, low_patch],
              facecolor='#152A52', labelcolor='white')

    plt.tight_layout()
    path = os.path.join(output_dir, f'attention_{band}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='#0A1628')
    plt.close()
    print(f"   ✅ Saved: attention_{band}.png")
    return path


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EpiConnectome — GATv2 Seizure Severity Classification")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    print("\n── Loading connectivity matrices ───────────────────────")
    dataset = load_connectivity_matrices(SOURCE_ATLAS, DSPM_RESULTS_DIR)

    if len(dataset) == 0:
        print("❌ No data loaded. Check paths.")
        return

    # Get ROI names from first sample
    roi_names = dataset[0]['roi_names'] if dataset else [f'ROI{i}' for i in range(N_ROIS)]
    print(f"ROI names ({len(roi_names)}): {roi_names[:4]}...")

    # ── Build graph dataset ───────────────────────────────────
    print("\n── Building graph dataset ──────────────────────────────")
    graphs_by_band, graphs_all, medians = build_graph_dataset(dataset)

    print(f"\n   Total graphs: {len(graphs_all)}")
    for band in FREQ_BANDS:
        print(f"   {band}: {len(graphs_by_band[band])} graphs")

    # ── Cross-validation per band ─────────────────────────────
    print("\n── Running GATv2 cross-validation ──────────────────────")
    all_metrics = {}

    for band in FREQ_BANDS:
        graphs = graphs_by_band[band]
        if len(graphs) < 6:
            print(f"   ⚠️  {band}: too few samples, skipping")
            continue
        metrics_df = run_cross_validation(graphs, band)
        all_metrics[band] = metrics_df

    # ── Combined (all bands) ──────────────────────────────────
    print("\n── Combined (all bands) ────────────────────────────────")
    if len(graphs_all) >= 10:
        metrics_combined = run_cross_validation(graphs_all, 'combined')
        all_metrics['combined'] = metrics_combined

    # ── Save metrics ──────────────────────────────────────────
    metrics_path = os.path.join(OUTPUT_DIR, 'GATv2_metrics.xlsx')
    with pd.ExcelWriter(metrics_path, engine='openpyxl') as writer:
        for name, df in all_metrics.items():
            df.to_excel(writer, sheet_name=name, index=False)

        # Summary sheet
        summary = []
        for name, df in all_metrics.items():
            summary.append({
                'Band': name,
                'Mean_Accuracy': round(df['acc'].mean(), 3),
                'Std_Accuracy':  round(df['acc'].std(), 3),
                'Mean_F1':       round(df['f1'].mean(), 3),
                'Std_F1':        round(df['f1'].std(), 3),
                'Mean_AUC':      round(df['auc'].mean(), 3),
                'Std_AUC':       round(df['auc'].std(), 3),
            })
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n📊 Metrics saved → {metrics_path}")

    # ── Train final model + attention visualization ───────────
    print("\n── Training final model for attention analysis ─────────")
    for band in FREQ_BANDS:
        graphs = graphs_by_band[band]
        if len(graphs) < 6:
            continue

        print(f"\n   {band.upper()} band...")
        loader    = DataLoader(graphs, batch_size=len(graphs), shuffle=True)
        model     = GATv2Classifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            train_epoch(model, loader, optimizer, criterion)

        node_attn = get_attention_weights(model, graphs, roi_names)
        plot_attention(node_attn, roi_names, band, OUTPUT_DIR)

        # Print top 5 most attended ROIs
        top5 = np.argsort(node_attn)[::-1][:5]
        print(f"   Top 5 attended ROIs ({band}):")
        for idx in top5:
            name = roi_names[idx] if idx < len(roi_names) else f'ROI{idx}'
            print(f"      {name}: {node_attn[idx]:.4f}")

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GATv2 COMPLETE")
    print(f"{'='*60}")
    print("\n── Summary ──")
    for name, df in all_metrics.items():
        print(f"   {name:10s}: "
              f"acc={df['acc'].mean():.3f}±{df['acc'].std():.3f}  "
              f"f1={df['f1'].mean():.3f}±{df['f1'].std():.3f}  "
              f"auc={df['auc'].mean():.3f}±{df['auc'].std():.3f}")

    print(f"\n📁 All results saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
