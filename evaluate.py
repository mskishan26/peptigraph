import torch
import argparse
import json
import os
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

from utils.data_loader import get_peptides_loaders
from utils.metrics import evaluate_model, compute_ap
from utils.visualization import plot_per_class_performance, plot_confusion_style_heatmap
from models.transformer import GraphTransformer

# Class names for Peptides-func
CLASS_NAMES = [
    'Antifungal',
    'Cell-cell communication',
    'Anticancer',
    'Drug delivery vehicle',
    'Antimicrobial',
    'Antiviral',
    'Antihypertensive',
    'Antibacterial',
    'Antiparasitic',
    'Toxic'
]

def compute_detailed_metrics(y_true, y_pred):
    """
    Compute detailed per-class metrics
    
    Returns:
        dict with per-class AP, ROC-AUC, and other metrics
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    num_classes = y_true.shape[1]
    
    metrics = {
        'ap_per_class': [],
        'auroc_per_class': [],
        'positive_rate': [],
        'mean_pred_prob': []
    }
    
    for i in range(num_classes):
        # Skip if all same label
        if len(np.unique(y_true[:, i])) == 1:
            metrics['ap_per_class'].append(np.nan)
            metrics['auroc_per_class'].append(np.nan)
        else:
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
            metrics['ap_per_class'].append(ap)
            metrics['auroc_per_class'].append(auroc)
        
        metrics['positive_rate'].append(y_true[:, i].mean())
        metrics['mean_pred_prob'].append(y_pred[:, i].mean())
    
    metrics['mean_ap'] = np.nanmean(metrics['ap_per_class'])
    metrics['mean_auroc'] = np.nanmean(metrics['auroc_per_class'])
    
    return metrics


def analyze_predictions(model, loader, device, output_dir):
    """
    Analyze model predictions in detail
    """
    print("\n" + "="*50)
    print("Analyzing Predictions")
    print("="*50)
    
    # Get predictions
    ap, loss, all_preds, all_labels = evaluate_model(
        model, loader, device, return_predictions=True
    )
    
    # Compute detailed metrics
    metrics = compute_detailed_metrics(all_labels, all_preds)
    
    # Print per-class results
    print("\nPer-Class Results:")
    print("-" * 70)
    print(f"{'Class':<30} {'AP':>8} {'AUROC':>8} {'Pos Rate':>10} {'Pred Prob':>10}")
    print("-" * 70)
    
    for i, class_name in enumerate(CLASS_NAMES):
        ap_score = metrics['ap_per_class'][i]
        auroc_score = metrics['auroc_per_class'][i]
        pos_rate = metrics['positive_rate'][i]
        pred_prob = metrics['mean_pred_prob'][i]
        
        ap_str = f"{ap_score:.4f}" if not np.isnan(ap_score) else "N/A"
        auroc_str = f"{auroc_score:.4f}" if not np.isnan(auroc_score) else "N/A"
        
        print(f"{class_name:<30} {ap_str:>8} {auroc_str:>8} {pos_rate:>10.4f} {pred_prob:>10.4f}")
    
    print("-" * 70)
    print(f"{'Mean':<30} {metrics['mean_ap']:>8.4f} {metrics['mean_auroc']:>8.4f}")
    print("-" * 70)
    
    # Save metrics
    metrics_dict = {
        'class_names': CLASS_NAMES,
        'ap_per_class': [float(x) if not np.isnan(x) else None for x in metrics['ap_per_class']],
        'auroc_per_class': [float(x) if not np.isnan(x) else None for x in metrics['auroc_per_class']],
        'positive_rate': [float(x) for x in metrics['positive_rate']],
        'mean_pred_prob': [float(x) for x in metrics['mean_pred_prob']],
        'mean_ap': float(metrics['mean_ap']),
        'mean_auroc': float(metrics['mean_auroc'])
    }
    
    with open(f'{output_dir}/detailed_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\nDetailed metrics saved to {output_dir}/detailed_metrics.json")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Per-class performance
    valid_indices = [i for i, x in enumerate(metrics['ap_per_class']) if not np.isnan(x)]
    valid_classes = [CLASS_NAMES[i] for i in valid_indices]
    valid_aps = [metrics['ap_per_class'][i] for i in valid_indices]
    
    plot_per_class_performance(
        valid_classes, valid_aps,
        save_path=f'{output_dir}/per_class_performance.png'
    )
    
    # Co-occurrence heatmap
    plot_confusion_style_heatmap(
        all_labels, all_preds, CLASS_NAMES,
        save_path=f'{output_dir}/class_cooccurrence.png'
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on Peptides-func')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Get model args from checkpoint
    model_args = checkpoint['args']
    
    # Set output directory
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_dir = f'{checkpoint_dir}/evaluation_{args.split}'
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Val AP: {checkpoint['best_val_ap']:.4f}")
    
    # Load data
    print(f"\nLoading {args.split} data...")
    train_loader, val_loader, test_loader, num_node_features, num_edge_features, num_tasks = get_peptides_loaders(
        batch_size=args.batch_size,
        num_workers=4
    )
    
    if args.split == 'train':
        loader = train_loader
    elif args.split == 'val':
        loader = val_loader
    else:
        loader = test_loader
    
    # Initialize model
    print("\nInitializing model...")
    model = GraphTransformer(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_tasks=num_tasks,
        hidden_dim=model_args['hidden_dim'],
        num_layers=model_args['num_layers'],
        num_heads=model_args['num_heads'],
        dropout=model_args['dropout'],
        pe_type=model_args['pe_type'],
        pe_dim=model_args['pe_dim'],
        use_edge_features=model_args['use_edge_features']
    ).to(args.device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    ap, loss = evaluate_model(model, loader, args.device)
    
    print(f"\n{args.split.capitalize()} Results:")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  Loss: {loss:.4f}")
    
    # Detailed analysis
    metrics = analyze_predictions(model, loader, args.device, args.output_dir)
    
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()