import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import argparse
import json
import os
from datetime import datetime
import numpy as np

from utils.data_loader import get_peptides_loaders
from utils.metrics import evaluate_model, compute_ap
from models.transformer import GraphTransformer

def train_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        out = model(data)
        
        loss = criterion(out, data.y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
        all_preds.append(torch.sigmoid(out).detach())
        all_labels.append(data.y.detach())
        
        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    avg_loss = total_loss / len(loader.dataset)
    train_ap = compute_ap(all_labels, all_preds)
    
    return avg_loss, train_ap

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_ap, args, filename):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_ap': best_val_ap,
        'args': vars(args)
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved to {filename}')

def main():
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pe_type', type=str, default='laplacian', choices=['laplacian', 'random_walk', 'none'])
    parser.add_argument('--pe_dim', type=int, default=16)
    parser.add_argument('--use_edge_features', action='store_true', default=True)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=50)
    
    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='peptides-transformer')
    parser.add_argument('--exp_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'transformer_h{args.hidden_dim}_l{args.num_layers}_heads{args.num_heads}_{args.pe_type}_{timestamp}'
    
    # Create checkpoint directory
    checkpoint_dir = f'checkpoints/{args.exp_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save args
    with open(f'{checkpoint_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.project_name, name=args.exp_name, config=vars(args))
    
    print("="*50)
    print(f"Experiment: {args.exp_name}")
    print(f"Device: {args.device}")
    print("="*50)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_node_features, num_edge_features, num_tasks = get_peptides_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = GraphTransformer(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_tasks=num_tasks,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pe_type=args.pe_type,
        pe_dim=args.pe_dim,
        use_edge_features=args.use_edge_features
    ).to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    best_val_ap = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_ap = train_epoch(model, train_loader, optimizer, args.device, epoch)
        
        # Validate
        val_ap, val_loss = evaluate_model(model, val_loader, args.device)
        
        # Step scheduler
        scheduler.step()
        
        # Logging
        print(f'\nEpoch {epoch}: Train Loss: {train_loss:.4f}, Train AP: {train_ap:.4f}')
        print(f'          Val Loss: {val_loss:.4f}, Val AP: {val_ap:.4f}')
        print(f'          LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_ap': train_ap,
                'val_loss': val_loss,
                'val_ap': val_ap,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_ap, args,
                f'{checkpoint_dir}/best_model.pt'
            )
            print(f'*** New best validation AP: {best_val_ap:.4f} ***')
        else:
            patience_counter += 1
        
        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_ap, args,
                f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pt'
            )
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch}')
            break
    
    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print("Loading best model and evaluating on test set...")
    print("="*50)
    
    checkpoint = torch.load(f'{checkpoint_dir}/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_ap, test_loss = evaluate_model(model, test_loader, args.device)
    
    print(f'\nTest AP: {test_ap:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    
    if args.use_wandb:
        wandb.log({
            'test_ap': test_ap,
            'test_loss': test_loss,
            'best_val_ap': best_val_ap
        })
        wandb.finish()
    
    # Save final results
    results = {
        'best_val_ap': best_val_ap,
        'test_ap': test_ap,
        'test_loss': test_loss,
        'num_params': num_params,
        'final_epoch': epoch
    }
    
    with open(f'{checkpoint_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {checkpoint_dir}/results.json")

if __name__ == '__main__':
    main()