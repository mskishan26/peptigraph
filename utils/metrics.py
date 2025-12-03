import torch
import numpy as np
from sklearn.metrics import average_precision_score

def compute_ap(y_true, y_pred):
    """
    Compute Average Precision for multi-label classification
    
    Args:
        y_true: [batch_size, num_tasks] true labels
        y_pred: [batch_size, num_tasks] predicted probabilities
        
    Returns:
        ap: average precision score
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    ap_scores = []
    for i in range(y_true.shape[1]):
        # Skip if all labels are the same (no positive or no negative)
        if len(np.unique(y_true[:, i])) == 1:
            continue
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        ap_scores.append(ap)
    
    if len(ap_scores) == 0:
        return 0.0
    
    return np.mean(ap_scores)

def evaluate_model(model, loader, device, return_predictions=False):
    """
    Evaluate model on a dataset
    
    Returns:
        ap: average precision score
        loss: average loss
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            out = model(data)
            loss = criterion(out, data.y)
            
            total_loss += loss.item() * data.num_graphs
            
            all_preds.append(torch.sigmoid(out))
            all_labels.append(data.y)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    ap = compute_ap(all_labels, all_preds)
    avg_loss = total_loss / len(loader.dataset)
    
    if return_predictions:
        return ap, avg_loss, all_preds, all_labels
    
    return ap, avg_loss