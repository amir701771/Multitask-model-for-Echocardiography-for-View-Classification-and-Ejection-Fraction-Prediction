
import torch
from sklearn.metrics import (accuracy_score, mean_absolute_error, mean_squared_error,
                             r2_score, cohen_kappa_score, f1_score)
import numpy as np
import logging

def calculate_accuracy(y_pred_logits, y_true):
    """Calculates classification accuracy."""

    if not isinstance(y_pred_logits, torch.Tensor) or not isinstance(y_true, torch.Tensor) or y_true.numel() == 0:

        return 0.0
    try:
        y_pred_labels = torch.argmax(y_pred_logits, dim=1)

        y_pred_np = y_pred_labels.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        return accuracy_score(y_true_np, y_pred_np)
    except Exception as e:
        logging.error(f"Accuracy calculation error: {e}", exc_info=True)
        return 0.0


def calculate_f1(y_pred_logits, y_true, average='weighted', labels=None):
    """Calculates F1 score, optionally using a predefined list of labels."""
    if not isinstance(y_pred_logits, torch.Tensor) or not isinstance(y_true, torch.Tensor) or y_true.numel() == 0:

        return 0.0
    try:
        y_pred_labels = torch.argmax(y_pred_logits, dim=1)
        y_pred_np = y_pred_labels.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        return f1_score(y_true_np, y_pred_np, average=average, labels=labels, zero_division=0)
    except Exception as e:
        logging.error(f"F1 ({average}) calculation error: {e}", exc_info=True)
        return 0.0

def calculate_kappa(y_pred_logits, y_true, labels=None):
    """Calculates Cohen's Kappa, optionally using a predefined list of labels."""
    if not isinstance(y_pred_logits, torch.Tensor) or not isinstance(y_true, torch.Tensor) or y_true.numel() == 0:

        return 0.0
    try:
        y_pred_labels = torch.argmax(y_pred_logits, dim=1)
        y_pred_np = y_pred_labels.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        if len(np.unique(y_true_np)) <= 1 and len(np.unique(y_pred_np)) <= 1 and len(np.unique(np.concatenate((y_true_np,y_pred_np)))) <= 1 :

            return 1.0 if np.array_equal(y_true_np, y_pred_np) else 0.0


        return cohen_kappa_score(y_true_np, y_pred_np, labels=labels)
    except ValueError as ve:

         logging.warning(f"Kappa calculation ValueError: {ve}. Check if provided 'labels' match data.")
         return 0.0
    except Exception as e:
        logging.error(f"Kappa calculation error: {e}", exc_info=True)
        return 0.0

def calculate_mae(y_pred, y_true):
    """Calculates Mean Absolute Error."""
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor) or y_true.numel() == 0:

        return float('inf')
    try:

        y_pred_np = y_pred.cpu().numpy().reshape(-1)
        y_true_np = y_true.cpu().numpy().reshape(-1)
        return mean_absolute_error(y_true_np, y_pred_np)
    except Exception as e:
        logging.error(f"MAE calculation error: {e}", exc_info=True)
        return float('inf')

def calculate_mse(y_pred, y_true):
    """Calculates Mean Squared Error."""
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor) or y_true.numel() == 0:

        return float('inf')
    try:
        y_pred_np = y_pred.cpu().numpy().reshape(-1)
        y_true_np = y_true.cpu().numpy().reshape(-1)
        return mean_squared_error(y_true_np, y_pred_np)
    except Exception as e:
        logging.error(f"MSE calculation error: {e}", exc_info=True)
        return float('inf')

def calculate_rmse(y_pred, y_true):
    """Calculates Root Mean Squared Error."""
    mse = calculate_mse(y_pred, y_true)

    return np.sqrt(mse) if mse != float('inf') else float('inf')

def calculate_r2(y_pred, y_true):
    """Calculates R-squared."""
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor) or y_true.numel() < 2: # R2 needs at least 2 samples

        return -float('inf')
    try:
        y_pred_np = y_pred.cpu().numpy().reshape(-1)
        y_true_np = y_true.cpu().numpy().reshape(-1)

        if np.var(y_true_np) < 1e-6:

            pass
        return r2_score(y_true_np, y_pred_np)
    except Exception as e:
        logging.error(f"R2 calculation error: {e}", exc_info=True)
        return -float('inf')

def compute_metrics_multitask(view_logits, ef_preds, view_labels, ef_labels, config={}):

    metrics = {}
    all_possible_labels = None # Initialize

    num_classes = config.get('num_views', None)
    if isinstance(num_classes, int) and num_classes > 0:
        all_possible_labels = list(range(num_classes))

    else:
        logging.warning(f"Could not determine 'num_views' from config ({config.get('num_views')}). "
                        "F1 and Kappa metrics might be less robust if not all classes are present.")

    vl = view_logits.detach() if isinstance(view_logits, torch.Tensor) else None
    efp = ef_preds.detach() if isinstance(ef_preds, torch.Tensor) else None
    vlb = view_labels.detach() if isinstance(view_labels, torch.Tensor) else None
    eflb = ef_labels.detach() if isinstance(ef_labels, torch.Tensor) else None

    if vl is not None and vlb is not None and vlb.numel() > 0:
        metrics['accuracy'] = calculate_accuracy(vl, vlb)

        metrics['f1_weighted'] = calculate_f1(vl, vlb, average='weighted', labels=all_possible_labels)
        metrics['kappa'] = calculate_kappa(vl, vlb, labels=all_possible_labels)
    else:
        if view_logits is not None or view_labels is not None:
             logging.debug("Classification metrics defaulted: Invalid logits/labels or empty.")
        metrics['accuracy'] = 0.0
        metrics['f1_weighted'] = 0.0
        metrics['kappa'] = 0.0


    if efp is not None and eflb is not None and eflb.numel() > 0:
        metrics['mae'] = calculate_mae(efp, eflb)
        metrics['mse'] = calculate_mse(efp, eflb)
        metrics['rmse'] = calculate_rmse(efp, eflb)
        metrics['r2'] = calculate_r2(efp, eflb)
    else:
        if ef_preds is not None or ef_labels is not None:
            logging.debug("Regression metrics defaulted: Invalid preds/labels or empty.")
        metrics['mae'] = float('inf')
        metrics['mse'] = float('inf')
        metrics['rmse'] = float('inf')
        metrics['r2'] = -float('inf')

    return metrics