
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
import os

def plot_losses(history, save_path=None):
    """Plots training and validation loss curves."""
    try:

        if not history or 'train_loss' not in history or not history['train_loss'] or 'val_loss' not in history or not history['val_loss']:
            logging.warning("Could not plot losses: Missing or empty 'train_loss' or 'val_loss' in history.")
            return

        epochs = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logging.debug(f"Loss plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot/save loss curves: {e}", exc_info=True)


def plot_metrics(history, metrics=None, save_path=None):

    try:
        if not history: logging.warning("History object is empty, cannot plot metrics."); return

        if metrics is None:

             metrics = sorted([k for k in history.keys() if k.startswith('val_') and 'loss' not in k])
             if not metrics: logging.warning("No validation metrics (excluding loss) found in history to plot."); return
             logging.debug(f"Auto-detected metrics to plot: {metrics}")

        first_valid_metric = None
        for m in metrics:
            if m in history and history[m]:
                first_valid_metric = m
                break
        if first_valid_metric is None:
             logging.warning("None of the specified metrics have data in history. Cannot plot metrics.")
             return

        epochs = range(1, len(history[first_valid_metric]) + 1)
        plt.figure(figsize=(10, 5))
        plotted_something = False

        for metric in metrics:
            if metric in history and history[metric]:

                 label_name = metric.replace('val_', '').replace('_', ' ').title()
                 try:
                     plt.plot(epochs, history[metric], 'o-', label=label_name)
                     plotted_something = True
                 except Exception as e:
                     logging.error(f"Could not plot metric '{metric}': {e}")
            else:
                logging.warning(f"Metric '{metric}' not found in history or is empty, skipping.")

        if not plotted_something:
             logging.warning("No valid metrics were plotted.")
             plt.close()
             return

        plt.title('Validation Metrics over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')

        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logging.debug(f"Metrics plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot/save metric curves: {e}", exc_info=True)

def plot_confusion_matrix(y_true_tensor, y_pred_logits_tensor, class_names=None, save_path=None, normalize=None):

    try:

        if not isinstance(y_true_tensor, torch.Tensor) or not isinstance(y_pred_logits_tensor, torch.Tensor):
             logging.error("plot_confusion_matrix expects tensor inputs.")
             return
        if y_true_tensor.numel() == 0 or y_pred_logits_tensor.numel() == 0:
             logging.warning("plot_confusion_matrix skipped: Empty input tensors.")
             return

        if class_names:
            num_classes = len(class_names)
        elif isinstance(y_pred_logits_tensor, torch.Tensor) and y_pred_logits_tensor.ndim > 1:
             num_classes = y_pred_logits_tensor.shape[1] # Get from logit dimension
             logging.debug(f"Determined num_classes={num_classes} from logits shape.")
             if class_names is None: # If names weren't passed, use indices as labels
                 class_names = [str(i) for i in range(num_classes)]
                 logging.warning(f"Using index labels {class_names} for confusion matrix as class_names were not provided.")
        else:
             logging.error("plot_confusion_matrix: Cannot determine number of classes from inputs.")
             return

        if num_classes <= 0:
             logging.error(f"plot_confusion_matrix: Invalid number of classes determined ({num_classes}).")
             return
        all_possible_labels = list(range(num_classes)) # e.g., [0, 1] if num_classes=2

        y_pred_labels_tensor = torch.argmax(y_pred_logits_tensor, dim=1)

        y_true_np = y_true_tensor.detach().cpu().numpy()
        y_pred_np = y_pred_labels_tensor.detach().cpu().numpy()

        cm = confusion_matrix(y_true_np, y_pred_np,
                              labels=all_possible_labels,
                              normalize=normalize)


        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)

        fig, ax = plt.subplots(figsize=(8, 6) if num_classes<10 else (10,8))
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f' if normalize else 'd')
        ax.set_title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logging.debug(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    except Exception as e:
        logging.error(f"Failed to plot/save confusion matrix: {e}", exc_info=True)

def plot_ef_scatter(y_true_tensor, y_pred_tensor, save_path=None):

    try:

        if not isinstance(y_true_tensor, torch.Tensor) or not isinstance(y_pred_tensor, torch.Tensor):
             logging.error("plot_ef_scatter expects tensor inputs.")
             return
        if y_true_tensor.numel() == 0 or y_pred_tensor.numel() == 0:
             logging.warning("plot_ef_scatter skipped: Empty input tensors.")
             return


        y_true_np = y_true_tensor.detach().cpu().numpy().flatten() # Flatten just in case
        y_pred_np = y_pred_tensor.detach().cpu().numpy().flatten()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true_np, y_pred_np, alpha=0.6, label='Predictions')

        min_val = min(np.min(y_true_np), np.min(y_pred_np)) if len(y_true_np)>0 else 0
        max_val = max(np.max(y_true_np), np.max(y_pred_np)) if len(y_true_np)>0 else 100
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x (Ideal)') # Add identity line
        ax.set_xlabel("True EF (%)")
        ax.set_ylabel("Predicted EF (%)")
        ax.set_title("EF Prediction: True vs. Predicted")
        ax.legend()
        ax.grid(True)
        pad = (max_val - min_val) * 0.05 # Add 5% padding
        ax.set_xlim(min_val - pad, max_val + pad)
        ax.set_ylim(min_val - pad, max_val + pad)


        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logging.debug(f"EF scatter plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    except Exception as e:
        logging.error(f"Failed to plot/save EF scatter plot: {e}", exc_info=True)

