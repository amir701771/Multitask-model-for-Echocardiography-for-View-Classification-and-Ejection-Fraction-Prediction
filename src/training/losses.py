
import torch
import torch.nn as nn
import logging

class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        logging.info(f"Using CrossEntropyLossWrapper (reduction={reduction})")

    def forward(self, outputs, targets):

        loss_c = self.loss_fn(outputs, targets.long())
        return loss_c, loss_c, torch.tensor(0.0, device=loss_c.device)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, reduction='mean'):
        super().__init__()
        if not (alpha >= 0 and beta >= 0):
             raise ValueError("Loss weights alpha and beta must be non-negative.")
        if alpha == 0 and beta == 0:
            logging.warning("Both alpha and beta are zero. The loss will always be zero!")

        self.alpha = alpha
        self.beta = beta
        self.classification_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.regression_loss = nn.L1Loss(reduction=reduction) # MAE for EF

        logging.info(f"CombinedLoss initialized: alpha (Class Weight) = {alpha}, beta (Reg Weight) = {beta}")
        logging.info(f"Using Classification Loss: {type(self.classification_loss).__name__}")
        logging.info(f"Using Regression Loss: {type(self.regression_loss).__name__}")

    def forward(self, view_logits, ef_predictions, view_labels, ef_labels):
        ef_labels = ef_labels.float()
        ef_predictions = ef_predictions.float()
        view_labels = view_labels.long()

        loss_c = self.classification_loss(view_logits, view_labels) if self.alpha > 0 else torch.tensor(0.0, device=view_logits.device)
        loss_r = self.regression_loss(ef_predictions, ef_labels) if self.beta > 0 else torch.tensor(0.0, device=ef_predictions.device)

        total_loss = (self.alpha * loss_c) + (self.beta * loss_r)
        return total_loss, loss_c, loss_r # Return components for logging