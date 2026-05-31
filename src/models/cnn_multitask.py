import torch

import torch.nn as nn

import logging

try:
    from .cnn_view_classifier import get_cnn_backbone
except ImportError:
    logging.error("Could not import get_cnn_backbone from .cnn_view_classifier. Ensure it exists and defines video models.")

    def get_cnn_backbone(name="resnet18", pretrained=True):
        import torchvision.models as models # Local import for fallback
        logging.warning("Using fallback get_cnn_backbone - Video models might not be supported!")
        if name == "resnet18":
             weights = models.ResNet18_Weights.DEFAULT if pretrained else None
             model = models.resnet18(weights=weights)
             num_features = model.fc.in_features; model.fc = nn.Identity()
             return model, num_features
        else: raise ValueError(f"Unsupported backbone in fallback: {name}")


class MultiTaskCNN(nn.Module):

    def __init__(self, backbone_name="resnet18", pretrained=True, num_view_classes=2, dropout_rate=0.3):
        super().__init__()
        self.backbone_name = backbone_name

        try:
            self.backbone, num_features = get_cnn_backbone(backbone_name, pretrained)
        except Exception as e:
            logging.error(f"Failed to initialize backbone '{backbone_name}' using get_cnn_backbone: {e}", exc_info=True)
            raise e

        self.dropout = nn.Dropout(p=dropout_rate)


        self.view_classifier = nn.Linear(num_features, num_view_classes)
        self.ef_regressor = nn.Linear(num_features, 1)

        logging.info(f"MultiTaskCNN initialized with {backbone_name} backbone.")
        logging.info(f"View Classification Head: Input={num_features}, Output={num_view_classes}")
        logging.info(f"EF Regression Head: Input={num_features}, Output=1")
        logging.info(f"Dropout rate before heads: {dropout_rate}")

    def forward(self, x):

        logging.debug(f"DEBUG MultiTaskCNN forward(): Input x shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")

        try:
            features = self.backbone(x)
        except Exception as e:
            logging.error(f"Error during backbone forward pass with input shape {x.shape}: {e}", exc_info=True)
            raise e

        if features.shape[0] != x.shape[0]:

            logging.warning(f"Batch size mismatch after backbone! Input: {x.shape[0]}, Output: {features.shape[0]}. This might indicate incorrect handling of sequence data for this backbone type.")

            if features.shape[0] % x.shape[0] == 0:
                 num_frames_in_batch = features.shape[0] // x.shape[0]
                 num_features_bb = features.shape[-1]
                 logging.warning(f"Attempting to aggregate features assuming {num_frames_in_batch} frames per sample...")
                 try:
                     features_view = features.view(x.shape[0], num_frames_in_batch, num_features_bb)
                     features = features_view.mean(dim=1) # Average across frames
                     logging.info(f"Aggregation successful. New feature shape: {features.shape}")
                 except Exception as agg_e:
                     logging.error(f"Failed to aggregate features despite batch size mismatch: {agg_e}")

                     raise RuntimeError(f"Cannot reconcile feature shape {features.shape} with batch size {x.shape[0]}") from agg_e

        dropped_features = self.dropout(features)
        view_logits = self.view_classifier(dropped_features)
        ef_prediction = self.ef_regressor(dropped_features)
        
        # Squeeze and Clamp EF prediction (Safety Requirement: 10% - 95%)
        ef_prediction = ef_prediction.squeeze(-1)
        ef_prediction = torch.clamp(ef_prediction, min=10.0, max=95.0)

        return view_logits, ef_prediction