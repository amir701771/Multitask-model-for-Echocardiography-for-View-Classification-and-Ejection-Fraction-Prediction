
import torch.nn as nn
import torchvision.models as models

from torchvision.models.video import r2plus1d_18
import logging

def get_cnn_backbone(name="resnet18", pretrained=True):

    model = None
    num_features = 0
    name = name.lower()

    logging.debug(f"Attempting to load backbone: {name} (pretrained={pretrained})")

    if name == "resnet18":

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        num_features = model.fc.in_features # 512
        model.fc = nn.Identity()
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        num_features = model.fc.in_features # 512
        model.fc = nn.Identity()
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features # 2048
        model.fc = nn.Identity()

    elif name == "r2plus1d_18":
        weights = models.video.R2Plus1D_18_Weights.DEFAULT if pretrained else None
        model = r2plus1d_18(weights=weights)
        num_features = model.fc.in_features # 512 for r2plus1d_18
        model.fc = nn.Identity() # Remove classifier

    else:

        raise ValueError(f"Unsupported backbone: {name}")

    status = "pretrained" if pretrained and weights is not None else "randomly initialized"
    logging.info(f"Loaded {name} backbone ({status}). Output features (before final layer): {num_features}")
    return model, num_features


class ViewClassifierCNN(nn.Module):

    def __init__(self, backbone_name="resnet18", pretrained=True, num_view_classes=2, dropout_rate=0.3):
        super().__init__()
        self.backbone, num_features = get_cnn_backbone(backbone_name, pretrained)


        self.dropout = nn.Dropout(p=dropout_rate)

        self.classifier = nn.Linear(num_features, num_view_classes)

        logging.info(f"ViewClassifierCNN initialized with {backbone_name} backbone.")
        logging.info(f"Classifier Head: Input={num_features}, Output={num_view_classes}")
        logging.info(f"Dropout rate: {dropout_rate}")

    def forward(self, x):

        features = self.backbone(x)

        dropped_features = self.dropout(features)
        logits = self.classifier(dropped_features) # Shape: [B, num_view_classes]

        return logits

    def get_backbone_state_dict(self):

        if hasattr(self, 'backbone') and self.backbone is not None:
            return self.backbone.state_dict()
        else:
            logging.warning("Backbone not found, cannot return state_dict.")
            return None