import torch
import torch.nn as nn
import math
from zhijian.models.backbone.wrapper import ModelWrapper

class CLIPModelWrapper(ModelWrapper):
    def __init__(self, model, feature_dim=0, num_classes=0, initial_weights=None):
        super().__init__(model)

        if feature_dim | num_classes != 0:
            self.classification_head = nn.Linear(feature_dim, num_classes)
            if initial_weights is None:
                initial_weights = torch.zeros_like(self.classification_head.weight)
                nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
            self.classification_head.weight = nn.Parameter(initial_weights.clone())
            self.classification_head.bias = nn.Parameter(
                torch.zeros_like(self.classification_head.bias))

    def forward(self, image, text):
        logits_per_image, logits_per_text = self.model(image, text)
        return logits_per_image

    def forward_image(self, images, return_features=False):
        features = self.model.encode_image(images)
        features = (features / features.norm(dim=-1, keepdim=True)).float()
        logits = self.classification_head(features)

        if return_features:
            return logits, features
        return logits
