# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/3/17
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


class MlpForImageClassification(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 16, num_hidden_layers: int = 1, num_pixels: int = 784):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.fc_layers = nn.ModuleList([nn.Linear(num_pixels, hidden_size)])
        for _ in range(num_hidden_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))

        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values, labels=None):
        pixel_values = pixel_values.view(pixel_values.size(0), -1)  # flatten

        for layer in self.fc_layers:
            pixel_values = F.relu(layer(pixel_values))
        logits = self.fc_out(pixel_values)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return ImageClassifierOutputWithNoAttention(
            loss=loss, logits=logits, hidden_states=pixel_values
        )
