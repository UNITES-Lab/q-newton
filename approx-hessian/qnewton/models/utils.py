# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/11/26
from copy import deepcopy
import torch
from torch import linalg as LA
from sklearn.decomposition import PCA
from torch import nn
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertEmbeddings,
    BertConfig
)
from transformers import MistralForCausalLM


class BertProjectedEmbeddings(BertEmbeddings):
    def __init__(self, hidden_size: int, config: BertConfig):
        super().__init__(config)
        self.projector = nn.Linear(config.hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.projector.weight, std=0.02)

    def forward(self, *args, **kwargs):
        embeddings = super().forward(*args, **kwargs)
        embeddings = self.projector(embeddings)
        return embeddings


def load_and_freeze_pretrained_embedding_for_bert(
        model: BertPreTrainedModel,
        pretrained_model_name_or_path: str,
) -> BertPreTrainedModel:
    pretrained_bert = BertModel.from_pretrained(pretrained_model_name_or_path)
    add_projection = model.config.hidden_size != pretrained_bert.config.hidden_size
    if add_projection:
        print(f"Adding a linear layer to project hidden size from {pretrained_bert.config.hidden_size} "
              f"to {model.config.hidden_size}")
        model.bert.embeddings = BertProjectedEmbeddings(model.config.hidden_size, pretrained_bert.config)
    else:
        model.bert.embeddings = deepcopy(pretrained_bert.embeddings)
    # set only projector in embeddings trainable
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    if add_projection:
        model.bert.embeddings.projector.weight.requires_grad = True

    # copy weights from pretrained_bert to model except for the projector
    pretrained_bert_embeddings_state_dict = pretrained_bert.embeddings.state_dict()
    print(f"Copying embedding weights from {pretrained_model_name_or_path} to model")
    for name, param in model.bert.embeddings.named_parameters():
        if "projector.weight" not in name:
            param.data.copy_(pretrained_bert_embeddings_state_dict[name])

    return model


def load_and_freeze_pretrained_embedding_head_for_mistral(
        model: MistralForCausalLM,
        pretrained_model_name_or_path: str,
) -> MistralForCausalLM:
    """
    Load pretrained embedding & lm_head and down-project them by SVD
    """
    pretrained_mistral = MistralForCausalLM.from_pretrained(pretrained_model_name_or_path)
    to_update = model.config.hidden_size != pretrained_mistral.config.hidden_size

    if not to_update:
        print("No need to update model")
        return model

    target_dim = model.config.hidden_size

    print(f"Down projecting embedding and lm_head using SVD towards {target_dim}")

    # Down-project embedding
    pretrained_embedding = pretrained_mistral.get_input_embeddings().weight.data  # (num_embeddings, source_dim)
    pca = PCA(n_components=target_dim)
    down_projected_embedding = pca.fit_transform(pretrained_embedding)
    model.model.embed_tokens.weight.data.copy_(torch.from_numpy(down_projected_embedding))
    model.model.embed_tokens.weight.requires_grad = False

    # Down-project lm_head
    pretrained_lm_head = pretrained_mistral.lm_head.weight.data  # (num_embeddings, source_dim)
    pca = PCA(n_components=target_dim)
    down_projected_lm_head = pca.fit_transform(pretrained_lm_head)
    model.lm_head.weight.data.copy_(torch.from_numpy(down_projected_lm_head))
    model.lm_head.weight.requires_grad = False

    return model
