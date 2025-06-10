import torch
import torch.nn as nn
import random
import numpy as np
import copy
from datasets import Dataset
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

# TRL, Transformers, Unsloth imports
from unsloth import FastLanguageModel
from transformers import PreTrainedModel, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler

# ==============================================================================
# 1. Core Classes (ConditionalLM and InjectionMethod Enum)
# ==============================================================================
# These are the same classes we finalized before.

from enum import Enum


class InjectionMethod(Enum):
    PREPEND_EMBEDDING = "prepend_embedding"
    ADD_AFTER_LAYER_N = "add_after_layer_n"
    ADD_TO_EVERY_LAYER = "add_to_every_layer"
    ELEMENTWISE_PRODUCT_EVERY_LAYER = "elementwise_product_every_layer"


class ConditionalLM(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(
            self,
            language_model: PreTrainedModel,
            custom_vector_size: int,
            injection_method: InjectionMethod = InjectionMethod.PREPEND_EMBEDDING,
            injection_layer_index: Optional[int] = None
    ):
        super().__init__(language_model.config)
        self.language_model = language_model
        self.custom_vector_size = custom_vector_size
        self.injection_method = injection_method
        self.injection_layer_index = injection_layer_index
        self.embedding_size = self.language_model.get_input_embeddings().embedding_dim
        self._validate_settings()
        self.projection_layer = nn.Sequential(nn.Linear(self.custom_vector_size, self.embedding_size), nn.ReLU(),
                                              nn.Linear(self.embedding_size, self.embedding_size))
        self.projected_vector_cache: Optional[torch.Tensor] = None
        self._register_hooks()

    def _validate_settings(self):
        if self.injection_method == InjectionMethod.ADD_AFTER_LAYER_N:
            if self.injection_layer_index is None: raise ValueError("`injection_layer_index` must be set.")
            num_layers = len(self.language_model.model.layers)
            if not (0 <= self.injection_layer_index < num_layers): raise ValueError(
                f"`injection_layer_index` must be between 0 and {num_layers - 1}.")

    def _register_hooks(self):
        if self.injection_method == InjectionMethod.ADD_TO_EVERY_LAYER:
            for layer in self.language_model.model.layers: layer.register_forward_hook(self._addition_hook)
        elif self.injection_method == InjectionMethod.ADD_AFTER_LAYER_N:
            self.language_model.model.layers[self.injection_layer_index].register_forward_hook(self._addition_hook)

    def _addition_hook(
            self,
            module: nn.Module,
            inputs: Any,
            outputs: Any
    ) -> Any:
        hidden_states = outputs[0]
        modified_hidden_states = hidden_states + self.projected_vector_cache.unsqueeze(1)
        return (modified_hidden_states,) + outputs[1:]

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            custom_vector: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if custom_vector is None: return self.language_model(input_ids=input_ids, **kwargs)
        if self.injection_method == InjectionMethod.PREPEND_EMBEDDING:
            projected_vector = self.projection_layer(custom_vector).unsqueeze(1)
            token_embeddings = self.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([projected_vector, token_embeddings], dim=1)
            attention_mask, labels = kwargs.get("attention_mask"), kwargs.get("labels")
            if attention_mask is not None:
                proj_mask = torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype,
                                       device=attention_mask.device)
                kwargs["attention_mask"] = torch.cat([proj_mask, attention_mask], dim=1)
            if labels is not None:
                proj_label = torch.full((labels.shape[0], 1), -100, dtype=labels.dtype, device=labels.device)
                kwargs["labels"] = torch.cat([proj_label, labels], dim=1)
            return self.language_model(inputs_embeds=inputs_embeds, **kwargs)
        elif self.injection_method in [InjectionMethod.ADD_TO_EVERY_LAYER, InjectionMethod.ADD_AFTER_LAYER_N,
                                       InjectionMethod.ELEMENTWISE_PRODUCT_EVERY_LAYER]:
            self.projected_vector_cache = self.projection_layer(custom_vector)
            outputs = self.language_model(input_ids=input_ids, **kwargs)
            self.projected_vector_cache = None
            return outputs
        else:
            raise NotImplementedError(f"Injection method {self.injection_method} is not implemented.")


def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()
