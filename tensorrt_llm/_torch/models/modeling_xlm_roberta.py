# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XLM-RoBERTa encoder exposed as a dense sentence-embedding model (BGE-M3)."""

from typing import Dict, Optional

import torch
from torch import nn
from transformers import XLMRobertaConfig

from tensorrt_llm.llmapi.utils import logger_debug

from ..attention.backends import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_bert import BertModel, convert_bert_to_tllm
from .modeling_utils import register_auto_model


@register_auto_model("XLMRobertaModel")
class XLMRobertaModel(nn.Module):
    """XLM-RoBERTa backbone with CLS pooling and L2 normalization.

    `forward` returns one L2-normalized embedding per sequence ([batch, hidden]),
    i.e. the dense output of BGE-M3 (`pooling_mode_cls_token` -> `Normalize`).

    The encoder layers are the BERT ones (post-LN, full attention); XLM-RoBERTa
    only differs by its position ids, which start at `pad_token_id + 1` instead
    of 0. The offset is applied here rather than passed as an input, so that the
    encoder CUDA graphs keep working.
    """

    def __init__(self, model_config: ModelConfig[XLMRobertaConfig]):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config
        self.position_offset = (config.pad_token_id
                                if config.pad_token_id is not None else 1) + 1
        self.model = BertModel(model_config, add_pooling_layer=False)
        for _, module in self.named_modules():
            if callable(getattr(module, "_create_weights", None)):
                module._create_weights()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        position_ids = position_ids.flatten() + self.position_offset

        hidden_states = self.model(attn_metadata=attn_metadata,
                                   input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids)

        # hidden_states is the packed [total_tokens, hidden] tensor : the CLS
        # token of each sequence sits at the start offset of that sequence.
        seq_lens = attn_metadata.seq_lens_cuda
        cls_indices = torch.cumsum(seq_lens, dim=0, dtype=torch.long) - seq_lens
        pooled = hidden_states[cls_indices]
        # normalized in fp32 so that the unit norm is exact for fp16/bf16 weights
        normalized = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        return normalized.to(pooled.dtype)

    @property
    def config(self):
        # Like BertForSequenceClassification, this model does not inherit from
        # DecoderModelForCausalLM : expose the fields the executor reads.
        config = self.model_config.pretrained_config
        setattr(config, "num_key_value_heads", config.num_attention_heads)
        return config

    def infer_max_seq_len(self) -> int:
        # positions [pad_token_id + 1, max_position_embeddings) are usable
        return self.config.max_position_embeddings - self.position_offset

    def load_weights(self, weights: Dict):
        prefix = "roberta." if any(
            k.startswith("roberta.") for k in weights.keys()) else ""
        tllm_weights = convert_bert_to_tllm(
            weights,
            num_layers=self.model_config.pretrained_config.num_hidden_layers,
            prefix=prefix)

        for name, module in self.named_modules():
            if len(module._parameters) == 0:
                continue
            if name not in tllm_weights:
                raise ValueError(f"No checkpoint weights for module `{name}`")
            logger_debug(f"loading for: {name}")
            if hasattr(module, "load_weights"):
                module.load_weights(weights=tllm_weights[name])
            else:
                module_weights = tllm_weights[name][0]
                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n][:])

        # the plain `nn.LayerNorm` / `Embedding` modules are created in the
        # default dtype : align everything on the model dtype
        self.to(self.model_config.torch_dtype)
