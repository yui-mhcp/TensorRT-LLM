# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import XLMRobertaConfig
from transformers import XLMRobertaModel as HFXLMRobertaModel

from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_xlm_roberta import XLMRobertaModel

# A reduced BGE-M3 (XLM-RoBERTa large) configuration
XLM_ROBERTA_CONFIG = {
    "architectures": ["XLMRobertaModel"],
    "attention_probs_dropout_prob": 0.1,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 1026,
    "model_type": "xlm-roberta",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "pad_token_id": 1,
    "position_embedding_type": "absolute",
    "type_vocab_size": 1,
    "use_cache": True,
    "vocab_size": 1000,
}


@dataclass(repr=False)
class Scenario:
    backend: str

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}"


class TestXLMRobertaModel(unittest.TestCase):

    @parameterized.expand(
        [Scenario(backend="VANILLA"),
         Scenario(backend="TRTLLM")], lambda testcase_func, param_num, param:
        f"{testcase_func.__name__}[{param.args[0]}]")
    def test_xlm_roberta_allclose_to_hf(self, scenario: Scenario):
        """Compare the pooled embeddings of a mixed-length batch to HF"""
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)
        hf_config = XLMRobertaConfig.from_dict(deepcopy(XLM_ROBERTA_CONFIG))
        dtype = torch.float16
        hf_config.torch_dtype = dtype
        device = torch.device("cuda")

        hf_model = HFXLMRobertaModel(hf_config, add_pooling_layer=False).to(
            dtype).to(device).eval()

        model_config = ModelConfig(pretrained_config=hf_config,
                                   attn_backend=backend)
        tllm_model = XLMRobertaModel(model_config).to(device)
        tllm_model.load_weights(hf_model.state_dict())
        self.assertEqual(tllm_model.infer_max_seq_len(), 1024)

        # around the 64-token boundary, and well beyond it
        seq_lens = [8, 63, 64, 65, 300]
        sequences = [
            torch.cat([
                torch.tensor([0]),
                torch.randint(3, hf_config.vocab_size, (n - 2, )),
                torch.tensor([2])
            ]).to(device) for n in seq_lens
        ]

        with torch.inference_mode():
            hf_outputs = torch.cat([
                hf_model(input_ids=seq[None]).last_hidden_state[:, 0]
                for seq in sequences
            ])
            hf_outputs = torch.nn.functional.normalize(hf_outputs.float(),
                                                       dim=-1)

        attn_metadata = metadata_cls(
            max_num_requests=len(seq_lens),
            max_num_tokens=8192,
            kv_cache_manager=None,
            request_ids=list(range(len(seq_lens))),
            prompt_lens=seq_lens,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            num_contexts=len(seq_lens),
        )
        attn_metadata.max_seq_len = max(seq_lens)
        attn_metadata.prepare()

        input_ids = torch.cat(sequences).int()
        position_ids = torch.cat([torch.arange(n) for n in seq_lens
                                  ]).int().unsqueeze(0).to(device)

        with torch.inference_mode():
            tllm_outputs = tllm_model(input_ids=input_ids,
                                      position_ids=position_ids,
                                      attn_metadata=attn_metadata)

        self.assertEqual(tuple(tllm_outputs.shape),
                         (len(seq_lens), hf_config.hidden_size))
        torch.testing.assert_close(
            tllm_outputs.float(),
            hf_outputs,
            rtol=1.5e-2,
            atol=1.5e-2,
            msg="TRT-LLM and HF XLM-RoBERTa embeddings mismatch")
        cosine = torch.nn.functional.cosine_similarity(tllm_outputs.float(),
                                                       hf_outputs,
                                                       dim=-1)
        self.assertGreater(cosine.min().item(), 0.999)


if __name__ == "__main__":
    unittest.main()
