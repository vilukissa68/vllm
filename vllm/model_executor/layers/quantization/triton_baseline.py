#!/usr/bin/env python3

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.rans import RansEmbeddingMethod
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs

import torch

try:
    from comp_inference import (
        ccore,
        reconstruct_from_exp_and_mantissa,
        fused_rans_linear_triton,
        fused_rans_linear_transposed_triton,
        rans_decomp_triton,
        uninterleave_mantissas,
        rans_decomp_triton_tiled,
        fused_rans_embedding_triton,
        triton_matmul,
    )

    print("SUCCESS: comp_inference loaded.")
except Exception as e:
    print(f"CRITICAL ERROR loading comp_inference: {e}")
    raise e


class TritonBaselineConfig(QuantizationConfig):
    @classmethod
    def get_name(cls) -> str:
        return "triton_baseline"

    @classmethod
    def get_supported_act_dtypes(cls) -> list:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "TritonBaselineConfig":
        return cls()

    def get_quant_method(self, layer, prefix: str):
        if isinstance(layer, LinearBase):
            return TritonBaselineLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return TritonBaselineEmbeddingMethod(self)
        return None


# --- 3. STANDARD LINEAR LAYERS ---
class TritonBaselineLinearMethod(LinearMethodBase):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Safely handle Tensor Parallelism sharding dimensions
        local_output_size = sum(output_partition_sizes)
        local_input_size = (
            input_size_per_partition
            if isinstance(input_size_per_partition, int)
            else sum(input_size_per_partition)
        )

        weight = torch.nn.Parameter(
            torch.empty(local_output_size, local_input_size, dtype=params_dtype)
        )

        # CRITICAL FIX: Tell vLLM exactly which dimensions to slice along when merging
        weight_attrs = {"input_dim": 1, "output_dim": 0}
        if hasattr(layer, "weight_loader"):
            weight_attrs["weight_loader"] = layer.weight_loader

        set_weight_attrs(weight, weight_attrs)
        layer.register_parameter("weight", weight)

        if extra_weight_attrs.get("bias", False):
            bias = torch.nn.Parameter(
                torch.empty(local_output_size, dtype=params_dtype)
            )
            bias_attrs = {"output_dim": 0}
            if hasattr(layer, "weight_loader"):
                bias_attrs["weight_loader"] = layer.weight_loader
            set_weight_attrs(bias, bias_attrs)
            layer.register_parameter("bias", bias)

    def apply(self, layer, x, bias=None) -> torch.Tensor:
        out = triton_matmul(x, layer.weight.t())
        if bias is not None:
            out += bias
        return out


# --- 4. EMBEDDING / LM_HEAD LAYERS ---
class TritonBaselineEmbeddingMethod(LinearMethodBase):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        local_hidden_size = (
            input_size_per_partition
            if isinstance(input_size_per_partition, int)
            else sum(input_size_per_partition)
        )
        local_vocab_size = (
            output_partition_sizes[0]
            if isinstance(output_partition_sizes, list)
            else output_size
        )

        weight = torch.nn.Parameter(
            torch.empty(local_vocab_size, local_hidden_size, dtype=params_dtype)
        )

        # Apply the exact same structural tags to the embedding weight
        weight_attrs = {"input_dim": 1, "output_dim": 0}
        if hasattr(layer, "weight_loader"):
            weight_attrs["weight_loader"] = layer.weight_loader

        set_weight_attrs(weight, weight_attrs)
        layer.register_parameter("weight", weight)

    def embedding(self, layer, x) -> torch.Tensor:
        return torch.nn.functional.embedding(x, layer.weight)

    # Inside TritonBaselineEmbeddingMethod           
    def apply(self, layer, x, bias=None) -> torch.Tensor:
        out = triton_matmul(x, layer.weight.t())
        if bias is not None:
            out += bias
        return out
