#!/usr/bin/env python3

from typing import Any, List, Dict, Optional

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)

from vllm.model_executor.utils import set_weight_attrs

from typing import Any, List, Optional, Tuple
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)


try:
    from comp_inference import (
        ccore,
        reconstruct_from_exp_and_mantissa,
        fused_rans_linear_triton,
        rans_decomp_triton,
    )

    print("SUCCESS: comp_inference loaded.")
except Exception as e:
    print(f"CRITICAL ERROR loading comp_inference: {e}")
    raise e


class RansConfig(QuantizationConfig):
    def __init__(self, block_size: int = 4096):
        self.block_size = block_size

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "rans"

    @classmethod
    def get_supported_act_dtypes(cls) -> list:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # No specific GPU capability required, native cuda rans implementation required
        return -1

    @staticmethod
    def get_config_filenames() -> list:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "RansConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return RansLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return RansEmbeddingMethod(self)
        return None


def register_rans_parameters(layer: torch.nn.Module):
    def rans_loader(param, loaded_weight, other=None):
        # 1. Determine the target device (assigned by vLLM)
        target_device = param.device

        # 2. Move loaded_weight to the target device
        # If target_device is CPU, we MUST call .pin_memory()
        if target_device.type == "cpu":
            # Pinning is required so the GPU can read this CPU memory directly
            final_weight = loaded_weight.to(target_device).pin_memory()
        else:
            final_weight = loaded_weight.to(target_device)

        # 3. Handle the 'empty(0)' initialization
        # If the parameter hasn't been allocated yet or size changed
        if param.numel() != final_weight.numel():
            # Re-assigning .data is safe here during the loading phase
            param.data = final_weight
        else:
            param.data.copy_(final_weight)

        param_name = getattr(param, "rans_name", "")
        if "info" in param_name:  # or check by shape/type
            info_list = loaded_weight.cpu().tolist()
            layer.rans_expanded_size = info_list[1]
            layer.rans_exp_compressed = bool(info_list[2])
            layer.rans_exp_num_streams = info_list[3]
            layer.rans_man_compressed = bool(info_list[4])
            layer.rans_man_num_streams = info_list[5]
            layer.rans_rank = info_list[6]
            layer.rans_shape = torch.Size(info_list[7 : 7 + layer.rans_rank])

    def create_param(suffix, dtype):
        name = f"rans_{suffix}"

        pinned_data = torch.empty(0, dtype=dtype, device="cpu").pin_memory()
        param = torch.nn.Parameter(pinned_data, requires_grad=False)
        param.rans_name = name

        # Tell vLLM to move param to GPU and keep it there
        # setattr(param, "force_gpu", True)
        setattr(param, "weight_loader", rans_loader)
        setattr(layer, name, param)

    create_param("info", torch.int32)

    # Exponent Buffers
    exp_params = {
        "exp_stream": torch.uint8,
        "exp_raw": torch.uint8,
        "exp_states": torch.int32,
        "exp_sizes": torch.int32,
        "exp_tables": torch.uint32,
        "exp_slot_map": torch.uint16,
    }
    for name, dtype in exp_params.items():
        create_param(name, dtype)

    # Mantissa Buffers
    man_params = {
        "man_stream": torch.uint8,
        "man_raw": torch.uint8,
        "man_states": torch.int32,
        "man_sizes": torch.int32,
        "man_tables": torch.uint32,
        "man_slot_map": torch.uint16,
    }
    for name, dtype in man_params.items():
        create_param(name, dtype)


class RansLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: RansConfig):
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
        # Register RANS Parameters
        register_rans_parameters(layer)

        # Bias
        if extra_weight_attrs.get("bias", False):
            layer.bias = Parameter(torch.empty(output_size, dtype=params_dtype))
            # setattr(layer.bias, "force_gpu", True)

    def apply(self, layer, x, bias=None) -> torch.Tensor:
        info = layer.rans_info
        if info.numel() == 0:
            if layer.rans_exp_raw.numel() > 0:
                weight = layer.rans_exp_raw.to(x.device, non_blocking=True)
                return torch.nn.functional.linear(x, weight, bias)
            raise RuntimeError(f"No RANS data found for layer {layer}")

        # Decompress Exponent
        expanded_size = layer.rans_expanded_size

        # Triton fused decomp matmul
        if True:
            N, K = layer.rans_shape[-2], layer.rans_shape[-1]
            M = x.view(-1, K).shape[0]
            out_shape = list(x.shape[:-1]) + [N]
            return fused_rans_linear_triton(
                x=x,
                exp_stream=layer.rans_exp_stream,
                man_stream=layer.rans_man_raw,
                exp_states=layer.rans_exp_states,
                exp_tables=layer.rans_exp_tables,
                exp_slot_map=layer.rans_exp_slot_map,
                exp_sizes=layer.rans_exp_sizes,
                bias=bias,
                weight_shape=layer.rans_shape,
                output_shape=out_shape,
            )

        if layer.rans_exp_compressed:  # Is Compressed
            stream = layer.rans_exp_stream.to(x.device)
            exp_states = layer.rans_exp_states.to(x.device)
            exp_sizes = layer.rans_exp_sizes.to(x.device)
            num_streams = layer.rans_exp_num_streams
            exp_tables = layer.rans_exp_tables.to(x.device)
            exp_slot_map = layer.rans_exp_slot_map.to(x.device)
            # raw_exp = torch.empty(expanded_size, dtype=torch.uint8, device=x.device)

            # ccore.decompress(
            #     stream,
            #     exp_states,
            #     exp_sizes,
            #     num_streams,
            #     exp_tables,
            #     exp_slot_map,
            #     raw_exp,
            # )
            N, K = layer.rans_shape[-2], layer.rans_shape[-1]
            out_shape = list(x.shape[:-1]) + [N]
            raw_exp = rans_decomp_triton(
                compressed_streams=stream,
                initial_states=exp_states,
                tables=exp_tables,
                slot_map=exp_slot_map,
                stream_sizes=exp_sizes,
                output_shape=layer.rans_shape,
            ).flatten()

        else:
            raw_exp = layer.rans_exp_raw.flatten()

        # Decompress Mantissa
        if layer.rans_man_compressed:
            raw_man = torch.empty(expanded_size, dtype=torch.uint8, device=x.device)

            ccore.decompress(
                layer.rans_man_stream,
                layer.rans_man_states,
                layer.rans_man_sizes,
                layer.rans_man_num_streams,
                layer.rans_man_tables,
                layer.rans_man_slot_map,
                raw_man,
            )
        else:
            raw_man = layer.rans_man_raw.flatten().to(x.device)

        # Reconstruct & Reshape
        weight = reconstruct_from_exp_and_mantissa(
            raw_exp, raw_man, dtype=torch.bfloat16
        )

        shape = layer.rans_shape
        weight = weight.view(shape)
        # Matmul
        return torch.nn.functional.linear(x, weight, bias)


class RansEmbeddingMethod(RansLinearMethod):
    def __init__(self, quant_config: RansConfig):
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
        # Register RANS Parameters
        register_rans_parameters(layer)

        # Bias
        if extra_weight_attrs.get("bias", False):
            layer.bias = Parameter(torch.empty(output_size, dtype=params_dtype))
            # setattr(layer.bias, "force_gpu", True)

    def embedding(self, layer, x, bias=None) -> torch.Tensor:
        print("Embedding called")
        info = layer.rans_info
        if info.numel() == 0:
            if layer.rans_exp_raw.numel() > 0:
                weight = layer.rans_exp_raw.to(x.device, non_blocking=True)
                return torch.nn.functional.linear(x, weight, bias)
            raise RuntimeError(f"No RANS data found for layer {layer}")

        # Decompress Exponent
        expanded_size = layer.rans_expanded_size

        if layer.rans_exp_compressed:
            stream = layer.rans_exp_stream.to(x.device)
            exp_states = layer.rans_exp_states.to(x.device)
            exp_sizes = layer.rans_exp_sizes.to(x.device)
            num_streams = layer.rans_exp_num_streams
            exp_tables = layer.rans_exp_tables.to(x.device)
            exp_slot_map = layer.rans_exp_slot_map.to(x.device)
            # raw_exp = torch.empty(expanded_size, dtype=torch.uint8, device=x.device)

            # ccore.decompress(
            #     stream,
            #     exp_states,
            #     exp_sizes,
            #     num_streams,
            #     exp_tables,
            #     exp_slot_map,
            #     raw_exp,
            # )
            N, K = layer.rans_shape[-2], layer.rans_shape[-1]
            out_shape = list(x.shape[:-1]) + [N]
            raw_exp = rans_decomp_triton(
                compressed_streams=stream,
                initial_states=exp_states,
                tables=exp_tables,
                slot_map=exp_slot_map,
                stream_sizes=exp_sizes,
                output_shape=layer.rans_shape,
            ).flatten()
        else:
            raw_exp = layer.rans_exp_raw.flatten()

        # Decompress Mantissa
        if layer.rans_man_compressed:
            stream_m = layer.rans_man_stream.to(x.device, non_blocking=True)
            raw_man = torch.empty(expanded_size, dtype=torch.uint8, device=x.device)

            ccore.decompress(
                stream_m,
                layer.rans_man_states,
                layer.rans_man_sizes,
                layer.rans_man_num_streams,
                layer.rans_man_tables,
                layer.rans_man_slot_map,
                raw_man,
            )
        else:
            raw_man = layer.rans_man_raw.flatten().to(x.device)

        # Reconstruct & Reshape
        weight = reconstruct_from_exp_and_mantissa(
            raw_exp, raw_man, dtype=torch.bfloat16
        )

        shape = layer.rans_shape
        weight = weight.view(shape)
        # Matmul
        return torch.nn.functional.embedding(x.to(torch.int32), weight, bias)
