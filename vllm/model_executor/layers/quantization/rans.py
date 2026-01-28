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
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)

try:
    from comp_inference import ccore, reconstruct_from_exp_and_mantissa

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
            # return RansLinearMethod(layer, prefix, self.block_size)
            return RansLinearMethod(self)
        return None


class RansLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: RansConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # DEFINE THE CUSTOM LOADER
        def rans_weight_loader(
            param: Parameter, loaded_weight: torch.Tensor, some_other_args: Any = None
        ):
            # Allow resizing
            if param.numel() != loaded_weight.numel():
                param.data = loaded_weight
            else:
                param.data.copy_(loaded_weight)

        # Handle possible bias
        if extra_weight_attrs.get("bias", False):
            print(
                "[Warning] Bias re-initialization for rANS quantization is not verified."
            )
            layer.bias = Parameter(torch.empty(output_size, dtype=params_dtype))
            setattr(layer.bias, "force_gpu", True)

        # Initialize rANS parameters
        def create_rans_params(prefix=""):
            p = "_" if prefix else ""

            # Meta Info
            info_param = Parameter(
                torch.empty(0, dtype=torch.int32), requires_grad=False
            )
            setattr(layer, f"{prefix}{p}rans_info", info_param)
            setattr(info_param, "weight_loader", rans_weight_loader)

            # Exponent
            params_exp = ["stream", "states", "sizes", "freqs", "cdf"]
            dtypes_exp = [
                torch.uint8,
                torch.int32,
                torch.int32,
                torch.uint16,
                torch.uint16,
            ]

            for suffix, dtype in zip(params_exp, dtypes_exp):
                param_name = f"{prefix}{p}rans_exp_{suffix}"
                param = Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
                setattr(layer, param_name, param)
                setattr(param, "weight_loader", rans_weight_loader)
                print(f"Created rANS param: {param_name} with dtype {dtype}")

            # Mantissa
            params_man = ["stream", "states", "sizes", "freqs", "cdf"]
            dtypes_man = [
                torch.uint8,
                torch.int32,
                torch.int32,
                torch.uint16,
                torch.uint16,
            ]

            for suffix, dtype in zip(params_man, dtypes_man):
                param_name = f"{prefix}{p}rans_man_{suffix}"
                param = Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
                setattr(layer, param_name, param)
                setattr(param, "weight_loader", rans_weight_loader)
                print(f"Created rANS param: {param_name} with dtype {dtype}")

            # Raw Data
            exp_raw = Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
            setattr(layer, f"{prefix}{p}rans_exp_raw", exp_raw)
            setattr(exp_raw, "weight_loader", rans_weight_loader)

            man_raw = Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
            setattr(layer, f"{prefix}{p}rans_man_raw", man_raw)
            setattr(man_raw, "weight_loader", rans_weight_loader)

        if isinstance(layer, QKVParallelLinear):
            create_rans_params("")
            create_rans_params("")
            create_rans_params("")
            layer.is_split_qkv = True

        elif isinstance(layer, MergedColumnParallelLinear):
            create_rans_params("")
            create_rans_params("")
            layer.is_split_gate_up = True

        else:
            create_rans_params("")
            layer.is_split_qkv = False
            layer.is_split_gate_up = False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Helper to decompress one specific set
        def decompress_part(prefix=""):
            p = "_" if prefix else ""

            # 1. Get Params by name
            info = getattr(layer, f"{prefix}{p}rans_info")
            expanded_size = info[1].item()
            is_exp_compressed = info[2].item() == 1
            is_man_compressed = info[4].item() == 1

            if is_exp_compressed:
                print("Decompressing Exponent via rANS...")
                num_streams = info[3].item()

                stream = getattr(layer, f"{prefix}{p}rans_exp_stream")
                states = getattr(layer, f"{prefix}{p}rans_exp_states")
                sizes = getattr(layer, f"{prefix}{p}rans_exp_sizes")
                freqs = getattr(layer, f"{prefix}{p}rans_exp_freqs")
                cdf = getattr(layer, f"{prefix}{p}rans_exp_cdf")

                raw_exponent = torch.empty(
                    expanded_size, dtype=torch.uint8, device=x.device
                )

                manager = ccore.RansManager(stream.numel())
                manager.decompress_into(
                    stream, states, sizes, num_streams, freqs, cdf, raw_exponent
                )
            else:
                raw_exponent = (
                    getattr(layer, f"{prefix}{p}rans_exp_raw")
                    .to(x.device, non_blocking=True)
                    .flatten()
                )

            if is_man_compressed:
                print("Decompressing Mantissa via rANS...")
                num_streams = info[5].item()

                stream = getattr(layer, f"{prefix}{p}rans_man_stream")
                states = getattr(layer, f"{prefix}{p}rans_man_states")
                sizes = getattr(layer, f"{prefix}{p}rans_man_sizes")
                freqs = getattr(layer, f"{prefix}{p}rans_man_freqs")
                cdf = getattr(layer, f"{prefix}{p}rans_man_cdf")

                raw_mantissa = torch.empty(
                    expanded_size, dtype=torch.uint8, device=x.device
                )
            else:
                raw_mantissa = (
                    getattr(layer, f"{prefix}{p}rans_man_raw")
                    .to(x.device, non_blocking=True)
                    .flatten()
                )

            # 4. Reconstruct & Reshape
            weight = reconstruct_from_exp_and_mantissa(
                raw_exponent, raw_mantissa, dtype=torch.bfloat16
            )

            rank = info[6].item()
            shape = torch.Size(info[7 : 7 + rank].tolist())
            return weight.view(shape)

        if hasattr(layer, "is_split_qkv") and layer.is_split_qkv:
            # Decompress 3 parts
            w_q = decompress_part("")
            w_k = decompress_part("")
            w_v = decompress_part("")

            # Fuse on GPU (Fast)
            # Output dim is 0 for Linear weights
            weight = torch.cat([w_q, w_k, w_v], dim=0)

        elif hasattr(layer, "is_split_gate_up") and layer.is_split_gate_up:
            w_gate = decompress_part("")
            w_up = decompress_part("")
            weight = torch.cat([w_gate, w_up], dim=0)

        else:
            weight = decompress_part("")

        # Apply linear
        result = torch.nn.functional.linear(x, weight, bias)

        # Drop weight
        del weight

        return result
