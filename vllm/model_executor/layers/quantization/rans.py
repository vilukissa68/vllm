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
            layer.rans_info = Parameter(
                torch.empty(0, dtype=torch.int32), requires_grad=False
            )
            setattr(layer.rans_info, "weight_loader", rans_weight_loader)

            # Exponent
            setattr(
                layer,
                f"{prefix}{p}rans_info",
                Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_exp_stream",
                Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_exp_states",
                Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_exp_sizes",
                Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_exp_freqs",
                Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_exp_cdf",
                Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False),
            )

            setattr(
                getattr(layer, f"{prefix}{p}rans_exp_stream"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_exp_states"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_exp_sizes"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_exp_freqs"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_exp_cdf"),
                "weight_loader",
                rans_weight_loader,
            )

            # Mantissa
            setattr(
                layer,
                f"{prefix}{p}rans_man_stream",
                Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_man_states",
                Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_man_sizes",
                Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_man_freqs",
                Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_man_cdf",
                Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False),
            )

            setattr(
                getattr(layer, f"{prefix}{p}rans_man_stream"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_man_states"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_man_sizes"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_man_freqs"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_man_cdf"),
                "weight_loader",
                rans_weight_loader,
            )

            # Fallbacks
            setattr(
                layer,
                f"{prefix}{p}rans_exp_raw",
                Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}{p}rans_man_raw",
                Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False),
            )

            setattr(
                getattr(layer, f"{prefix}{p}rans_exp_raw"),
                "weight_loader",
                rans_weight_loader,
            )
            setattr(
                getattr(layer, f"{prefix}{p}rans_man_raw"),
                "weight_loader",
                rans_weight_loader,
            )

        if isinstance(layer, QKVParallelLinear):
            create_rans_params("q")
            create_rans_params("k")
            create_rans_params("v")
            layer.is_split_qkv = True

        elif isinstance(layer, MergedColumnParallelLinear):
            create_rans_params("gate")
            create_rans_params("up")
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

            if is_exp_compressed:
                print("Decompressing Exponent via rANS...")
                stream = getattr(layer, f"{prefix}{p}rans_exp_stream")
                states = getattr(layer, f"{prefix}{p}rans_exp_states")
                sizes = getattr(layer, f"{prefix}{p}rans_exp_sizes")
                freqs = getattr(layer, f"{prefix}{p}rans_exp_freqs")
                cdf = getattr(layer, f"{prefix}{p}rans_exp_cdf")

                num_streams = info[3].item()

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
                stream = getattr(layer, f"{prefix}{p}rans_man_stream")
                states = getattr(layer, f"{prefix}{p}rans_man_states")
                sizes = getattr(layer, f"{prefix}{p}rans_man_sizes")
                freqs = getattr(layer, f"{prefix}{p}rans_man_freqs")
                cdf = getattr(layer, f"{prefix}{p}rans_man_cdf")

                num_streams = info[5].item()
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

        if getattr(layer, "is_split_qkv", False):
            # Decompress 3 parts
            w_q = decompress_part("q")
            w_k = decompress_part("k")
            w_v = decompress_part("v")

            # Fuse on GPU (Fast)
            # Output dim is 0 for Linear weights
            weight = torch.cat([w_q, w_k, w_v], dim=0)

        elif getattr(layer, "is_split_gate_up", False):
            w_gate = decompress_part("gate")
            w_up = decompress_part("up")
            weight = torch.cat([w_gate, w_up], dim=0)

        else:
            weight = decompress_part("")

        # Apply linear
        result = torch.nn.functional.linear(x, weight, bias)

        # Drop weight
        del weight

        return result
