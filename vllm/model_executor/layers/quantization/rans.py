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

from vllm.model_executor.utils import set_weight_attrs

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


from typing import Any, List, Optional, Tuple
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)


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
        # 1. Custom Loader (Handles Init vs Load size mismatch)
        def rans_loader(param, loaded_weight):
            if param.numel() != loaded_weight.numel():
                param.data = loaded_weight
            else:
                param.data.copy_(loaded_weight)

        # 2. Universal Param Creator
        # No 'if QKV', no 'if Merged'. Every layer gets the same single set of params.
        # This works because your file already contains 'qkv_proj.rans_exp_stream' etc.
        def create_param(suffix, dtype):
            name = f"rans_{suffix}"  # No prefix!
            param = Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
            setattr(layer, name, param)
            setattr(param, "weight_loader", rans_loader)

        # Metadata
        create_param("info", torch.int32)

        # Exponent
        for name in [
            "exp_stream",
            "exp_raw",
            "exp_states",
            "exp_sizes",
            "exp_freqs",
            "exp_cdf",
        ]:
            # Correct dtypes map
            dt = (
                torch.int32
                if "states" in name or "sizes" in name
                else torch.int16
                if "freqs" in name or "cdf" in name
                else torch.bfloat16
                if "raw" in name
                else torch.uint8
            )
            create_param(name, dt)

        # Mantissa
        for name in [
            "man_stream",
            "man_raw",
            "man_states",
            "man_sizes",
            "man_freqs",
            "man_cdf",
        ]:
            dt = (
                torch.int32
                if "states" in name or "sizes" in name
                else torch.int16
                if "freqs" in name or "cdf" in name
                else torch.uint8
            )
            create_param(name, dt)

        # Bias
        if extra_weight_attrs.get("bias", False):
            layer.bias = Parameter(torch.empty(output_size, dtype=params_dtype))
            setattr(layer.bias, "force_gpu", True)

    def apply(self, layer, x, bias=None) -> torch.Tensor:
        # 1. Check Metadata
        info = layer.rans_info

        # Fallback if layer wasn't compressed
        if info.numel() == 0:
            if layer.rans_exp_raw.numel() > 0:
                weight = layer.rans_exp_raw.to(x.device, non_blocking=True)
                return torch.nn.functional.linear(x, weight, bias)
            raise RuntimeError(f"No RANS data found for layer {layer}")

        # 2. Decompress Exponent
        expanded_size = info[1].item()

        if info[2].item() == 1:  # Is Compressed
            stream = layer.rans_exp_stream.to(x.device, non_blocking=True)
            raw_exp = torch.empty(expanded_size, dtype=torch.uint8, device=x.device)

            # Note: Add back your 'if cached' metadata optimization here!

            ccore.RansManager(stream.numel()).decompress_into(
                stream,
                layer.rans_exp_states.to(
                    x.device, dtype=torch.int32, non_blocking=True
                ),
                layer.rans_exp_sizes.to(x.device, dtype=torch.int32, non_blocking=True),
                info[3].item(),
                layer.rans_exp_freqs.to(x.device, dtype=torch.int16, non_blocking=True),
                layer.rans_exp_cdf.to(x.device, dtype=torch.int16, non_blocking=True),
                raw_exp,
            )
        else:
            raw_exp = layer.rans_exp_raw.to(x.device, non_blocking=True).flatten()

        # 3. Decompress Mantissa
        if info[4].item() == 1:
            stream_m = layer.rans_man_stream.to(x.device, non_blocking=True)
            raw_man = torch.empty(expanded_size, dtype=torch.uint8, device=x.device)

            ccore.RansManager(stream_m.numel()).decompress_into(
                stream_m,
                layer.rans_man_states.to(
                    x.device, dtype=torch.int32, non_blocking=True
                ),
                layer.rans_man_sizes.to(x.device, dtype=torch.int32, non_blocking=True),
                info[5].item(),
                layer.rans_man_freqs.to(x.device, dtype=torch.int16, non_blocking=True),
                layer.rans_man_cdf.to(x.device, dtype=torch.int16, non_blocking=True),
                raw_man,
            )
        else:
            raw_man = layer.rans_man_raw.to(x.device, non_blocking=True).flatten()

        # 4. Reconstruct & Reshape
        weight = reconstruct_from_exp_and_mantissa(
            raw_exp, raw_man, dtype=torch.bfloat16
        )

        rank = info[6].item()
        shape = torch.Size(info[7 : 7 + rank].tolist())
        weight = weight.view(shape)  # e.g. [4096, 2048] for QKV

        # 5. Compute
        return torch.nn.functional.linear(x, weight, bias)
