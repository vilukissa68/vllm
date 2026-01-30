#!/usr/bin/env python3

from typing import Any, List, Dict, Optional

import torch
from torch.nn import Parameter
from torch.nn.Parameter import UninitializedParameter

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
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # 1. Custom Loader
        # Handles resizing parameters when the compressed stream size is unknown at init
        def rans_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            if param.numel() != loaded_weight.numel():
                param.data = loaded_weight
            else:
                param.data.copy_(loaded_weight)

        # 2. Shard Definitions
        # Defines how we split the vLLM layer back into HF components
        if isinstance(layer, QKVParallelLinear):
            # Qwen/Llama Attention: Fused QKV in vLLM -> Split Q, K, V in File
            shards = ["q", "k", "v"]
            layer.is_split_qkv = True
            layer.is_split_gate_up = False
        elif isinstance(layer, MergedColumnParallelLinear):
            # Qwen/Llama MLP: Fused Gate+Up in vLLM -> Split Gate, Up in File
            shards = ["gate", "up"]
            layer.is_split_gate_up = True
            layer.is_split_qkv = False
        else:
            # Standard (O_proj, Down_proj): 1:1 match
            shards = [""]  # Empty prefix
            layer.is_split_qkv = False
            layer.is_split_gate_up = False

        # Save shards list for apply()
        layer.rans_shards_list = shards

        # 3. Create Parameters for each Shard
        for shard in shards:
            # Prefix format: "q_" or ""
            p = f"{shard}_" if shard else ""

            # Helper to register
            def reg(suffix, dtype):
                # Name: q_rans_exp_stream
                name = f"{p}rans_{suffix}"
                param = Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
                setattr(layer, name, param)
                setattr(param, "weight_loader", rans_weight_loader)

            # Metadata
            reg("info", torch.int32)

            # Exponent
            reg("exp_stream", torch.uint8)
            reg("exp_states", torch.int32)
            reg("exp_sizes", torch.int32)
            reg("exp_freqs", torch.int16)
            reg("exp_cdf", torch.int16)
            reg("exp_raw", torch.bfloat16)

            # Mantissa
            reg("man_stream", torch.uint8)
            reg("man_states", torch.int32)
            reg("man_sizes", torch.int32)
            reg("man_freqs", torch.int16)
            reg("man_cdf", torch.int16)
            reg("man_raw", torch.uint8)

            # Bias (Split Biases)
            if extra_weight_attrs.get("bias", False):
                bias_name = f"{p}bias" if shard else "bias"
                b_param = Parameter(torch.empty(0, dtype=params_dtype))
                setattr(layer, bias_name, b_param)
                setattr(b_param, "weight_loader", rans_weight_loader)
                setattr(b_param, "force_gpu", True)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Optimization: Runs ONCE on CPU after loading.
        1. Moves static metadata (LUTs) to GPU.
        2. Leaves huge streams on CPU (Pinned).
        """
        device = torch.device("cuda")

        # We iterate the shards we defined in create_weights
        for shard in layer.rans_shards_list:
            p = f"{shard}_" if shard else ""

            # Check if this shard was actually loaded/compressed
            # We check 'info' because it's the header
            info_param = getattr(layer, f"{p}rans_info")

            if info_param.numel() > 0:
                # --- COMPRESSED SHARD ---

                # Move LUTs/States to GPU now
                # We update the parameter data in-place to point to GPU
                getattr(layer, f"{p}rans_exp_states").data = getattr(
                    layer, f"{p}rans_exp_states"
                ).to(device)
                getattr(layer, f"{p}rans_exp_sizes").data = getattr(
                    layer, f"{p}rans_exp_sizes"
                ).to(device)
                getattr(layer, f"{p}rans_exp_freqs").data = getattr(
                    layer, f"{p}rans_exp_freqs"
                ).to(device)
                getattr(layer, f"{p}rans_exp_cdf").data = getattr(
                    layer, f"{p}rans_exp_cdf"
                ).to(device)

                # Handle Mantissa Metadata
                # (Assuming is_man_compressed is checked via info in apply, or check param size here)
                if getattr(layer, f"{p}rans_man_stream").numel() > 0:
                    getattr(layer, f"{p}rans_man_states").data = getattr(
                        layer, f"{p}rans_man_states"
                    ).to(device)
                    getattr(layer, f"{p}rans_man_sizes").data = getattr(
                        layer, f"{p}rans_man_sizes"
                    ).to(device)
                    getattr(layer, f"{p}rans_man_freqs").data = getattr(
                        layer, f"{p}rans_man_freqs"
                    ).to(device)
                    getattr(layer, f"{p}rans_man_cdf").data = getattr(
                        layer, f"{p}rans_man_cdf"
                    ).to(device)
                else:
                    # Raw fallback mantissa -> GPU
                    getattr(layer, f"{p}rans_man_raw").data = getattr(
                        layer, f"{p}rans_man_raw"
                    ).to(device)

            else:
                # --- RAW FALLBACK SHARD ---
                # Move raw exponent to GPU
                raw = getattr(layer, f"{p}rans_exp_raw")
                if raw.numel() > 0:
                    raw.data = raw.to(device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        decompressed_parts = []
        biases = []

        for shard in layer.rans_shards_list:
            p = f"{shard}_" if shard else ""

            # 1. Check Info
            info = getattr(layer, f"{p}rans_info")

            if info.numel() == 0:
                # Raw Fallback logic
                raw = getattr(layer, f"{p}rans_exp_raw")
                if raw.numel() == 0:
                    # If this happens, a shard is missing.
                    # For QKV, this is fatal. For research debugging, maybe print warning.
                    raise RuntimeError(
                        f"Missing weights for shard '{shard}' in layer {layer}"
                    )
                decompressed_parts.append(raw)  # Already on GPU from process_weights
            else:
                # 2. Decompress
                expanded_size = info[1].item()
                is_man_comp = info[4].item() == 1

                # EXPONENT
                stream = getattr(layer, f"{p}rans_exp_stream").to(
                    x.device, non_blocking=True
                )
                raw_exp = torch.empty(expanded_size, dtype=torch.uint8, device=x.device)

                ccore.RansManager(stream.numel()).decompress_into(
                    stream,
                    getattr(layer, f"{p}rans_exp_states"),  # Already on GPU
                    getattr(layer, f"{p}rans_exp_sizes"),
                    info[3].item(),
                    getattr(layer, f"{p}rans_exp_freqs"),
                    getattr(layer, f"{p}rans_exp_cdf"),
                    raw_exp,
                )

                # MANTISSA
                if is_man_comp:
                    stream_m = getattr(layer, f"{p}rans_man_stream").to(
                        x.device, non_blocking=True
                    )
                    raw_man = torch.empty(
                        expanded_size, dtype=torch.uint8, device=x.device
                    )
                    ccore.RansManager(stream_m.numel()).decompress_into(
                        stream_m,
                        getattr(layer, f"{p}rans_man_states"),
                        getattr(layer, f"{p}rans_man_sizes"),
                        info[5].item(),
                        getattr(layer, f"{p}rans_man_freqs"),
                        getattr(layer, f"{p}rans_man_cdf"),
                        raw_man,
                    )
                else:
                    raw_man = getattr(layer, f"{p}rans_man_raw")  # Already on GPU

                # RECONSTRUCT
                weight = reconstruct_from_exp_and_mantissa(
                    raw_exp, raw_man, dtype=torch.bfloat16
                )

                # RESHAPE
                rank = info[6].item()
                shape = torch.Size(info[7 : 7 + rank].tolist())
                decompressed_parts.append(weight.view(shape))

            # 3. Collect Bias
            bias_name = f"{p}bias" if shard else "bias"
            if hasattr(layer, bias_name):
                b = getattr(layer, bias_name)
                if b.numel() > 0:
                    biases.append(b.to(x.device))

        # --- FUSION ---

        if len(decompressed_parts) == 1:
            final_weight = decompressed_parts[0]
        else:
            # Fuse Q+K+V (dim 0)
            final_weight = torch.cat(decompressed_tensors, dim=0)

        final_bias = None
        if biases:
            if len(biases) == 1:
                final_bias = biases[0]
            else:
                final_bias = torch.cat(biases, dim=0)

        if final_bias is None and bias is not None:
            final_bias = bias

        return torch.nn.functional.linear(x, final_weight, final_bias)
