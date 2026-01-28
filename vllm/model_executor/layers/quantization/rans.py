#!/usr/bin/env python3

from typing import Any, List, Dict, Optional 

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase

try:
    import comp_inference as rans_lib
    from comp_inference import ccore
    print("SUCCESS: comp_inference loaded.")
except Exception as e:
    # Print the FULL error so we know if it's "ModuleNotFound" or "libc10.so"
    print(f"CRITICAL ERROR loading comp_inference: {e}")
    # Raise it so we don't fail silently later
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

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            #return RansLinearMethod(layer, prefix, self.block_size)
            return RansLinearMethod(self)
        return None

class RansLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: RansConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int],
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       **extra_weight_attrs):

        # DEFINE THE CUSTOM LOADER
        def rans_weight_loader(param: Parameter, loaded_weight: torch.Tensor, some_other_args: Any = None):
            # Allow resizing
            if param.numel() != loaded_weight.numel():
                param.data = loaded_weight
            else:
                param.data.copy_(loaded_weight)

        # Handle possible bias
        if extra_weight_attrs.get("bias", False):
            print("[Warning] Bias re-initialization for rANS quantization is not verified.")
            layer.bias = Parameter(torch.empty(output_size, dtype=params_dtype))
            setattr(layer.bias, "force_gpu", True)

        # Initialize rANS parameters
        # Meta Info
        layer.rans_info = Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False)
        setattr(layer.rans_info, "weight_loader", rans_weight_loader)
        
        # Exponent
        layer.rans_exp_stream = Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
        layer.rans_exp_states = Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False)
        layer.rans_exp_sizes  = Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False)
        layer.rans_exp_freqs  = Parameter(torch.empty(0, dtype=torch.uint16), requires_grad=False)
        layer.rans_exp_cdf    = Parameter(torch.empty(0, dtype=torch.uint16), requires_grad=False)

        setattr(layer.rans_exp_stream, "weight_loader", rans_weight_loader)
        setattr(layer.rans_exp_states, "weight_loader", rans_weight_loader)
        setattr(layer.rans_exp_sizes,  "weight_loader", rans_weight_loader)
        setattr(layer.rans_exp_freqs,  "weight_loader", rans_weight_loader)
        setattr(layer.rans_exp_cdf,    "weight_loader", rans_weight_loader)
        
        # Mantissa
        layer.rans_man_stream = Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
        layer.rans_man_states = Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False)
        layer.rans_man_sizes  = Parameter(torch.empty(0, dtype=torch.int32), requires_grad=False)
        layer.rans_man_freqs  = Parameter(torch.empty(0, dtype=torch.uint16), requires_grad=False)
        layer.rans_man_cdf    = Parameter(torch.empty(0, dtype=torch.uint16), requires_grad=False)

        setattr(layer.rans_man_stream, "weight_loader", rans_weight_loader)
        setattr(layer.rans_man_states, "weight_loader", rans_weight_loader)
        setattr(layer.rans_man_sizes, "weight_loader", rans_weight_loader)
        setattr(layer.rans_man_freqs, "weight_loader", rans_weight_loader)
        setattr(layer.rans_man_cdf, "weight_loader", rans_weight_loader)
        
        # Fallbacks
        layer.rans_exp_raw = Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
        layer.rans_man_raw = Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)

        setattr(layer.rans_exp_raw, "weight_loader", rans_weight_loader)
        setattr(layer.rans_man_raw, "weight_loader", rans_weight_loader)


    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Decompress the weights using rANS
        info = layer.rans_info # NOTE: Convert to list
        expanded_size = info[1].item() # original weight numel
        is_exp_compressed = info[2].item()
        is_man_compressed = info[4].item()

        if is_exp_compressed:
            print("Decompressing Exponent via rANS...")
            num_streams = info[3].item()

            # Move exponent to GPU for decompression
            exp_stream_gpu = layer.rans_exp_stream.to(torch.device(x.device), dtype=torch.uint8, non_blocking=True)
            exp_states_gpu = layer.rans_exp_states.to(x.device, dtype=torch.int32, non_blocking=True)
            exp_sizes_gpu  = layer.rans_exp_sizes.to(x.device, dtype=torch.int32, non_blocking=True)
            exp_freqs_gpu  = layer.rans_exp_freqs.to(x.device,dtype=torch.uint16, non_blocking=True)
            exp_cdf_gpu    = layer.rans_exp_cdf.to(x.device, dtype=torch.uint16, non_blocking=True)
            
            # Alloc Output
            raw_exponent = torch.empty(expanded_size, dtype=torch.uint8, device=x.device).flatten()

            # Decompress
            manager = ccore.RansManager(exp_stream_gpu.numel())
            manager.decompress_into(
                exp_stream_gpu, exp_states_gpu, exp_sizes_gpu, num_streams,
                exp_freqs_gpu, exp_cdf_gpu, raw_exponent
            )
        else: # Fallback to raw
            raw_exponent = layer.rans_exp_raw.to(x.device, non_blocking=True).flatten()

        if is_man_compressed:
            num_streams = info[5].item()

            # Move mantissa to GPU for decompression
            man_stream_gpu = layer.rans_man_stream.to(x.device, dtype=torch.uint8, non_blocking=True)
            man_states_gpu = layer.rans_man_states.to(x.device, dtype=torch.int32, non_blocking=True)
            man_sizes_gpu  = layer.rans_man_sizes.to(x.device, dtype=torch.int32, non_blocking=True)
            man_freqs_gpu  = layer.rans_man_freqs.to(x.device, dtype=torch.uint16, non_blocking=True)
            man_cdf_gpu    = layer.rans_man_cdf.to(x.device, dtype=torch.utin16, non_blocking=True)
            
            # Alloc Output
            raw_mantissa = torch.empty(expanded_size, dtype=torch.uint8, device=x.device).flatten()
            
            # Decompress
            manager = ccore.RansManager(man_stream_gpu.numel())
            manager.decompress_into(
                man_stream_gpu, man_states_gpu, man_sizes_gpu, num_streams,
                man_freqs_gpu, man_cdf_gpu, raw_mantissa
            )
        else: # Fallback to raw
            raw_mantissa = layer.rans_man_raw.to(x.device, non_blocking=True).flatten()

        # Reconstruct bf16 weights
        weight = reconstruct_from_exp_and_mantissa(
            raw_exponent,
            raw_mantissa,
            dtype=torch.bfloat16
        )

        # Reshape back to original shape
        rank = info[6].item()
        shape = torch.Size(info[7 : 7+rank].tolist())
        weight = weight.reshape(shape)

        # Apply linear
        result = torch.nn.functional.linear(x, weight, bias)

        # Drop weight
        del weight

        return result
