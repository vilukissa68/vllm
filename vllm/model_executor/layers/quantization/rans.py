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

from vllm.distributed import get_tensor_model_parallel_rank
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
        uninterleave_mantissas,
        rans_decomp_triton_tiled,
    )

    print("SUCCESS: comp_inference loaded.")
except Exception as e:
    print(f"CRITICAL ERROR loading comp_inference: {e}")
    raise e

from safetensors import safe_open

# Read for USE_RANS_JIT env var
import os

USE_RANS_JIT = os.environ.get("USE_RANS_JIT", "0") == "1"


# Inside your RansConfig or where you initialize it
def load_checkpoint_metadata(self, model_path: str):
    import os

    # Find the safetensors file(s)
    for f in os.listdir(model_path):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, f), framework="pt") as f_open:
                self.all_checkpoint_keys.update(f_open.keys())


class RansConfig(QuantizationConfig):
    def __init__(self, layer_configs: dict, default_th: int, default_tw: int):
        self.layer_configs = layer_configs
        self.default_th = default_th
        self.default_tw = default_tw

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
        return cls(
            layer_configs=config.get("layer_configs", {}),
            default_th=config.get("default_tile_height", 1024),
            default_tw=config.get("default_tile_width", 32),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return RansLinearMethod(self, prefix)
        # elif isinstance(layer, VocabParallelEmbedding):
        #    return RansEmbeddingMethod(self)
        return None


def register_rans_parameters(layer: torch.nn.Module, output_partition_sizes: list):
    # Determine TP sharding context
    n_local = sum(output_partition_sizes)
    tp_rank = get_tensor_model_parallel_rank()

    layer.nl = n_local // layer.tile_width  # Number of local tiles in N dimension
    layer.ts = tp_rank * layer.nl
    layer.te = layer.ts + layer.nl

    def rans_loader(param, loaded_weight, shard_id=None):
        param_name = getattr(param, "rans_name", "")

        nk = layer.nk
        ng = layer.ng
        ts, te = layer.ts, layer.te

        if any(x in param_name for x in ["offsets", "max_lens"]):
            expected_global = nk * ng
            # Strip sentinel if exists, then slice 2D vertical strip
            grid = loaded_weight[:expected_global].view(nk, ng)
            loaded_weight = grid[:, ts:te].contiguous().view(-1)

        elif "states" in param_name:
            # Global: [nk, ng, 32_lanes]
            grid = loaded_weight.view(nk, ng, layer.tile_width)
            loaded_weight = grid[:, ts:te, :].contiguous().view(-1)

        elif "man_raw" in param_name:
            # Global: [nk, ng, 1024_K, 32_N]
            grid = loaded_weight.view(nk, ng, layer.tile_height, layer.tile_width)
            loaded_weight = grid[:, ts:te, :, :].contiguous().view(-1)

        # Move to target device
        target_device = param.device
        final_weight = loaded_weight.to(target_device)
        if target_device.type == "cpu":
            final_weight = final_weight.pin_memory()

        if param.numel() != final_weight.numel():
            param.data = final_weight
        else:
            param.data.copy_(final_weight)

        if "info" in param_name:
            info = loaded_weight.cpu().tolist()
            rank = info[6]

            layer.rans_expanded_size = info[1]
            layer.rans_exp_compressed = bool(info[2])
            layer.rans_exp_num_streams = info[3]
            layer.rans_man_compressed = bool(info[4])
            layer.rans_man_num_streams = info[5]
            layer.num_tiles_n = info[7]
            layer.num_tiles_k = info[8]
            layer.tile_height = info[9]
            layer.tile_width = info[10]

            global_shape = info[11 : 11 + rank]
            layer.rans_shape = (global_shape[1], global_shape[0])

    def create_param(suffix, dtype):
        name = f"rans_{suffix}"
        param = torch.nn.Parameter(
            torch.empty(0, dtype=dtype, device="cpu").pin_memory(), requires_grad=False
        )
        param.rans_name = name
        # Force vLLM to pass us the global tensor in rans_loader
        param.is_sharded = False
        setattr(param, "weight_loader", rans_loader)
        setattr(layer, name, param)

    create_param("info", torch.int32)

    # Exponent Buffers
    exp_params = {
        "exp_stream": torch.uint8,
        "exp_raw": torch.uint8,
        "exp_states": torch.uint32,
        "exp_sizes": torch.uint32,
        "exp_tables": torch.uint32,
        "exp_slot_map": torch.uint16,
        "exp_tile_offsets": torch.uint32,
        "exp_tile_max_lens": torch.uint32,
    }
    for name, dtype in exp_params.items():
        create_param(name, dtype)

    # Mantissa Buffers
    man_params = {
        "man_stream": torch.uint8,
        "man_raw": torch.uint8,
        "man_states": torch.uint32,
        "man_sizes": torch.uint32,
        "man_tables": torch.uint32,
        "man_slot_map": torch.uint16,
    }
    for name, dtype in man_params.items():
        create_param(name, dtype)


class RansLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: RansConfig, prefix: str):
        self.quant_config = quant_config
        self.prefix = prefix

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
        # Look up layer-specific tile settings, falling back to defaults if not specified
        layer_settings = self.quant_config.layer_configs.get(self.prefix, {})
        layer.tile_height = layer_settings.get(
            "tile_height", self.quant_config.default_th
        )
        layer.tile_width = layer_settings.get(
            "tile_width", self.quant_config.default_tw
        )

        layer.nk = (input_size + layer.tile_height - 1) // layer.tile_height
        layer.ng = (output_size + layer.tile_width - 1) // layer.tile_width
        layer.rans_shape = (input_size, sum(output_partition_sizes))

        register_rans_parameters(layer, output_partition_sizes)

        if extra_weight_attrs.get("bias", False):
            layer.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(sum(output_partition_sizes), dtype=params_dtype)
                ),
            )

    def apply(self, layer, x, bias=None) -> torch.Tensor:
        device = x.device

        N, K = layer.rans_shape

        tile_k = layer.tile_height
        tile_n = layer.tile_width

        # Helper to move tensors
        def _get(p):
            return (
                p.to(device, non_blocking=True).contiguous() if p is not None else None
            )

        # Fused kernel
        fused_result = fused_rans_linear_triton(
            x=x,
            compressed_data=_get(layer.rans_exp_stream),
            initial_states=_get(layer.rans_exp_states).to(torch.uint32),
            tables=_get(layer.rans_exp_tables),
            slot_map=_get(layer.rans_exp_slot_map),
            weight_shape=(K, N),
            tile_offsets=_get(layer.rans_exp_tile_offsets).to(torch.uint32),
            tile_max_lens=_get(layer.rans_exp_tile_max_lens).to(torch.uint32),
            tile_k=tile_k,
            tile_n=tile_n,
            mantissas=_get(layer.rans_man_raw).to(torch.uint8),
            bias=_get(bias),
        )

        if USE_RANS_JIT:
            decomp_exp_local = (
                rans_decomp_triton_tiled(
                    compressed_streams=_get(layer.rans_exp_stream).to(torch.uint8),
                    initial_states=_get(layer.rans_exp_states).to(torch.uint32),
                    tables=_get(layer.rans_exp_tables).to(torch.uint32),
                    slot_map=_get(layer.rans_exp_slot_map).to(torch.uint16),
                    output_shape=(K, N),
                    tile_offsets=_get(layer.rans_exp_tile_offsets).to(torch.uint32),
                    tile_max_lens=_get(layer.rans_exp_tile_max_lens).to(torch.uint32),
                    tile_k=tile_k,
                    tile_n=tile_n,
                )
                # .contiguous()
                # .view(K, N)
            )

            # Uninterleave mantissas
            decomp_man_local = uninterleave_mantissas(
                _get(layer.rans_man_raw), K, N, TILE_K=tile_k, TILE_N=tile_n
            )

            # Reconstruct weights
            reassembled_weight_local = reconstruct_from_exp_and_mantissa(
                decomp_exp_local, decomp_man_local, dtype=torch.bfloat16
            )

            reference_result = torch.nn.functional.linear(
                x, reassembled_weight_local.t().contiguous(), _get(bias)
            )

            return reference_result

        return fused_result


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
        # Initialize metadat
        layer.num_tiles_n = 0
        layer.num_tiles_k = 0
        layer.rans_shape = torch.Size([0, 0])
        layer.tile_height = 0
        layer.tile_width = 0

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
