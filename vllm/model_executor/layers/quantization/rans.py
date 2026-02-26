#!/usr/bin/env python3

from typing import Any, List, Dict, Optional

import torch
from torch._prims_common import is_contiguous
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
        fused_rans_linear_transposed_triton,
        rans_decomp_triton,
        uninterleave_mantissas,
        rans_decomp_triton_tiled,
        fused_rans_embedding_triton,
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
        elif isinstance(layer, VocabParallelEmbedding):
            return RansEmbeddingMethod(self, prefix)
        return None


def register_rans_parameters(
    layer: torch.nn.Module, output_partition_sizes: list, quant_config: object
):
    """
    Registers empty tensors for the rANS quant method and defines the loader.
    Dynamically routes layers to GPU or CPU based on a strict VRAM budget.
    """
    # 1. Determine TP sharding context
    n_local = sum(output_partition_sizes)
    tp_rank = get_tensor_model_parallel_rank()

    layer.nl = n_local // layer.tile_width  # Number of local tiles in N dimension
    layer.ts = tp_rank * layer.nl
    layer.te = layer.ts + layer.nl

    # --- 2. DYNAMIC BYTE-ROUTER ---
    # Initialize the global trackers on the config object if they don't exist
    if not hasattr(quant_config, "vram_used_bytes"):
        quant_config.vram_used_bytes = 0
        # Read the exact byte budget set by the benchmark script
        # Fallback to 6GB (approx 60% of a 10GB limit) if not set
        budget_str = os.environ.get("RANS_WEIGHT_BUDGET_BYTES", str(6 * 1024**3))
        quant_config.vram_budget_bytes = int(budget_str)

    # Estimate the exact byte footprint of this specific layer's tiles
    nk = layer.nk
    ng = layer.ng
    th = layer.tile_height
    tw = layer.tile_width

    # Calculate bytes based on tensor dtypes (uint16=2, uint32=4)
    bytes_man_raw = (nk * ng * th * tw) * 2
    bytes_states = (nk * ng * tw) * 4
    bytes_meta = (nk * ng) * 8  # offsets (4) + max_lens (4)
    bytes_streams = (nk * ng * th * tw) * 1  # Approx 1 byte per element for exponents

    total_layer_bytes = bytes_man_raw + bytes_states + bytes_meta + bytes_streams

    # Make the routing decision based on remaining budget
    if (
        quant_config.vram_used_bytes + total_layer_bytes
        <= quant_config.vram_budget_bytes
    ):
        target_device = "cuda"
        quant_config.vram_used_bytes += total_layer_bytes
    else:
        target_device = "cpu"

    # --- 3. THE LOADER ---
    def rans_loader(param, loaded_weight, shard_id=None):
        param_name = getattr(param, "rans_name", "")

        nk_local = layer.nk
        ng_local = layer.ng
        ts, te = layer.ts, layer.te

        # Slice the 2D grid for Tensor Parallelism if necessary
        if any(x in param_name for x in ["offsets", "max_lens"]):
            expected_global = nk_local * ng_local
            # Strip sentinel if exists, then slice 2D vertical strip
            grid = loaded_weight[:expected_global].view(nk_local, ng_local)
            loaded_weight = grid[:, ts:te].contiguous().view(-1)

        elif "states" in param_name:
            # Global: [nk, ng, 32_lanes]
            grid = loaded_weight.view(nk_local, ng_local, layer.tile_width)
            loaded_weight = grid[:, ts:te, :].contiguous().view(-1)

        elif "man_raw" in param_name:
            # Global: [nk, ng, 1024_K, 32_N]
            grid = loaded_weight.view(
                nk_local, ng_local, layer.tile_height, layer.tile_width
            )
            loaded_weight = grid[:, ts:te, :, :].contiguous().view(-1)

        # Move to target device (Maintains the CUDA vs CPU decision made in create_param)
        # dest_device = param.device
        dest_device = torch.device(
            param.rans_target_device
        )  # Ensure we use the device we originally routed to
        final_weight = loaded_weight.to(dest_device)

        # Pin memory if it was routed to the CPU to guarantee fast PCIe transfers
        if dest_device.type == "cpu":
            final_weight = final_weight.pin_memory()

        if param.numel() != final_weight.numel():
            param.data = final_weight
        else:
            param.data.copy_(final_weight)

        # Extract metadata from the info tensor
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

    # --- 4. PARAMETER CREATION ---
    def create_param(suffix, dtype):
        name = f"rans_{suffix}"

        # Always initialize on cuda
        empty_tensor = torch.empty(0, dtype=dtype, device="cuda")

        param = torch.nn.Parameter(empty_tensor, requires_grad=False)
        param.rans_name = name

        param.rans_target_device = (
            target_device  # Store the routing decision on the param itself
        )

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
        # "man_stream": torch.uint8,
        "man_raw": torch.uint8,
        # "man_states": torch.uint32,
        # "man_sizes": torch.uint32,
        # "man_tables": torch.uint32,
        # "man_slot_map": torch.uint16,
    }
    for name, dtype in man_params.items():
        create_param(name, dtype)


# def register_rans_parameters(
#     layer: torch.nn.Module, output_partition_sizes: list, quant_config: object
# ):
#     # Determine TP sharding context
#     n_local = sum(output_partition_sizes)
#     tp_rank = get_tensor_model_parallel_rank()

#     layer.nl = n_local // layer.tile_width  # Number of local tiles in N dimension
#     layer.ts = tp_rank * layer.nl
#     layer.te = layer.ts + layer.nl

#     if not hasattr(quant_config, "vram_used_bytes"):
#         quant_config.vram_used_bytes = 0
#         # Set your budget (e.g., 10GB = 10 * 1024**3).
#         # In a real setup, calculate this dynamically from torch.cuda.get_device_properties()
#         quant_config.vram_budget_bytes = 10 * 1024**3

#     def rans_loader(param, loaded_weight, shard_id=None):
#         param_name = getattr(param, "rans_name", "")

#         nk = layer.nk
#         ng = layer.ng
#         ts, te = layer.ts, layer.te

#         if any(x in param_name for x in ["offsets", "max_lens"]):
#             expected_global = nk * ng
#             # Strip sentinel if exists, then slice 2D vertical strip
#             grid = loaded_weight[:expected_global].view(nk, ng)
#             loaded_weight = grid[:, ts:te].contiguous().view(-1)

#         elif "states" in param_name:
#             # Global: [nk, ng, 32_lanes]
#             grid = loaded_weight.view(nk, ng, layer.tile_width)
#             loaded_weight = grid[:, ts:te, :].contiguous().view(-1)

#         elif "man_raw" in param_name:
#             # Global: [nk, ng, 1024_K, 32_N]
#             grid = loaded_weight.view(nk, ng, layer.tile_height, layer.tile_width)
#             loaded_weight = grid[:, ts:te, :, :].contiguous().view(-1)

#         # Move to target device
#         target_device = param.device
#         final_weight = loaded_weight.to(target_device)
#         if target_device.type == "cpu":
#             final_weight = final_weight.pin_memory()

#         if param.numel() != final_weight.numel():
#             param.data = final_weight
#         else:
#             param.data.copy_(final_weight)

#         if "info" in param_name:
#             info = loaded_weight.cpu().tolist()
#             rank = info[6]

#             layer.rans_expanded_size = info[1]
#             layer.rans_exp_compressed = bool(info[2])
#             layer.rans_exp_num_streams = info[3]
#             layer.rans_man_compressed = bool(info[4])
#             layer.rans_man_num_streams = info[5]
#             layer.num_tiles_n = info[7]
#             layer.num_tiles_k = info[8]
#             layer.tile_height = info[9]
#             layer.tile_width = info[10]

#             global_shape = info[11 : 11 + rank]
#             layer.rans_shape = (global_shape[1], global_shape[0])

#     def create_param(suffix, dtype):
#         name = f"rans_{suffix}"
#         param = torch.nn.Parameter(
#             torch.empty(0, dtype=dtype, device="cpu").pin_memory(), requires_grad=False
#         )
#         param.rans_name = name
#         # Force vLLM to pass us the global tensor in rans_loader
#         param.is_sharded = False
#         setattr(param, "weight_loader", rans_loader)
#         setattr(layer, name, param)

#     create_param("info", torch.int32)

#     # Exponent Buffers
#     exp_params = {
#         "exp_stream": torch.uint8,
#         "exp_raw": torch.uint8,
#         "exp_states": torch.uint32,
#         "exp_sizes": torch.uint32,
#         "exp_tables": torch.uint32,
#         "exp_slot_map": torch.uint16,
#         "exp_tile_offsets": torch.uint32,
#         "exp_tile_max_lens": torch.uint32,
#     }
#     for name, dtype in exp_params.items():
#         create_param(name, dtype)

#     # Mantissa Buffers
#     man_params = {
#         "man_stream": torch.uint8,
#         "man_raw": torch.uint8,
#         "man_states": torch.uint32,
#         "man_sizes": torch.uint32,
#         "man_tables": torch.uint32,
#         "man_slot_map": torch.uint16,
#     }
#     for name, dtype in man_params.items():
#         create_param(name, dtype)


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

        register_rans_parameters(layer, output_partition_sizes, self.quant_config)

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
            # if hasattr(p, "device") and p.device != device:
            #     print(f"param {p.rans_name} is on {p.device}, moving to {device}")
            # if hasattr(p, "is_contiguous") and not p.is_contiguous():
            #     print(f"param {p.rans_name} is not contiguous, making it contiguous")
            return (
                p.to(device, non_blocking=True).contiguous() if p is not None else None
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
        return fused_result


class RansEmbeddingMethod(LinearMethodBase):  # Inherits the same vLLM quant base
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
        layer_settings = self.quant_config.layer_configs.get(self.prefix, {})
        layer.tile_height = layer_settings.get(
            "tile_height", self.quant_config.default_th
        )
        layer.tile_width = layer_settings.get(
            "tile_width", self.quant_config.default_tw
        )

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

        layer.nk = (local_vocab_size + layer.tile_height - 1) // layer.tile_height
        layer.ng = (local_hidden_size + layer.tile_width - 1) // layer.tile_width
        layer.rans_shape = (local_vocab_size, local_hidden_size)

        register_rans_parameters(layer, [local_hidden_size], self.quant_config)

    def embedding(self, layer, x) -> torch.Tensor:
        device = x.device

        N, K = layer.rans_shape

        tile_k = layer.tile_height
        tile_n = layer.tile_width

        def _get(p):
            return (
                p.to(device, non_blocking=True).contiguous() if p is not None else None
            )

        if USE_RANS_JIT:
            decomp_exp_local = rans_decomp_triton_tiled(
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

            decomp_man_local = uninterleave_mantissas(
                _get(layer.rans_man_raw), K, N, TILE_K=tile_k, TILE_N=tile_n
            )

            reassembled_weight_local = reconstruct_from_exp_and_mantissa(
                decomp_exp_local, decomp_man_local, dtype=torch.bfloat16
            )

            reference_result = torch.nn.functional.embedding(
                x, reassembled_weight_local
            )
            return reference_result
        fused_result = fused_rans_embedding_triton(
            x=x,
            compressed_data=_get(layer.rans_exp_stream),
            initial_states=_get(layer.rans_exp_states).to(torch.uint32),
            tables=_get(layer.rans_exp_tables).to(torch.uint32),
            slot_map=_get(layer.rans_exp_slot_map).to(torch.uint16),
            weight_shape=(K, N),
            tile_offsets=_get(layer.rans_exp_tile_offsets).to(torch.uint32),
            tile_max_lens=_get(layer.rans_exp_tile_max_lens).to(torch.uint32),
            tile_k=tile_k,
            tile_n=tile_n,
            mantissas=_get(layer.rans_man_raw).to(torch.uint16),
        )
        return fused_result

    # def apply(self, layer, x, bias=None) -> torch.Tensor:
    #     return self.embedding(layer, x)

    def apply(self, layer, x, bias=None) -> torch.Tensor:
        # NOTE: Apply is called for the lm_head and executes linear layer instead of embedding

        device = x.device
        N, K = layer.rans_shape
        tile_k = layer.tile_height
        tile_n = layer.tile_width

        def _get(p):
            return (
                p.to(device, non_blocking=True).contiguous() if p is not None else None
            )

        if USE_RANS_JIT:
            decomp_exp_local = rans_decomp_triton_tiled(
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

            decomp_man_local = uninterleave_mantissas(
                _get(layer.rans_man_raw), K, N, TILE_K=tile_k, TILE_N=tile_n
            )

            reassembled_weight_local = reconstruct_from_exp_and_mantissa(
                decomp_exp_local, decomp_man_local, dtype=torch.bfloat16
            )

            return torch.nn.functional.linear(x, reassembled_weight_local, bias)

        fused_result = fused_rans_linear_transposed_triton(
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
            mantissas=_get(layer.rans_man_raw).to(torch.uint16),
        )
        return fused_result
