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
        fused_rans_linear_triton_uncoalesced,
        fused_rans_linear_transposed_triton,
        fused_rans_linear_transposed_triton_uncoalesced,
        rans_decomp_triton,
        uninterleave_mantissas,
        rans_decomp_triton_tiled,
        fused_rans_embedding_triton,
        fused_rans_embedding_triton_uncoalesced,
        triton_matmul,
    )

    print("SUCCESS: comp_inference loaded.")
except Exception as e:
    print(f"CRITICAL ERROR loading comp_inference: {e}")
    raise e

from safetensors import safe_open

# Read for USE_RANS_JIT env var
import os

USE_RANS_JIT = os.environ.get("USE_RANS_JIT", "0") == "1"


def compute_stream_sizes(stream_offsets, compressed_stream, device):
    """
    Computes stream sizes by subtracting adjacent offsets.
    Safely casts to int64 because PyTorch lacks CUDA subtraction kernels for uint32.
    """
    offsets_i64 = stream_offsets.to(torch.int64)
    total_bytes = torch.tensor(
        [compressed_stream.numel()], dtype=torch.int64, device=device
    )
    shifted_offsets = torch.cat([offsets_i64[1:], total_bytes])

    return (shifted_offsets - offsets_i64).to(torch.uint32)


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
        # if isinstance(layer, LinearBase):
        if isinstance(layer, LinearBase):
            return RansLinearMethod(self, prefix)
        elif isinstance(layer, VocabParallelEmbedding):
            return RansEmbeddingMethod(self, prefix)
        return None


class RansWorkspace:
    _buffer = None

    @classmethod
    def reserve_if_needed(cls, num_bytes, device):
        """
        Eagerly allocates the shared workspace to ensure vLLM's memory profiler
        and the RANS loader account for this memory usage immediately.
        """
        # Calculate elements needed for bfloat16 (2 bytes)
        elements_needed = (num_bytes + 1) // 2

        if cls._buffer is None:
            # First allocation
            cls._buffer = torch.zeros(
                elements_needed, device=device, dtype=torch.bfloat16
            )
        elif cls._buffer.numel() < elements_needed:
            # Grow buffer
            current_size = cls._buffer.numel() * 2
            cls._buffer = None
            torch.cuda.empty_cache()

            # Allocate new larger buffer
            cls._buffer = torch.zeros(
                elements_needed, device=device, dtype=torch.bfloat16
            )

    @classmethod
    def get_workspace(cls, M, N, split_k, device):
        elements = M * N * split_k
        # Safety check - though reserve_if_needed should have covered this
        if cls._buffer is None or cls._buffer.numel() < elements:
            cls.reserve_if_needed(elements * 2, device)

        return cls._buffer[:elements].view(split_k, M, N)


def register_rans_parameters(
    layer: torch.nn.Module, output_partition_sizes: list, quant_config: object
):
    """
    Registers empty tensors for the rANS quant method and defines the loader.
    Dynamically routes to GPU or CPU at load-time based on true physical VRAM.
    """
    # 1. Determine TP sharding context
    n_local = sum(output_partition_sizes)
    tp_rank = get_tensor_model_parallel_rank()

    layer.nl = n_local // layer.tile_width
    layer.ts = tp_rank * layer.nl
    layer.te = layer.ts + layer.nl

    # Check for dense packing
    # is_uncoalesced = getattr(quant_config, "uncoalesced_interleaving", False)
    # is_uncoalesced = layer.uncoalesced_interleaving
    is_uncoalesced = getattr(layer, "uncoalesced_interleaving", False)

    # --- THE LOADER ---
    def rans_loader(param, loaded_weight, shard_id=None):
        param_name = getattr(param, "rans_name", "")

        nk_local = layer.nk
        ng_local = layer.ng
        ts, te = layer.ts, layer.te

        # Slice the 2D grid for Tensor Parallelism if necessary
        # if any(x in param_name for x in ["offsets", "max_lens"]):
        #     expected_global = nk_local * ng_local
        #     grid = loaded_weight[:expected_global].view(nk_local, ng_local)
        #     loaded_weight = grid[:, ts:te].contiguous().view(-1)
        # elif "states" in param_name:
        #     grid = loaded_weight.view(nk_local, ng_local, 2, layer.tile_width)
        #     loaded_weight = grid[:, ts:te, :].contiguous().view(-1)

        if any(x in param_name for x in ["offsets", "max_lens", "states"]):
            if is_uncoalesced and param_name in [
                "rans_exp_stream_offsets",
                "rans_exp_states",
            ]:
                # DENSE PACKING: 1 value per stream. Shape: [nk, ng * tile_width]
                grid = loaded_weight.view(nk_local, ng_local * layer.tile_width)
                loaded_weight = (
                    grid[:, ts * layer.tile_width : te * layer.tile_width]
                    .contiguous()
                    .view(-1)
                )

            elif not is_uncoalesced and param_name == "rans_exp_states":
                # LEGACY ILP2 STATES: [nk, ng, 2, tw]
                grid = loaded_weight.view(nk_local, ng_local, 2, layer.tile_width)
                loaded_weight = grid[:, ts:te, :].contiguous().view(-1)

            elif not is_uncoalesced and param_name in [
                "rans_exp_tile_offsets",
                "rans_exp_tile_max_lens",
            ]:
                # LEGACY TILE ARRAYS: 1 value per tile. Shape [nk, ng]
                expected_global = nk_local * ng_local
                grid = loaded_weight[:expected_global].view(nk_local, ng_local)
                loaded_weight = grid[:, ts:te].contiguous().view(-1)

        elif "man_raw" in param_name:
            grid = loaded_weight.view(
                nk_local, ng_local, layer.tile_height, layer.tile_width
            )
            loaded_weight = grid[:, ts:te, :, :].contiguous().view(-1)

        elif "info" in param_name:
            info = loaded_weight.cpu().tolist()
            rank = info[6]
            global_shape = info[11 : 11 + rank]

            # For linear layers: [Out, In]. For Transposed: [Vocab, Hidden] or vice versa.
            # We assume the larger dimension is likely the one driving workspace cost.
            dim_0, dim_1 = global_shape[1], global_shape[0]
            N_dim = max(dim_0, dim_1)
            K_dim = min(dim_0, dim_1)

            layer.rans_shape = (dim_0, dim_1)

            # Calculate Workspace Requirements
            MAX_BATCH_GUESS = 8192  # Profiling often assumes a full context fill

            # Dynamic Split-K: If N is huge (Vocab), we know we force Split-K=1
            # If N is small (MLP), we force Split-K=8.
            if N_dim > 32000:
                layer.split_k = 1
            else:
                layer.split_k = 8

            workspace_bytes = layer.split_k * MAX_BATCH_GUESS * N_dim * 2  # bfloat16

            # This expands the singleton buffer NOW, forcing VRAM usage up.
            from .rans import RansWorkspace

            RansWorkspace.reserve_if_needed(workspace_bytes, device="cuda")

        # Dynamic parameter routing
        budget_str = os.environ.get("RANS_WEIGHT_BUDGET_BYTES", str(6 * 1024**3))
        vram_budget = int(budget_str)

        actual_bytes = loaded_weight.numel() * loaded_weight.element_size()

        if torch.cuda.memory_allocated() + actual_bytes <= vram_budget:
            dest_device = torch.device("cuda")
        else:
            dest_device = torch.device("cpu")

        # Move to target device
        final_weight = loaded_weight.to(dest_device)

        # Pin memory if it spilled to CPU
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

    # --- PARAMETER CREATION ---
    def create_param(suffix, dtype):
        name = f"rans_{suffix}"

        # Initialize an empty shell. The loader will dynamically resize it
        # and move it to the correct device later.
        empty_tensor = torch.empty(0, dtype=dtype, device="cuda")
        param = torch.nn.Parameter(empty_tensor, requires_grad=False)
        param.rans_name = name

        # Force vLLM to pass us the global tensor in rans_loader
        param.is_sharded = False
        setattr(param, "weight_loader", rans_loader)
        setattr(layer, name, param)

    create_param("info", torch.int32)

    exp_params = {
        "exp_stream": torch.uint8,
        "exp_raw": torch.uint8,
        "exp_states": torch.uint32,
        # "exp_sizes": torch.uint32,
        "exp_tables": torch.uint32,
        "exp_slot_map": torch.uint16,
        # "exp_tile_offsets": torch.uint32,
        # "exp_tile_max_lens": torch.uint32,
    }

    for name, dtype in exp_params.items():
        create_param(name, dtype)

        # Conditionally create the correct offset/size tracking arrays
    if is_uncoalesced:
        create_param("exp_stream_offsets", torch.uint32)
    else:
        create_param("exp_tile_offsets", torch.uint32)
        create_param("exp_tile_max_lens", torch.uint32)

    man_params = {
        "man_raw": torch.uint8,
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

        layer.uncoalesced_interleaving = layer_settings.get(
            "uncoalesced_interleaving", False
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

        is_uncoalesced = layer.uncoalesced_interleaving

        # Helper to move tensors
        def _get(p):
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

        M = x.view(-1, K).shape[0]
        workspace = RansWorkspace.get_workspace(M, N, layer.split_k, device)

        if is_uncoalesced:
            stream_offsets = _get(layer.rans_exp_stream_offsets).to(torch.uint32)
            compressed_stream = _get(layer.rans_exp_stream)

            # stream_sizes[i] = stream_offsets[i+1] - stream_offsets[i]
            # The final stream size uses the total byte length of the stream array
            total_bytes = torch.tensor(
                [compressed_stream.numel()], dtype=torch.uint32, device=device
            )

            stream_sizes = compute_stream_sizes(
                stream_offsets, compressed_stream, device
            )

            fused_result = fused_rans_linear_triton_uncoalesced(
                x=x,
                compressed_data=compressed_stream,
                initial_states=_get(layer.rans_exp_states).to(torch.uint32),
                tables=_get(layer.rans_exp_tables),
                slot_map=_get(layer.rans_exp_slot_map),
                weight_shape=(K, N),
                stream_offsets=stream_offsets,
                stream_sizes=stream_sizes,
                tile_k=tile_k,
                tile_n=tile_n,
                mantissas=_get(layer.rans_man_raw),
                bias=_get(bias),
                SPLIT_K=layer.split_k,
                workspace=workspace,
            )
        else:
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
                mantissas=_get(layer.rans_man_raw),
                bias=_get(bias),
                SPLIT_K=layer.split_k,
                workspace=workspace,
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

        layer.uncoalesced_interleaving = layer_settings.get(
            "uncoalesced_interleaving", False
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

        is_uncoalesced = layer.uncoalesced_interleaving

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

        if is_uncoalesced:
            stream_offsets = _get(layer.rans_exp_stream_offsets).to(torch.uint32)
            compressed_stream = _get(layer.rans_exp_stream)

            # stream_sizes[i] = stream_offsets[i+1] - stream_offsets[i]
            # The final stream size uses the total byte length of the stream array
            total_bytes = torch.tensor(
                [compressed_stream.numel()], dtype=torch.uint32, device=device
            )
            stream_sizes = compute_stream_sizes(
                stream_offsets, compressed_stream, device
            )

            fused_result = fused_rans_embedding_triton_uncoalesced(
                x=x,
                compressed_data=compressed_stream,
                initial_states=_get(layer.rans_exp_states).to(torch.uint32),
                tables=_get(layer.rans_exp_tables),
                slot_map=_get(layer.rans_exp_slot_map),
                weight_shape=(K, N),
                stream_offsets=stream_offsets,
                stream_sizes=stream_sizes,  # Passing the dynamically calculated sizes!
                tile_k=tile_k,
                tile_n=tile_n,
                mantissas=_get(layer.rans_man_raw),
            )

        else:
            fused_result = fused_rans_embedding_triton(
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
                mantissas=_get(layer.rans_man_raw),
            )
        return fused_result

    def apply(self, layer, x, bias=None) -> torch.Tensor:
        device = x.device
        N, K = layer.rans_shape
        K_input = x.shape[-1]

        tile_k = layer.tile_height
        tile_n = layer.tile_width

        is_uncoalesced = layer.uncoalesced_interleaving

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

        x_flat = x.view(-1, K_input)
        M = x_flat.shape[0]

        workspace = RansWorkspace.get_workspace(M, K, layer.split_k, device)

        if is_uncoalesced:
            stream_offsets = _get(layer.rans_exp_stream_offsets).to(torch.uint32)
            compressed_stream = _get(layer.rans_exp_stream)

            # stream_sizes[i] = stream_offsets[i+1] - stream_offsets[i]
            # The final stream size uses the total byte length of the stream array
            total_bytes = torch.tensor(
                [compressed_stream.numel()], dtype=torch.uint32, device=device
            )
            stream_sizes = compute_stream_sizes(
                stream_offsets, compressed_stream, device
            )

            fused_result = fused_rans_linear_transposed_triton_uncoalesced(
                x=x,
                compressed_data=compressed_stream,
                initial_states=_get(layer.rans_exp_states).to(torch.uint32),
                tables=_get(layer.rans_exp_tables),
                slot_map=_get(layer.rans_exp_slot_map),
                weight_shape=(K, N),
                stream_offsets=stream_offsets,
                stream_sizes=stream_sizes,
                tile_k=tile_k,
                tile_n=tile_n,
                mantissas=_get(layer.rans_man_raw),
                workspace=workspace,
                SPLIT_K=layer.split_k,
            )

        else:
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
                mantissas=_get(layer.rans_man_raw),
                workspace=workspace,
                SPLIT_K=layer.split_k,
            )
        return fused_result


def patch_logits_processor_for_rans(model):
    """
    Overrides the default vLLM LogitsProcessor to use our fused rANS kernel
    instead of the standard PyTorch F.linear.
    """
    original_forward = model.logits_processor.forward

    def fused_rans_logits_forward(hidden_states, gathered_weight, *args, **kwargs):
        # gathered_weight is usually embed_tokens.weight.
        # We ignore it and grab our compressed parameters directly from the embedding layer!
        embed_layer = model.model.embed_tokens

        # If the embedding layer is rANS compressed
        if hasattr(embed_layer, "rans_exp_stream"):
            device = hidden_states.device
            N, K = embed_layer.rans_shape  # Vocab size, Hidden size

            # The exact same math as your Linear layer: hidden_states @ W^T
            logits = fused_rans_linear_transposed_triton(
                x=hidden_states,
                compressed_data=embed_layer.rans_exp_stream,
                initial_states=embed_layer.rans_exp_states,
                tables=embed_layer.rans_exp_tables,
                slot_map=embed_layer.rans_exp_slot_map,
                weight_shape=(K, N),
                tile_offsets=embed_layer.rans_exp_tile_offsets,
                tile_max_lens=embed_layer.rans_exp_tile_max_lens,
                tile_k=embed_layer.tile_height,
                tile_n=embed_layer.tile_width,
                mantissas=embed_layer.rans_man_raw,
                workspace=RansWorkspace.get_workspace(
                    hidden_states.shape[0], K, embed_layer.split_k, device
                ),
                SPLIT_K=embed_layer.split_k,
            )
            return logits

        # Fallback for uncompressed models
        return original_forward(hidden_states, gathered_weight, *args, **kwargs)

    # Apply the patch
    model.logits_processor.forward = fused_rans_logits_forward
