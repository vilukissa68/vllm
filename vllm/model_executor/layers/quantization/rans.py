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
        # elif isinstance(layer, VocabParallelEmbedding):
        #    return RansEmbeddingMethod(self)
        return None


# info_list = loaded_weight.cpu().tolist()
# layer.rans_expanded_size = info_list[1]
# layer.rans_exp_compressed = bool(info_list[2])
# layer.rans_exp_num_streams = info_list[3]
# layer.rans_man_compressed = bool(info_list[4])
# layer.rans_man_num_streams = info_list[5]
# layer.rans_rank = info_list[6]
# layer.num_tiles_n = info_list[7]
# layer.num_tiles_k = info_list[8]
# layer.rans_shape = torch.Size(info_list[9 : 9 + layer.rans_rank])


def register_rans_parameters(layer: torch.nn.Module, output_partition_sizes: list):
    # Determine TP sharding context
    n_local = sum(output_partition_sizes)
    tp_rank = get_tensor_model_parallel_rank()

    # Pre-calculate tile-based sharding indices
    # These are used to slice the GLOBAL tensors into LOCAL shards
    layer.nl = n_local // 32
    layer.ts = tp_rank * layer.nl
    layer.te = layer.ts + layer.nl

    def rans_loader(param, loaded_weight, shard_id=None):
        param_name = getattr(param, "rans_name", "")

        # nk: rows of tiles, ng: global columns of tiles
        nk = layer.nk
        ng = layer.ng
        ts, te = layer.ts, layer.te

        # --- 1. Metadata Slicing (Offsets / MaxLens) ---
        if any(x in param_name for x in ["offsets", "max_lens"]):
            expected_global = nk * ng
            # Strip sentinel if exists, then slice 2D vertical strip
            grid = loaded_weight[:expected_global].view(nk, ng)
            loaded_weight = grid[:, ts:te].contiguous().view(-1)

        # --- 2. States Slicing (States are per-tile-lane) ---
        elif "states" in param_name:
            # Global: [nk, ng, 32_lanes]
            grid = loaded_weight.view(nk, ng, 32)
            loaded_weight = grid[:, ts:te, :].contiguous().view(-1)

        # --- 3. Mantissas Slicing (K-major interleaved) ---
        elif "man_raw" in param_name:
            # Global: [nk, ng, 1024_K, 32_N]
            grid = loaded_weight.view(nk, ng, 1024, 32)
            loaded_weight = grid[:, ts:te, :, :].contiguous().view(-1)

        # Move to target device (handles CPU offloading automatically)
        target_device = param.device
        final_weight = loaded_weight.to(target_device)
        if target_device.type == "cpu":
            final_weight = final_weight.pin_memory()

        if param.numel() != final_weight.numel():
            param.data = final_weight
        else:
            param.data.copy_(final_weight)

        # --- 4. Info Metadata Extraction ---
        if "info" in param_name:
            info = loaded_weight.cpu().tolist()
            # info[9:11] stores (K, N) global.
            # We store the LOCAL shard size for the apply method
            rank = info[6]
            global_shape = info[9 : 9 + rank]
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
        # 1. Fix Tiling Constants
        layer.nk = (input_size + 1023) // 1024
        layer.ng = (output_size + 31) // 32

        # 2. Initialize logical shape to avoid bootstrap errors
        layer.rans_shape = (input_size, sum(output_partition_sizes))

        # 3. Register
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
        print(f"Applying RANS Linear: Input shape {x.shape}, Expected K={K}, N={N}")

        assert (
            N == x.shape[-1]
        ), f"Input feature dimension K mismatch: {N} vs {x.shape[-1]}"

        # Debug Prints
        print(f"--- Layer Debug ---")
        print(f"  Input X: {list(x.shape)}, dtype: {x.dtype}")
        print(f"  K_actual: {x.shape[-1]}, N_local: {layer.rans_exp_states.numel()}")

        print(f"  RANS shape: ({layer.rans_shape[1]}, {layer.rans_shape[0]})")

        # Sanity check for NaNs in input
        if torch.isnan(x).any():
            print("  WARNING: Input X contains NaNs!")

        # Helper to move tensors
        def _get(p):
            return p.to(device, non_blocking=True) if p is not None else None

        # 2. Launch Fused Kernel
        fused_result = fused_rans_linear_triton(
            x=x,
            compressed_data=_get(layer.rans_exp_stream),
            tile_offsets=_get(layer.rans_exp_tile_offsets).to(torch.uint32),
            tile_max_lens=_get(layer.rans_exp_tile_max_lens).to(torch.uint32),
            initial_states=_get(layer.rans_exp_states).to(torch.uint32),
            mantissas=_get(layer.rans_man_raw),
            slot_map=_get(layer.rans_exp_slot_map),
            tables=_get(layer.rans_exp_tables),
            # K=K,
            # N=N,
            bias=_get(bias),
        )

        DEBUG_MODE = True
        if DEBUG_MODE:
            # nk: rows of tiles (e.g., 2), nl: local tiles wide (e.g., 32)
            nk = layer.nk
            nl = N // 32

            decomp_exp_local = rans_decomp_triton_tiled(
                compressed_streams=_get(layer.rans_exp_stream).to(torch.uint8),
                initial_states=_get(layer.rans_exp_states).to(torch.uint32),
                tables=_get(layer.rans_exp_tables).to(torch.uint32),
                slot_map=_get(layer.rans_exp_slot_map).to(torch.uint16),
                output_shape=(K, N),
                tile_offsets=_get(layer.rans_exp_tile_offsets).to(torch.uint32),
                tile_max_lens=_get(layer.rans_exp_tile_max_lens).to(torch.uint32),
                tile_k=1024,
                tile_n=32,
            )

            # B. Separate Mantissa Uninterleaving
            decomp_man_local = uninterleave_mantissas(
                _get(layer.rans_man_raw), K, N, TILE_K=1024, TILE_N=32
            )

            # C. Weight Reconstruction
            reassembled_weight_local = reconstruct_from_exp_and_mantissa(
                decomp_exp_local, decomp_man_local, dtype=torch.bfloat16
            )

            reference_result = torch.nn.functional.linear(
                x, reassembled_weight_local, bias
            )

            # --- Comparison & Analysis ---
            diff = torch.abs(fused_result - reference_result)
            max_diff = diff.max().item()

            # Use a print that identifies the specific layer dimensions
            print(f"[DEBUG {K}x{N}] Max Diff: {max_diff:.6f}")

            if not torch.allclose(fused_result, reference_result, atol=1e-2):
                print(f"CRITICAL: Divergence detected in {K}x{N} layer!")
                print("Fused Sample:", fused_result)
                print("Ref Sample:  ", reference_result)
                exit(1)
            else:
                print(f"SUCCESS: {K}x{N} fused kernel matches reference.")

            # Return the reference during debug to isolate if kernel is broken or logic is broken
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
