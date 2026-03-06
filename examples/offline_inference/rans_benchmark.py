#!/usr/bin/env python3
import os
import sys
import argparse
import time
import json
import gc
import threading
import torch
import numpy as np
import multiprocessing
from datetime import datetime
from typing import Dict, Any, List
from transformers import AutoConfig

# Try importing pynvml for power monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# def get_dynamic_model_size(model_path: str) -> int:
#     safetensors_bytes = 0
#     bin_bytes = 0

#     if os.path.exists(model_path):
#         for root, _, files in os.walk(model_path):
#             for file in files:
#                 filepath = os.path.join(root, file)
#                 if file.endswith(".safetensors"):
#                     safetensors_bytes += os.path.getsize(filepath)
#                 # Ignore small config bins like training_args.bin
#                 elif file.endswith(".bin") and not file.startswith("training_args"):
#                     bin_bytes += os.path.getsize(filepath)
#     else:
#         try:
#             from huggingface_hub import model_info

#             info = model_info(model_path, files_metadata=True)
#             for sibling in info.siblings:
#                 if sibling.size is not None:
#                     if sibling.rfilename.endswith(".safetensors"):
#                         safetensors_bytes += sibling.size
#                     elif sibling.rfilename.endswith(".bin"):
#                         bin_bytes += sibling.size
#         except Exception as e:
#             print(f"[Warn] Could not determine size for {model_path}: {e}")
#             return 0

#     # vLLM will load safetensors if they exist. If not, it falls back to bin.
#     # We return the size of the format that will actually be loaded in memory.
#     if safetensors_bytes > 0:
#         return safetensors_bytes
#     return bin_bytes

# import os
# import json


# def get_dynamic_model_size(model_path: str, target_dtype: str = "bfloat16") -> int:
#     safetensors_bytes = 0
#     bin_bytes = 0
#     seen_files = set()

#     if os.path.exists(model_path):
#         for root, _, files in os.walk(model_path):
#             for file in files:
#                 filepath = os.path.join(root, file)
#                 real_filepath = os.path.realpath(filepath)

#                 if real_filepath in seen_files:
#                     continue
#                 seen_files.add(real_filepath)

#                 if file.endswith(".safetensors"):
#                     safetensors_bytes += os.path.getsize(real_filepath)
#                 elif file.endswith(".bin") and not file.startswith("training_args"):
#                     bin_bytes += os.path.getsize(real_filepath)
#     else:
#         # Fallback for HF Hub
#         try:
#             from huggingface_hub import model_info

#             info = model_info(model_path, files_metadata=True)
#             for sibling in info.siblings:
#                 if sibling.size is not None:
#                     if sibling.rfilename.endswith(".safetensors"):
#                         safetensors_bytes += sibling.size
#                     elif sibling.rfilename.endswith(".bin"):
#                         bin_bytes += sibling.size
#         except Exception as e:
#             print(f"[Warn] Could not determine size for {model_path}: {e}")
#             return 0

#     raw_size = safetensors_bytes if safetensors_bytes > 0 else bin_bytes

#     # --- THE FIX: Adjust for Downcasting ---
#     config_path = os.path.join(model_path, "config.json")
#     native_dtype = "float32"  # Assume worst case if no config found
#     if "qwen3" in model_path.lower():
#         native_dtype = "bfloat16"

#     if os.path.exists(config_path):
#         try:
#             with open(config_path, "r") as f:
#                 config = json.load(f)
#                 native_dtype = config.get("dtype") or config.get(
#                     "torch_dtype", "float32"
#                 )
#         except Exception:
#             pass

#     if native_dtype == "float32" and target_dtype in ["bfloat16", "float16"]:
#         print(
#             f"📉 Downcast detected: {native_dtype} on disk -> {target_dtype} in VRAM. Halving size estimate."
#         )
#         raw_size = raw_size // 2

#     return raw_size


# # --- HELPER: RAPL CPU/RAM POWER MONITOR ---
# class RAPLMonitor:
#     """Reads Intel/AMD RAPL energy counters directly from Linux sysfs."""

#     def __init__(self):
#         self.cpu_paths = []
#         self.ram_paths = []
#         self.available = False

#         base_dir = "/sys/class/powercap/"
#         if not os.path.exists(base_dir):
#             return

#         try:
#             for d in os.listdir(base_dir):
#                 if d.startswith("intel_rapl:"):
#                     pkg_dir = os.path.join(base_dir, d)
#                     name_file = os.path.join(pkg_dir, "name")
#                     energy_file = os.path.join(pkg_dir, "energy_uj")

#                     # Identify CPU Package
#                     if os.path.exists(name_file) and os.path.exists(energy_file):
#                         with open(name_file, "r") as f:
#                             name = f.read().strip()
#                         if "package" in name:
#                             self.cpu_paths.append(energy_file)

#                     # Identify DRAM (System RAM)
#                     for sub_d in os.listdir(pkg_dir):
#                         if sub_d.startswith(f"{d}:"):
#                             sub_dir = os.path.join(pkg_dir, sub_d)
#                             sub_name_file = os.path.join(sub_dir, "name")
#                             sub_energy_file = os.path.join(sub_dir, "energy_uj")

#                             if os.path.exists(sub_name_file) and os.path.exists(
#                                 sub_energy_file
#                             ):
#                                 with open(sub_name_file, "r") as f:
#                                     sub_name = f.read().strip()
#                                 if "dram" in sub_name:
#                                     self.ram_paths.append(sub_energy_file)

#             self.available = len(self.cpu_paths) > 0
#             if self.available:
#                 print(
#                     f"[RAPL] Found {len(self.cpu_paths)} CPU(s) and {len(self.ram_paths)} RAM node(s)."
#                 )
#         except Exception as e:
#             print(f"[Warn] RAPL initialization failed: {e}")
#             self.available = False

#     def _read_energy(self, paths):
#         total_uj = 0
#         for p in paths:
#             try:
#                 with open(p, "r") as f:
#                     total_uj += int(f.read().strip())
#             except PermissionError:
#                 print(f"[Warn] Permission denied reading RAPL. See instructions below.")
#                 self.available = False
#             except Exception:
#                 pass
#         return total_uj

#     def get_energy_uj(self):
#         return self._read_energy(self.cpu_paths), self._read_energy(self.ram_paths)


# # --- HELPER: COMBINED POWER MONITOR ---
# class PowerMonitor:
#     def __init__(self, gpu_id=0, interval=0.1):
#         self.interval = interval
#         self.is_recording = False
#         self.power_readings = []
#         self.thread = None
#         self.gpu_id = gpu_id
#         self.gpu_available = False

#         # Initialize RAPL Monitor
#         self.rapl = RAPLMonitor()
#         self.start_cpu_uj = 0
#         self.start_ram_uj = 0

#         if PYNVML_AVAILABLE:
#             try:
#                 pynvml.nvmlInit()
#                 self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
#                 self.gpu_available = True
#             except Exception:
#                 pass

#     def _record_gpu(self):
#         while self.is_recording and self.gpu_available:
#             try:
#                 power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
#                 self.power_readings.append(power_mw / 1000.0)
#             except:
#                 pass
#             time.sleep(self.interval)

#     def start(self):
#         self.power_readings = []

#         # Snapshot RAPL accumulators
#         if self.rapl.available:
#             self.start_cpu_uj, self.start_ram_uj = self.rapl.get_energy_uj()

#         self.is_recording = True
#         if self.gpu_available:
#             self.thread = threading.Thread(target=self._record_gpu)
#             self.thread.start()

#     def stop(self):
#         self.is_recording = False
#         if self.thread is not None:
#             self.thread.join()

#         # 1. Calculate GPU Energy (via polling integration)
#         avg_gpu_power = 0.0
#         gpu_energy_j = 0.0
#         if self.power_readings:
#             avg_gpu_power = sum(self.power_readings) / len(self.power_readings)
#             gpu_energy_j = sum(p * self.interval for p in self.power_readings)

#         # 2. Calculate CPU & RAM Energy (via hardware accumulators)
#         cpu_energy_j = 0.0
#         ram_energy_j = 0.0
#         if self.rapl.available:
#             end_cpu_uj, end_ram_uj = self.rapl.get_energy_uj()

#             # Handle hardware counter wrap-around (rare but possible)
#             cpu_diff_uj = max(0, end_cpu_uj - self.start_cpu_uj)
#             ram_diff_uj = max(0, end_ram_uj - self.start_ram_uj)

#             # Convert microjoules to joules
#             cpu_energy_j = cpu_diff_uj / 1_000_000.0
#             ram_energy_j = ram_diff_uj / 1_000_000.0

#         return avg_gpu_power, gpu_energy_j, cpu_energy_j, ram_energy_j


# # --- WORKER FUNCTION ---
# def benchmark_worker(
#     gpu_id: int,
#     task_queue: multiprocessing.Queue,
#     result_queue: multiprocessing.Queue,
#     args_dict: Dict,
#     model_meta: Dict,
# ):
#     """
#     Worker process that owns a specific GPU and processes benchmark tasks.
#     """
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     os.environ[
#         "VLLM_TORCH_COMPILE_LEVEL"
#     ] = "0"  # Disable dynamo to protect rANS kernel

#     # Import vLLM here to avoid CUDA context issues in parent
#     import vllm
#     from vllm import LLM, SamplingParams
#     from vllm.model_executor.layers.quantization.rans import (
#         patch_logits_processor_for_rans,
#     )

#     # Re-init CUDA for this process
#     if torch.cuda.is_available():
#         torch.cuda.init()
#         total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
#         _ = torch.zeros(1, device="cuda")
#         free_mem, total_vram_bytes = torch.cuda.mem_get_info()
#         cuda_context_overhead = total_vram_bytes - free_mem
#     else:
#         raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")

#     power_monitor = PowerMonitor(gpu_id=0)  # It sees itself as device 0 now

#     # --- STATIC OVERHEAD CALCULATION (Independent of mode) ---
#     print(f"[GPU {gpu_id}] Calculating static memory overhead...")
#     hf_config = AutoConfig.from_pretrained(
#         args_dict["baseline_model"], trust_remote_code=True
#     )

#     # 1. Calculate Exact KV Cache Bytes required for the largest sweep
#     num_layers = getattr(
#         hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 0)
#     )
#     num_kv_heads = getattr(
#         hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads", 1)
#     )
#     head_dim = getattr(
#         hf_config,
#         "head_dim",
#         hf_config.hidden_size // getattr(hf_config, "num_attention_heads", 1),
#     )

#     bytes_per_token = (
#         2 * 2 * num_layers * num_kv_heads * head_dim
#     )  # 2 (K+V) * 2 bytes (fp16)
#     max_tokens_needed = max(args_dict["batch_sizes"]) * (
#         max(args_dict["prompt_lens"]) + max(args_dict["gen_lens"])
#     )
#     required_kv_bytes = max_tokens_needed * bytes_per_token
#     required_kv_bytes = max(required_kv_bytes, int(args_dict["kv_cache"] * (1024**3)))

#     # Ensure we get the correct hidden size dimension
#     hidden_size = getattr(hf_config, "hidden_size", getattr(hf_config, "d_model", 4096))
#     if hidden_size == 0:
#         hidden_size = 5120

#     # 2. Estimate Activation Memory (Hidden states during prefill)
#     max_prefill_tokens = max(args_dict["batch_sizes"]) * max(args_dict["prompt_lens"])
#     activation_bytes = max_prefill_tokens * hidden_size * 2 * 4
#     activation_bytes = max_prefill_tokens * hf_config.hidden_size * 2 * 4

#     # 3. Base fragmentation buffer
#     vllm_fragmentation_buffer = int(total_vram_bytes * 0.04)

#     # Pre-extract MLP dim for rANS workspace calculation later
#     mlp_dim = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)

#     print("   Static Overhead Components:")
#     print(f"   - CUDA Context: {cuda_context_overhead/1e9:.2f} GB")
#     print(f"   - Max KV Cache: {required_kv_bytes/1e9:.2f} GB")
#     print(f"   - Activation Buffer: {activation_bytes/1e9:.2f} GB")
#     print(f"   - vLLM Fragmentation Buffer: {vllm_fragmentation_buffer/1e9:.2f} GB")

#     # --- TASK LOOP ---
#     while True:
#         task = task_queue.get()
#         if task is None:  # Sentinel
#             break

#         mode, vram_util = task
#         print(f"[GPU {gpu_id}] Starting task: {mode} @ {vram_util*100:.0f}% VRAM")

#         # --- DYNAMIC OVERHEAD (Dependent on mode) ---
#         if "rans" in mode:
#             # Matches your backend: MAX_BATCH_GUESS = 8192, Split-K = 8
#             rans_workspace_bytes = 8 * 8192 * mlp_dim * 2
#         else:
#             rans_workspace_bytes = 0

#         DYNAMIC_RESERVE_BYTES = (
#             cuda_context_overhead
#             + required_kv_bytes
#             + activation_bytes
#             + rans_workspace_bytes
#             + vllm_fragmentation_buffer
#         )

#         print(
#             f"[GPU {gpu_id}] Dynamic Reserve Computed: {DYNAMIC_RESERVE_BYTES / 1e9:.2f} GB "
#             f"(Context: {cuda_context_overhead/1e9:.2f}G, KV: {required_kv_bytes/1e9:.2f}G, "
#             f"Acts: {activation_bytes/1e9:.2f}G, rANS WS: {rans_workspace_bytes/1e9:.2f}G)"
#         )

#         # --- EXACT SPILLOVER MATH ---
#         target_gpu_bytes = int(total_vram_bytes * vram_util)

#         if "rans" in mode:
#             os.environ["USE_RANS_JIT"] = "1" if "unfused" in mode else "0"
#             model_path = args_dict["compressed_model"]
#             model_bytes = model_meta["rans_bytes"]

#             # The weights allowed on GPU for rANS
#             allowed_weight_budget = max(0, target_gpu_bytes - DYNAMIC_RESERVE_BYTES)
#             os.environ["RANS_WEIGHT_BUDGET_BYTES"] = str(allowed_weight_budget)
#         else:  # Baseline
#             model_path = args_dict["baseline_model"]
#             model_bytes = model_meta["base_bytes"] + 1

#         # Calculate exact spillover required to fit into the target util boundary
#         actual_spillover_bytes = max(
#             0, model_bytes + DYNAMIC_RESERVE_BYTES - target_gpu_bytes
#         )
#         # Give 1% Error for spill_over to ensure vLLM definitely doesn't OOM
#         actual_spillover_bytes = int(actual_spillover_bytes * 1.01)
#         actual_spillover_gb = actual_spillover_bytes / (1024**3)

#         print(
#             f"[GPU {gpu_id}] Model Size: {model_bytes/1e9:.2f} GB | Target GPU Budget: {target_gpu_bytes/1e9:.2f} GB | "
#             f"Calculated Spillover: {actual_spillover_gb:.2f} GB"
#         )

#         print(
#             f" required_kv_bytes: {required_kv_bytes/1e9:.2f} GB | activation_bytes: {activation_bytes/1e9:.2f} GB | "
#         )
#         print(
#             f" rANS workspace: {rans_workspace_bytes/1e9:.2f} GB | Fragmentation buffer: {vllm_fragmentation_buffer/1e9:.2f} GB"
#         )
#         print(f" Total Dynamic Reserve: {DYNAMIC_RESERVE_BYTES/1e9:.2f} GB")

#         # Clean slate
#         if torch.cuda.is_available():
#             torch.cuda.reset_peak_memory_stats()
#             torch.cuda.empty_cache()
#             gc.collect()

#         if mode == "baseline":
#             quantization = None
#         elif mode == "triton_baseline":
#             quantization = "triton_baseline"
#         elif mode == "rans_fused":
#             quantization = "rans"
#         elif mode == "rans_unfused":
#             quantization = "rans"
#         else:
#             print(f"[GPU {gpu_id}] Unknown mode: {mode}")
#             continue

#         # --- INITIALIZE ENGINE ---
#         try:
#             if args_dict["no_offload"]:
#                 llm = LLM(
#                     model=model_path,
#                     quantization=quantization,
#                     dtype="bfloat16",
#                     enforce_eager=True,
#                     trust_remote_code=True,
#                     max_model_len=max(args_dict["prompt_lens"])
#                     + max(args_dict["gen_lens"])
#                     + 128,
#                     tensor_parallel_size=1,
#                     disable_log_stats=True,
#                     enable_chunked_prefill=False,
#                 )
#             else:
#                 if mode == "baseline":
#                     llm = LLM(
#                         model=model_path,
#                         dtype="bfloat16",
#                         enforce_eager=True,
#                         trust_remote_code=True,
#                         gpu_memory_utilization=vram_util,
#                         # vLLM interprets this as EXACTLY how much to offload
#                         cpu_offload_gb=actual_spillover_gb,
#                         max_model_len=max(args_dict["prompt_lens"])
#                         + max(args_dict["gen_lens"])
#                         + 128,
#                         tensor_parallel_size=1,
#                         disable_log_stats=True,
#                         enable_chunked_prefill=False,
#                     )
#                 elif mode == "triton_baseline":
#                     llm = LLM(
#                         model=model_path,
#                         quantization="triton_baseline",
#                         dtype="bfloat16",
#                         enforce_eager=True,
#                         trust_remote_code=True,
#                         gpu_memory_utilization=vram_util,
#                         # vLLM interprets this as EXACTLY how much to offload
#                         cpu_offload_gb=actual_spillover_gb,
#                         max_model_len=max(args_dict["prompt_lens"])
#                         + max(args_dict["gen_lens"])
#                         + 128,
#                         tensor_parallel_size=1,
#                         disable_log_stats=True,
#                         enable_chunked_prefill=False,
#                     )
#                 else:
#                     llm = LLM(
#                         model=model_path,
#                         quantization="rans",
#                         dtype="bfloat16",
#                         enforce_eager=True,
#                         trust_remote_code=True,
#                         gpu_memory_utilization=vram_util,
#                         # cpu_offload_gb=actual_spillover_gb, # Managed by RANS budget
#                         max_model_len=max(args_dict["prompt_lens"])
#                         + max(args_dict["gen_lens"])
#                         + 128,
#                         tensor_parallel_size=1,
#                         disable_log_stats=True,
#                         enable_chunked_prefill=False,
#                     )

#             # Extract Cache Stats (Tokens & GB)
#             try:
#                 cache_config = llm.llm_engine.cache_config
#                 kv_cache_tokens = cache_config.num_gpu_blocks * cache_config.block_size

#                 hf_config = llm.llm_engine.model_config.hf_config
#                 num_layers = getattr(
#                     hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 0)
#                 )
#                 num_kv_heads = getattr(
#                     hf_config,
#                     "num_key_value_heads",
#                     getattr(hf_config, "num_attention_heads", 1),
#                 )

#                 if hasattr(hf_config, "head_dim"):
#                     head_dim = hf_config.head_dim
#                 else:
#                     head_dim = hf_config.hidden_size // getattr(
#                         hf_config, "num_attention_heads", 1
#                     )

#                 bytes_per_token = 4 * num_layers * num_kv_heads * head_dim
#                 kv_cache_gb = (kv_cache_tokens * bytes_per_token) / (1024**3)

#             except Exception as e:
#                 print(f"[Warn] Could not calculate KV Cache specs: {e}")
#                 kv_cache_tokens = 0
#                 kv_cache_gb = 0.0

#             # --- RUN SWEEPS ---
#             for bs in args_dict["batch_sizes"]:
#                 for pl in args_dict["prompt_lens"]:
#                     for gl in args_dict["gen_lens"]:
#                         req_tokens = bs * (pl + gl)
#                         if kv_cache_tokens > 0 and req_tokens > kv_cache_tokens:
#                             continue

#                         prompts = [{"prompt_token_ids": [100] * pl} for _ in range(bs)]
#                         sp = SamplingParams(
#                             max_tokens=gl, temperature=0.0, ignore_eos=True
#                         )

#                         try:
#                             llm.generate(
#                                 prompts=[{"prompt_token_ids": [100] * 10}],
#                                 sampling_params=SamplingParams(max_tokens=2),
#                                 use_tqdm=False,
#                             )
#                         except:
#                             pass

#                         latencies = []
#                         gpu_powers = []
#                         gpu_energies = []
#                         cpu_energies = []
#                         ram_energies = []

#                         torch.cuda.reset_peak_memory_stats()

#                         try:
#                             for _ in range(args_dict["eval_runs"]):
#                                 power_monitor.start()
#                                 t0 = time.perf_counter()
#                                 req_outputs = llm.generate(
#                                     prompts=prompts, sampling_params=sp, use_tqdm=False
#                                 )
#                                 t1 = time.perf_counter()
#                                 (
#                                     avg_gpu_power,
#                                     gpu_energy,
#                                     cpu_energy,
#                                     ram_energy,
#                                 ) = power_monitor.stop()

#                                 gen_tokens = sum(
#                                     len(o.outputs[0].token_ids) for o in req_outputs
#                                 )
#                                 if gen_tokens > 0:
#                                     latencies.append(gen_tokens / (t1 - t0))
#                                 gpu_powers.append(avg_gpu_power)
#                                 gpu_energies.append(gpu_energy)
#                                 cpu_energies.append(cpu_energy)
#                                 ram_energies.append(ram_energy)

#                             peak_allocated = torch.cuda.max_memory_allocated()
#                             peak_reserved = torch.cuda.max_memory_reserved()

#                             result_record = {
#                                 "mode": mode,
#                                 "vram_util_config": vram_util,
#                                 "total_vram_gb": total_vram_bytes / 1e9,
#                                 "batch_size": bs,
#                                 "prompt_len": pl,
#                                 "gen_len": gl,
#                                 "kv_cache_tokens": kv_cache_tokens,
#                                 "kv_cache_gb": kv_cache_gb,
#                                 "cpu_offload_gb": actual_spillover_gb,
#                                 "avg_toks_sec": float(np.mean(latencies))
#                                 if latencies
#                                 else 0.0,
#                                 "avg_power_w": float(np.mean(gpu_powers))
#                                 if gpu_powers
#                                 else 0.0,
#                                 "gpu_energy_j": float(np.mean(gpu_energies))
#                                 if gpu_energies
#                                 else 0.0,
#                                 "cpu_energy_j": float(np.mean(cpu_energies))
#                                 if cpu_energies
#                                 else 0.0,
#                                 "ram_energy_j": float(np.mean(ram_energies))
#                                 if ram_energies
#                                 else 0.0,
#                                 "peak_allocated_gb": peak_allocated / (1024**3),
#                                 "peak_reserved_gb": peak_reserved / (1024**3),
#                             }

#                             result_queue.put(result_record)

#                         except Exception as e:
#                             print(f"[GPU {gpu_id}] Run Failed: {e}")
#                             torch.cuda.empty_cache()

#             del llm
#             gc.collect()
#             torch.cuda.empty_cache()

#             try:
#                 from vllm.distributed.parallel_state import (
#                     destroy_model_parallel,
#                     destroy_distributed_environment,
#                 )

#                 destroy_model_parallel()
#                 destroy_distributed_environment()
#             except:
#                 pass

#         except Exception as e:
#             print(f"[GPU {gpu_id}] Engine Init Failed for {mode}: {e}")

#     print(f"[GPU {gpu_id}] Worker Finished.")


# # --- MAIN ---
# def main():
#     multiprocessing.set_start_method("spawn", force=True)

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--baseline_model", type=str, default="Qwen/Qwen3-14B")
#     parser.add_argument("--compressed_model", type=str, required=True)
#     parser.add_argument("--gpus", type=str, default="0")

#     parser.add_argument(
#         "--modes",
#         type=str,
#         default="rans_fused,baseline,rans_unfused",
#         help="Comma-separated modes to run",
#     )

#     parser.add_argument(
#         "--vram_utils", type=str, default="0.9", help="Comma list: 0.8,0.9"
#     )
#     parser.add_argument("--batch_sizes", type=str, default="1,4")
#     parser.add_argument("--prompt_lens", type=str, default="128")
#     parser.add_argument("--gen_lens", type=str, default="128")
#     parser.add_argument("--eval_runs", type=int, default=3)
#     parser.add_argument(
#         "--kv_cache", type=float, default=2.5, help="GB reserved for KV cache"
#     )
#     parser.add_argument(
#         "--no_offload",
#         action="store_true",
#         help="Disable explicit CPU offload (for rANS)",
#     )

#     args = parser.parse_args()

#     gpu_list = [int(x) for x in args.gpus.split(",")]
#     vram_utils = [float(x) for x in args.vram_utils.split(",")]
#     active_modes = [x.strip() for x in args.modes.split(",")]

#     args_dict = vars(args)
#     args_dict["vram_utils"] = vram_utils
#     args_dict["batch_sizes"] = [int(x) for x in args.batch_sizes.split(",")]
#     args_dict["prompt_lens"] = [int(x) for x in args.prompt_lens.split(",")]
#     args_dict["gen_lens"] = [int(x) for x in args.gen_lens.split(",")]

#     print(f"🚀 STARTING PARALLEL BENCHMARK on GPUs: {gpu_list}")
#     print(f"🎯 Modes active: {active_modes}")
#     print("[1/2] Analyzing Models...")

#     base_bytes = get_dynamic_model_size(args.baseline_model)
#     rans_bytes = get_dynamic_model_size(args.compressed_model)
#     model_meta = {"base_bytes": base_bytes, "rans_bytes": rans_bytes}

#     print(f"   Baseline: {base_bytes/1e9:.2f} GB")
#     print(f"   rANS:     {rans_bytes/1e9:.2f} GB")

#     task_queue = multiprocessing.Queue()
#     result_queue = multiprocessing.Queue()

#     tasks = []
#     for util in vram_utils:
#         for mode in active_modes:
#             tasks.append((mode, util))

#     for t in tasks:
#         task_queue.put(t)

#     for _ in gpu_list:
#         task_queue.put(None)

#     workers = []
#     for gpu_id in gpu_list:
#         p = multiprocessing.Process(
#             target=benchmark_worker,
#             args=(gpu_id, task_queue, result_queue, args_dict, model_meta),
#         )
#         p.start()
#         workers.append(p)

#     filename = f"rans_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     output_data = {
#         "metadata": {
#             "timestamp": datetime.now().isoformat(),
#             "baseline_model": args.baseline_model,
#             "compressed_model": args.compressed_model,
#             "baseline_bytes": base_bytes,
#             "rans_bytes": rans_bytes,
#         },
#         "results": [],
#     }

#     with open(filename, "w") as f:
#         json.dump(output_data, f, indent=4)

#     completed_records = 0

#     try:
#         while any(p.is_alive() for p in workers) or not result_queue.empty():
#             try:
#                 record = result_queue.get(timeout=1.0)
#                 output_data["results"].append(record)
#                 completed_records += 1

#                 with open(filename, "w") as f:
#                     json.dump(output_data, f, indent=4)
#                     f.flush()

#                 print(
#                     f"   [REC] {record['mode']} (Util: {record['vram_util_config']}) "
#                     f"-> {record['avg_toks_sec']:.1f} t/s | Actual Offload: {record['cpu_offload_gb']:.1f}GB"
#                 )

#             except multiprocessing.queues.Empty:
#                 continue

#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted! Terminating workers...")
#         for p in workers:
#             p.terminate()
#         for p in workers:
#             p.join()
#         sys.exit(1)

#     print(f"\n✅ Sweep Complete. {completed_records} records saved to {filename}")


# if __name__ == "__main__":
#     main()


import os
import json
import gc
import sys
import time
import threading
import multiprocessing
import numpy as np
import argparse
from typing import Dict
from datetime import datetime

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[Warn] pynvml not installed. GPU power metrics will be 0.")

import torch
from transformers import AutoConfig


def get_dynamic_model_size(model_path: str, target_dtype: str = "bfloat16") -> int:
    safetensors_bytes = 0
    bin_bytes = 0
    seen_files = set()

    if os.path.exists(model_path):
        for root, _, files in os.walk(model_path):
            for file in files:
                filepath = os.path.join(root, file)
                real_filepath = os.path.realpath(filepath)

                if real_filepath in seen_files:
                    continue
                seen_files.add(real_filepath)

                if file.endswith(".safetensors"):
                    safetensors_bytes += os.path.getsize(real_filepath)
                elif file.endswith(".bin") and not file.startswith("training_args"):
                    bin_bytes += os.path.getsize(real_filepath)
    else:
        try:
            from huggingface_hub import model_info

            info = model_info(model_path, files_metadata=True)
            for sibling in info.siblings:
                if sibling.size is not None:
                    if sibling.rfilename.endswith(".safetensors"):
                        safetensors_bytes += sibling.size
                    elif sibling.rfilename.endswith(".bin"):
                        bin_bytes += sibling.size
        except Exception as e:
            print(f"[Warn] Could not determine size for {model_path}: {e}")
            return 0

    raw_size = safetensors_bytes if safetensors_bytes > 0 else bin_bytes

    config_path = os.path.join(model_path, "config.json")
    native_dtype = "float32"
    if "qwen3" in model_path.lower():
        native_dtype = "bfloat16"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                native_dtype = config.get("dtype") or config.get(
                    "torch_dtype", "float32"
                )
        except Exception:
            pass

    if native_dtype == "float32" and target_dtype in ["bfloat16", "float16"]:
        print(
            f"📉 Downcast detected: {native_dtype} on disk -> {target_dtype} in VRAM. Halving size estimate."
        )
        raw_size = raw_size // 2

    return raw_size


class RAPLMonitor:
    def __init__(self):
        self.cpu_paths = []
        self.ram_paths = []
        self.available = False
        base_dir = "/sys/class/powercap/"
        if not os.path.exists(base_dir):
            return
        try:
            for d in os.listdir(base_dir):
                if d.startswith("intel_rapl:"):
                    pkg_dir = os.path.join(base_dir, d)
                    name_file = os.path.join(pkg_dir, "name")
                    energy_file = os.path.join(pkg_dir, "energy_uj")
                    if os.path.exists(name_file) and os.path.exists(energy_file):
                        with open(name_file, "r") as f:
                            name = f.read().strip()
                        if "package" in name:
                            self.cpu_paths.append(energy_file)
                    for sub_d in os.listdir(pkg_dir):
                        if sub_d.startswith(f"{d}:"):
                            sub_dir = os.path.join(pkg_dir, sub_d)
                            sub_name_file = os.path.join(sub_dir, "name")
                            sub_energy_file = os.path.join(sub_dir, "energy_uj")
                            if os.path.exists(sub_name_file) and os.path.exists(
                                sub_energy_file
                            ):
                                with open(sub_name_file, "r") as f:
                                    sub_name = f.read().strip()
                                if "dram" in sub_name:
                                    self.ram_paths.append(sub_energy_file)
            self.available = len(self.cpu_paths) > 0
        except Exception:
            self.available = False

    def _read_energy(self, paths):
        total_uj = 0
        for p in paths:
            try:
                with open(p, "r") as f:
                    total_uj += int(f.read().strip())
            except Exception:
                pass
        return total_uj

    def get_energy_uj(self):
        return self._read_energy(self.cpu_paths), self._read_energy(self.ram_paths)


class PowerMonitor:
    def __init__(self, gpu_id=0, interval=0.1):
        self.interval = interval
        self.is_recording = False
        self.power_readings = []
        self.thread = None
        self.gpu_id = gpu_id
        self.gpu_available = False
        self.rapl = RAPLMonitor()
        self.start_cpu_uj = 0
        self.start_ram_uj = 0
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.gpu_available = True
            except Exception:
                pass

    def _record_gpu(self):
        while self.is_recording and self.gpu_available:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_readings.append(power_mw / 1000.0)
            except:
                pass
            time.sleep(self.interval)

    def start(self):
        self.power_readings = []
        if self.rapl.available:
            self.start_cpu_uj, self.start_ram_uj = self.rapl.get_energy_uj()
        self.is_recording = True
        if self.gpu_available:
            self.thread = threading.Thread(target=self._record_gpu)
            self.thread.start()

    def stop(self):
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()
        avg_gpu_power = (
            sum(self.power_readings) / len(self.power_readings)
            if self.power_readings
            else 0.0
        )
        gpu_energy_j = (
            sum(p * self.interval for p in self.power_readings)
            if self.power_readings
            else 0.0
        )
        cpu_energy_j, ram_energy_j = 0.0, 0.0
        if self.rapl.available:
            end_cpu_uj, end_ram_uj = self.rapl.get_energy_uj()
            cpu_energy_j = max(0, end_cpu_uj - self.start_cpu_uj) / 1e6
            ram_energy_j = max(0, end_ram_uj - self.start_ram_uj) / 1e6
        return avg_gpu_power, gpu_energy_j, cpu_energy_j, ram_energy_j


# --- WORKER FUNCTION ---
def benchmark_worker(
    gpu_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    args_dict: Dict,
    model_meta: Dict,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.quantization.rans import (
        patch_logits_processor_for_rans,
    )

    if torch.cuda.is_available():
        torch.cuda.init()
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        _ = torch.zeros(1, device="cuda")
        free_mem, total_vram_bytes = torch.cuda.mem_get_info()
        cuda_context_overhead = total_vram_bytes - free_mem
    else:
        raise RuntimeError("CUDA is not available.")

    power_monitor = PowerMonitor(gpu_id=0)
    hf_config = AutoConfig.from_pretrained(
        args_dict["baseline_model"], trust_remote_code=True
    )

    num_layers = getattr(
        hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 0)
    )
    num_kv_heads = getattr(
        hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads", 1)
    )
    head_dim = getattr(
        hf_config,
        "head_dim",
        hf_config.hidden_size // getattr(hf_config, "num_attention_heads", 1),
    )
    bytes_per_token = 2 * 2 * num_layers * num_kv_heads * head_dim
    hidden_size = (
        getattr(hf_config, "hidden_size", getattr(hf_config, "d_model", 4096)) or 5120
    )
    vllm_fragmentation_buffer = int(total_vram_bytes * 0.04)
    mlp_dim = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)

    while True:
        task = task_queue.get()
        if task is None:
            break

        # --- THE FIX: Unpack the fully flattened task tuple ---
        mode, vram_util, bs, pl, gl = task

        print(
            f"\n[GPU {gpu_id}] Starting task: {mode} | Util: {vram_util*100:.0f}% | BS: {bs} | PL: {pl} | GL: {gl}"
        )

        # --- DYNAMIC OVERHEAD RE-CALCULATED PER TASK ---
        max_tokens_needed = bs * (pl + gl)
        required_kv_bytes = max_tokens_needed * bytes_per_token
        # Respect the user's hard floor limit if provided, but default to exactly what is needed
        required_kv_bytes = max(
            required_kv_bytes, int(args_dict.get("kv_cache", 0) * (1024**3))
        )

        activation_bytes = bs * pl * hidden_size * 2 * 4
        rans_workspace_bytes = 8 * 8192 * mlp_dim * 2 if "rans" in mode else 0

        DYNAMIC_RESERVE_BYTES = (
            cuda_context_overhead
            + required_kv_bytes
            + activation_bytes
            + rans_workspace_bytes
            + vllm_fragmentation_buffer
        )
        target_gpu_bytes = int(total_vram_bytes * vram_util)

        if "rans" in mode:
            os.environ["USE_RANS_JIT"] = "1" if "unfused" in mode else "0"
            quantization_method = "rans"

            if "uncoal" in mode:
                model_path = args_dict["model_uncoal"]
                model_bytes = model_meta["uncoal_bytes"]
            else:
                model_path = args_dict["model_coal"]
                model_bytes = model_meta["coal_bytes"]

            allowed_weight_budget = max(0, target_gpu_bytes - DYNAMIC_RESERVE_BYTES)
            os.environ["RANS_WEIGHT_BUDGET_BYTES"] = str(allowed_weight_budget)
        else:
            model_path = args_dict["baseline_model"]
            model_bytes = model_meta["base_bytes"] + 1
            quantization_method = None if mode == "baseline" else "triton_baseline"

        actual_spillover_bytes = max(
            0, model_bytes + DYNAMIC_RESERVE_BYTES - target_gpu_bytes
        )
        actual_spillover_bytes = int(actual_spillover_bytes * 1.01)
        actual_spillover_gb = actual_spillover_bytes / (1024**3)

        print(
            f"[GPU {gpu_id}] Dynamic Reserve: {DYNAMIC_RESERVE_BYTES/1e9:.2f} GB (KV: {required_kv_bytes/1e9:.2f} GB, Act: {activation_bytes/1e9:.2f} GB)"
        )
        print(
            f"[GPU {gpu_id}] Target Util: {target_gpu_bytes/1e9:.2f} GB | CPU Spillover: {actual_spillover_gb:.2f} GB"
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

        try:
            if args_dict["no_offload"]:
                llm = LLM(
                    model=model_path,
                    quantization=quantization_method,
                    dtype="bfloat16",
                    enforce_eager=True,
                    trust_remote_code=True,
                    max_model_len=pl + gl + 128,
                    tensor_parallel_size=1,
                    disable_log_stats=True,
                    enable_chunked_prefill=False,
                )
            else:
                llm = LLM(
                    model=model_path,
                    quantization=quantization_method,
                    dtype="bfloat16",
                    enforce_eager=True,
                    trust_remote_code=True,
                    gpu_memory_utilization=vram_util,
                    cpu_offload_gb=actual_spillover_gb if "rans" not in mode else 0.0,
                    max_model_len=pl + gl + 128,
                    tensor_parallel_size=1,
                    disable_log_stats=True,
                    enable_chunked_prefill=False,
                )

            try:
                cache_config = llm.llm_engine.cache_config
                kv_cache_tokens = cache_config.num_gpu_blocks * cache_config.block_size
                kv_cache_gb = (kv_cache_tokens * bytes_per_token) / (1024**3)
            except Exception:
                kv_cache_tokens = 0
                kv_cache_gb = 0.0

            req_tokens = bs * (pl + gl)
            if kv_cache_tokens > 0 and req_tokens > kv_cache_tokens:
                print(
                    f"[GPU {gpu_id}] SKIP: Requested tokens ({req_tokens}) exceeds available KV Cache ({kv_cache_tokens})."
                )
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                continue

            prompts = [{"prompt_token_ids": [100] * pl} for _ in range(bs)]
            sp = SamplingParams(max_tokens=gl, temperature=0.0, ignore_eos=True)

            try:
                llm.generate(
                    prompts=[{"prompt_token_ids": [100] * 10}],
                    sampling_params=SamplingParams(max_tokens=2),
                    use_tqdm=False,
                )
            except:
                pass

            latencies, gpu_powers, gpu_energies, cpu_energies, ram_energies = (
                [],
                [],
                [],
                [],
                [],
            )
            torch.cuda.reset_peak_memory_stats()

            try:
                for _ in range(args_dict["eval_runs"]):
                    power_monitor.start()
                    t0 = time.perf_counter()
                    req_outputs = llm.generate(
                        prompts=prompts, sampling_params=sp, use_tqdm=False
                    )
                    t1 = time.perf_counter()
                    (
                        avg_gpu_power,
                        gpu_energy,
                        cpu_energy,
                        ram_energy,
                    ) = power_monitor.stop()

                    gen_tokens = sum(len(o.outputs[0].token_ids) for o in req_outputs)
                    if gen_tokens > 0:
                        latencies.append(gen_tokens / (t1 - t0))
                    gpu_powers.append(avg_gpu_power)
                    gpu_energies.append(gpu_energy)
                    cpu_energies.append(cpu_energy)
                    ram_energies.append(ram_energy)

                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()

                result_record = {
                    "mode": mode,
                    "vram_util_config": vram_util,
                    "total_vram_gb": total_vram_bytes / 1e9,
                    "batch_size": bs,
                    "prompt_len": pl,
                    "gen_len": gl,
                    "kv_cache_tokens": kv_cache_tokens,
                    "kv_cache_gb": kv_cache_gb,
                    "cpu_offload_gb": actual_spillover_gb,
                    "avg_toks_sec": float(np.mean(latencies)) if latencies else 0.0,
                    "avg_power_w": float(np.mean(gpu_powers)) if gpu_powers else 0.0,
                    "gpu_energy_j": float(np.mean(gpu_energies))
                    if gpu_energies
                    else 0.0,
                    "cpu_energy_j": float(np.mean(cpu_energies))
                    if cpu_energies
                    else 0.0,
                    "ram_energy_j": float(np.mean(ram_energies))
                    if ram_energies
                    else 0.0,
                    "peak_allocated_gb": peak_allocated / (1024**3),
                    "peak_reserved_gb": peak_reserved / (1024**3),
                }
                result_queue.put(result_record)
            except Exception as e:
                print(f"[GPU {gpu_id}] Run Failed: {e}")
                torch.cuda.empty_cache()

            del llm
            gc.collect()
            torch.cuda.empty_cache()

            try:
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel,
                    destroy_distributed_environment,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
            except:
                pass

        except Exception as e:
            print(f"[GPU {gpu_id}] Engine Init Failed for {mode} (BS:{bs}): {e}")

    print(f"[GPU {gpu_id}] Worker Finished.")


def main():
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--model_coal",
        type=str,
        required=True,
        help="Path to coalesced/padded rANS model",
    )
    parser.add_argument(
        "--model_uncoal",
        type=str,
        required=True,
        help="Path to uncoalesced/dense rANS model",
    )
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,rans_coal,rans_uncoal",
        help="Comma-separated modes. Options: baseline, rans_coal, rans_uncoal",
    )
    parser.add_argument(
        "--vram_utils", type=str, default="0.9", help="Comma list: 0.8,0.9"
    )
    parser.add_argument("--batch_sizes", type=str, default="1,4")
    parser.add_argument("--prompt_lens", type=str, default="128")
    parser.add_argument("--gen_lens", type=str, default="128")
    parser.add_argument("--eval_runs", type=int, default=3)
    parser.add_argument(
        "--kv_cache",
        type=float,
        default=0.0,
        help="Minimum GB reserved for KV cache (defaults to 0.0 to allow exact fit)",
    )
    parser.add_argument(
        "--no_offload", action="store_true", help="Disable explicit CPU offload"
    )

    args = parser.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",")]
    vram_utils = [float(x) for x in args.vram_utils.split(",")]
    active_modes = [x.strip() for x in args.modes.split(",")]

    args_dict = vars(args)
    args_dict["vram_utils"] = vram_utils
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    prompt_lens = [int(x) for x in args.prompt_lens.split(",")]
    gen_lens = [int(x) for x in args.gen_lens.split(",")]

    print(f"🚀 STARTING PARALLEL BENCHMARK on GPUs: {gpu_list}")
    print(f"🎯 Modes active: {active_modes}")
    print("[1/2] Analyzing Models...")

    base_bytes = get_dynamic_model_size(args.baseline_model)
    coal_bytes = get_dynamic_model_size(args.model_coal)
    uncoal_bytes = get_dynamic_model_size(args.model_uncoal)

    model_meta = {
        "base_bytes": base_bytes,
        "coal_bytes": coal_bytes,
        "uncoal_bytes": uncoal_bytes,
    }

    print(f"   Baseline:    {base_bytes/1e9:.2f} GB")
    print(f"   rANS Coal:   {coal_bytes/1e9:.2f} GB")
    print(f"   rANS Uncoal: {uncoal_bytes/1e9:.2f} GB")

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # --- THE FIX: Flatten the loops into the task queue ---
    tasks = []
    for util in vram_utils:
        for mode in active_modes:
            for bs in batch_sizes:
                for pl in prompt_lens:
                    for gl in gen_lens:
                        tasks.append((mode, util, bs, pl, gl))

    for t in tasks:
        task_queue.put(t)
    for _ in gpu_list:
        task_queue.put(None)

    workers = []
    for gpu_id in gpu_list:
        p = multiprocessing.Process(
            target=benchmark_worker,
            args=(gpu_id, task_queue, result_queue, args_dict, model_meta),
        )
        p.start()
        workers.append(p)

    filename = f"rans_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "baseline_model": args.baseline_model,
            "model_coal": args.model_coal,
            "model_uncoal": args.model_uncoal,
            "baseline_bytes": base_bytes,
            "coal_bytes": coal_bytes,
            "uncoal_bytes": uncoal_bytes,
        },
        "results": [],
    }

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)

    completed_records = 0
    total_tasks = len(tasks)
    try:
        while any(p.is_alive() for p in workers) or not result_queue.empty():
            try:
                record = result_queue.get(timeout=1.0)
                output_data["results"].append(record)
                completed_records += 1

                with open(filename, "w") as f:
                    json.dump(output_data, f, indent=4)
                    f.flush()

                print(
                    f"\n✅ [{completed_records}/{total_tasks}] COMPLETED: {record['mode']} "
                    f"(Util: {record['vram_util_config']} | BS: {record['batch_size']}) "
                    f"-> {record['avg_toks_sec']:.1f} t/s | Offload: {record['cpu_offload_gb']:.1f}GB"
                )
            except multiprocessing.queues.Empty:
                continue

    except KeyboardInterrupt:
        print("\n🛑 Interrupted! Terminating workers...")
        for p in workers:
            p.terminate()
        for p in workers:
            p.join()
        sys.exit(1)

    print(f"\n🎉 Sweep Complete. {completed_records} records saved to {filename}")


if __name__ == "__main__":
    main()
