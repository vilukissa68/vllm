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


# --- HELPER: DYNAMIC MODEL SIZE ---
def get_dynamic_model_size(model_path: str) -> int:
    total_bytes = 0
    if os.path.exists(model_path):
        for root, _, files in os.walk(model_path):
            for file in files:
                if file.endswith(".safetensors") or file.endswith(".bin"):
                    total_bytes += os.path.getsize(os.path.join(root, file))
    else:
        try:
            from huggingface_hub import model_info

            info = model_info(model_path, files_metadata=True)
            for sibling in info.siblings:
                if (
                    sibling.rfilename.endswith(".safetensors")
                    or sibling.rfilename.endswith(".bin")
                ) and sibling.size is not None:
                    total_bytes += sibling.size
        except Exception as e:
            print(f"[Warn] Could not determine size for {model_path}: {e}")
            return 0
    return total_bytes


# --- HELPER: POWER MONITOR ---
class PowerMonitor:
    def __init__(self, gpu_id=0, interval=0.1):
        self.interval = interval
        self.is_recording = False
        self.power_readings = []
        self.thread = None
        self.gpu_id = gpu_id
        self.available = False

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.available = True
            except Exception:
                pass

    def _record(self):
        while self.is_recording and self.available:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_readings.append(power_mw / 1000.0)
            except:
                pass
            time.sleep(self.interval)

    def start(self):
        self.power_readings = []
        self.is_recording = True
        if self.available:
            self.thread = threading.Thread(target=self._record)
            self.thread.start()

    def stop(self):
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()

        if not self.power_readings:
            return 0.0, 0.0

        avg_power = sum(self.power_readings) / len(self.power_readings)
        total_energy_joules = sum(p * self.interval for p in self.power_readings)
        return avg_power, total_energy_joules


# --- WORKER FUNCTION ---
def benchmark_worker(
    gpu_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    args_dict: Dict,
    model_meta: Dict,
):
    """
    Worker process that owns a specific GPU and processes benchmark tasks.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ[
        "VLLM_TORCH_COMPILE_LEVEL"
    ] = "0"  # Disable dynamo to protect rANS kernel

    # Import vLLM here to avoid CUDA context issues in parent
    import vllm
    from vllm import LLM, SamplingParams

    # Re-init CUDA for this process
    if torch.cuda.is_available():
        torch.cuda.init()
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        _ = torch.zeros(1, device="cuda")
        free_mem, total_vram_bytes = torch.cuda.mem_get_info()
        cuda_context_overhead = total_vram_bytes - free_mem
    else:
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")

    power_monitor = PowerMonitor(gpu_id=0)  # It sees itself as device 0 now

    # --- STATIC OVERHEAD CALCULATION (Independent of mode) ---
    print(f"[GPU {gpu_id}] Calculating static memory overhead...")
    hf_config = AutoConfig.from_pretrained(
        args_dict["baseline_model"], trust_remote_code=True
    )

    # 1. Calculate Exact KV Cache Bytes required for the largest sweep
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

    bytes_per_token = (
        2 * 2 * num_layers * num_kv_heads * head_dim
    )  # 2 (K+V) * 2 bytes (fp16)
    max_tokens_needed = max(args_dict["batch_sizes"]) * (
        max(args_dict["prompt_lens"]) + max(args_dict["gen_lens"])
    )
    required_kv_bytes = max_tokens_needed * bytes_per_token
    required_kv_bytes = max(required_kv_bytes, int(args_dict["kv_cache"] * (1024**3)))

    # 2. Estimate Activation Memory (Hidden states during prefill)
    max_prefill_tokens = max(args_dict["batch_sizes"]) * max(args_dict["prompt_lens"])
    activation_bytes = max_prefill_tokens * hf_config.hidden_size * 2 * 4

    # 3. Base fragmentation buffer
    vllm_fragmentation_buffer = int(total_vram_bytes * 0.005)

    # Pre-extract MLP dim for rANS workspace calculation later
    mlp_dim = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)

    print("   Static Overhead Components:")
    print(f"   - CUDA Context: {cuda_context_overhead/1e9:.2f} GB")
    print(f"   - Max KV Cache: {required_kv_bytes/1e9:.2f} GB")
    print(f"   - Activation Buffer: {activation_bytes/1e9:.2f} GB")
    print(f"   - vLLM Fragmentation Buffer: {vllm_fragmentation_buffer/1e9:.2f} GB")

    # --- TASK LOOP ---
    while True:
        task = task_queue.get()
        if task is None:  # Sentinel
            break

        mode, vram_util = task
        print(f"[GPU {gpu_id}] Starting task: {mode} @ {vram_util*100:.0f}% VRAM")

        # --- DYNAMIC OVERHEAD (Dependent on mode) ---
        if "rans" in mode:
            # Matches your backend: MAX_BATCH_GUESS = 8192, Split-K = 8
            rans_workspace_bytes = 8 * 8192 * mlp_dim * 2
        else:
            rans_workspace_bytes = 0

        DYNAMIC_RESERVE_BYTES = (
            cuda_context_overhead
            + required_kv_bytes
            + activation_bytes
            + rans_workspace_bytes
            + vllm_fragmentation_buffer
        )

        print(
            f"[GPU {gpu_id}] Dynamic Reserve Computed: {DYNAMIC_RESERVE_BYTES / 1e9:.2f} GB "
            f"(Context: {cuda_context_overhead/1e9:.2f}G, KV: {required_kv_bytes/1e9:.2f}G, "
            f"Acts: {activation_bytes/1e9:.2f}G, rANS WS: {rans_workspace_bytes/1e9:.2f}G)"
        )

        # --- EXACT SPILLOVER MATH ---
        target_gpu_bytes = int(total_vram_bytes * vram_util)

        if "rans" in mode:
            os.environ["USE_RANS_JIT"] = "1" if "unfused" in mode else "0"
            model_path = args_dict["compressed_model"]
            model_bytes = model_meta["rans_bytes"]

            # The weights allowed on GPU for rANS
            allowed_weight_budget = max(0, target_gpu_bytes - DYNAMIC_RESERVE_BYTES)
            os.environ["RANS_WEIGHT_BUDGET_BYTES"] = str(allowed_weight_budget)
        else:  # Baseline
            model_path = args_dict["baseline_model"]
            model_bytes = model_meta["base_bytes"]

        # Calculate exact spillover required to fit into the target util boundary
        actual_spillover_bytes = max(
            0, model_bytes + DYNAMIC_RESERVE_BYTES - target_gpu_bytes
        )
        # Give 1% Error for spill_over to ensure vLLM definitely doesn't OOM
        actual_spillover_bytes = int(actual_spillover_bytes * 1.01)
        actual_spillover_gb = actual_spillover_bytes / (1024**3)

        print(
            f"[GPU {gpu_id}] Model Size: {model_bytes/1e9:.2f} GB | Target GPU Budget: {target_gpu_bytes/1e9:.2f} GB | "
            f"Calculated Spillover: {actual_spillover_gb:.2f} GB"
        )

        print(
            f" required_kv_bytes: {required_kv_bytes/1e9:.2f} GB | activation_bytes: {activation_bytes/1e9:.2f} GB | "
        )
        print(
            f" rANS workspace: {rans_workspace_bytes/1e9:.2f} GB | Fragmentation buffer: {vllm_fragmentation_buffer/1e9:.2f} GB"
        )
        print(f" Total Dynamic Reserve: {DYNAMIC_RESERVE_BYTES/1e9:.2f} GB")

        # Clean slate
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

        # --- INITIALIZE ENGINE ---
        try:
            if args_dict["no_offload"]:
                if mode == "baseline":
                    llm = LLM(
                        model=model_path,
                        quantization="rans" if "rans" in mode else None,
                        dtype="bfloat16",
                        enforce_eager=True,
                        trust_remote_code=True,
                        max_model_len=max(args_dict["prompt_lens"])
                        + max(args_dict["gen_lens"])
                        + 128,
                        tensor_parallel_size=1,
                        disable_log_stats=True,
                    )
            else:
                if mode == "baseline":
                    llm = LLM(
                        model=model_path,
                        dtype="bfloat16",
                        enforce_eager=True,
                        trust_remote_code=True,
                        gpu_memory_utilization=vram_util,
                        # vLLM interprets this as EXACTLY how much to offload
                        cpu_offload_gb=actual_spillover_gb,
                        max_model_len=max(args_dict["prompt_lens"])
                        + max(args_dict["gen_lens"])
                        + 128,
                        tensor_parallel_size=1,
                        disable_log_stats=True,
                    )
                else:
                    llm = LLM(
                        model=model_path,
                        quantization="rans",
                        dtype="bfloat16",
                        enforce_eager=True,
                        trust_remote_code=True,
                        gpu_memory_utilization=vram_util,
                        # cpu_offload_gb=actual_spillover_gb, # Managed by RANS budget
                        max_model_len=max(args_dict["prompt_lens"])
                        + max(args_dict["gen_lens"])
                        + 128,
                        tensor_parallel_size=1,
                        disable_log_stats=True,
                    )

            # Extract Cache Stats (Tokens & GB)
            try:
                cache_config = llm.llm_engine.cache_config
                kv_cache_tokens = cache_config.num_gpu_blocks * cache_config.block_size

                hf_config = llm.llm_engine.model_config.hf_config
                num_layers = getattr(
                    hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 0)
                )
                num_kv_heads = getattr(
                    hf_config,
                    "num_key_value_heads",
                    getattr(hf_config, "num_attention_heads", 1),
                )

                if hasattr(hf_config, "head_dim"):
                    head_dim = hf_config.head_dim
                else:
                    head_dim = hf_config.hidden_size // getattr(
                        hf_config, "num_attention_heads", 1
                    )

                bytes_per_token = 4 * num_layers * num_kv_heads * head_dim
                kv_cache_gb = (kv_cache_tokens * bytes_per_token) / (1024**3)

            except Exception as e:
                print(f"[Warn] Could not calculate KV Cache specs: {e}")
                kv_cache_tokens = 0
                kv_cache_gb = 0.0

            # --- RUN SWEEPS ---
            for bs in args_dict["batch_sizes"]:
                for pl in args_dict["prompt_lens"]:
                    for gl in args_dict["gen_lens"]:
                        req_tokens = bs * (pl + gl)
                        if kv_cache_tokens > 0 and req_tokens > kv_cache_tokens:
                            continue

                        prompts = [{"prompt_token_ids": [100] * pl} for _ in range(bs)]
                        sp = SamplingParams(
                            max_tokens=gl, temperature=0.0, ignore_eos=True
                        )

                        try:
                            llm.generate(
                                prompts=[{"prompt_token_ids": [100] * 10}],
                                sampling_params=SamplingParams(max_tokens=2),
                                use_tqdm=False,
                            )
                        except:
                            pass

                        latencies = []
                        powers = []
                        energies = []

                        torch.cuda.reset_peak_memory_stats()

                        try:
                            for _ in range(args_dict["eval_runs"]):
                                power_monitor.start()
                                t0 = time.perf_counter()
                                req_outputs = llm.generate(
                                    prompts=prompts, sampling_params=sp, use_tqdm=False
                                )
                                t1 = time.perf_counter()
                                p_avg, e_total = power_monitor.stop()

                                gen_tokens = sum(
                                    len(o.outputs[0].token_ids) for o in req_outputs
                                )
                                if gen_tokens > 0:
                                    latencies.append(gen_tokens / (t1 - t0))
                                powers.append(p_avg)
                                energies.append(e_total)

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
                                "avg_toks_sec": float(np.mean(latencies))
                                if latencies
                                else 0.0,
                                "avg_power_w": float(np.mean(powers))
                                if powers
                                else 0.0,
                                "energy_j": float(np.mean(energies))
                                if energies
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
            print(f"[GPU {gpu_id}] Engine Init Failed for {mode}: {e}")

    print(f"[GPU {gpu_id}] Worker Finished.")


# --- MAIN ---
def main():
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--compressed_model", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0")

    parser.add_argument(
        "--modes",
        type=str,
        default="rans_fused,baseline,rans_unfused",
        help="Comma-separated modes to run",
    )

    parser.add_argument(
        "--vram_utils", type=str, default="0.9", help="Comma list: 0.8,0.9"
    )
    parser.add_argument("--batch_sizes", type=str, default="1,4")
    parser.add_argument("--prompt_lens", type=str, default="128")
    parser.add_argument("--gen_lens", type=str, default="128")
    parser.add_argument("--eval_runs", type=int, default=3)
    parser.add_argument(
        "--kv_cache", type=float, default=2.5, help="GB reserved for KV cache"
    )
    parser.add_argument(
        "--no_offload",
        action="store_true",
        help="Disable explicit CPU offload (for rANS)",
    )

    args = parser.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",")]
    vram_utils = [float(x) for x in args.vram_utils.split(",")]
    active_modes = [x.strip() for x in args.modes.split(",")]

    args_dict = vars(args)
    args_dict["vram_utils"] = vram_utils
    args_dict["batch_sizes"] = [int(x) for x in args.batch_sizes.split(",")]
    args_dict["prompt_lens"] = [int(x) for x in args.prompt_lens.split(",")]
    args_dict["gen_lens"] = [int(x) for x in args.gen_lens.split(",")]

    print(f"🚀 STARTING PARALLEL BENCHMARK on GPUs: {gpu_list}")
    print(f"🎯 Modes active: {active_modes}")
    print("[1/2] Analyzing Models...")

    base_bytes = get_dynamic_model_size(args.baseline_model)
    rans_bytes = get_dynamic_model_size(args.compressed_model)
    model_meta = {"base_bytes": base_bytes, "rans_bytes": rans_bytes}

    print(f"   Baseline: {base_bytes/1e9:.2f} GB")
    print(f"   rANS:     {rans_bytes/1e9:.2f} GB")

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    tasks = []
    for util in vram_utils:
        for mode in active_modes:
            tasks.append((mode, util))

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
            "compressed_model": args.compressed_model,
            "baseline_bytes": base_bytes,
            "rans_bytes": rans_bytes,
        },
        "results": [],
    }

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)

    completed_records = 0

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
                    f"   [REC] {record['mode']} (Util: {record['vram_util_config']}) "
                    f"-> {record['avg_toks_sec']:.1f} t/s | Actual Offload: {record['cpu_offload_gb']:.1f}GB"
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

    print(f"\n✅ Sweep Complete. {completed_records} records saved to {filename}")


if __name__ == "__main__":
    main()
