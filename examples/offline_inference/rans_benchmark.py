#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import time
import os

import pynvml
import threading
import time


class PowerMonitor:
    def __init__(self, gpu_id=0, interval=0.01):
        """
        Monitors GPU power consumption in a background thread.
        interval: Time in seconds between power samples (default 10ms).
        """
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.interval = interval
        self.is_recording = False
        self.power_readings = []  # Stores Watts
        self.thread = None

    def _record(self):
        while self.is_recording:
            # NVML returns power in milliwatts
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.power_readings.append(power_mw / 1000.0)
            time.sleep(self.interval)

    def start(self):
        self.power_readings = []
        self.is_recording = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def stop(self):
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()

        if not self.power_readings:
            return 0.0, 0.0

        avg_power = sum(self.power_readings) / len(self.power_readings)
        # Energy (Joules) = Power (Watts) * Time (Seconds)
        total_energy_joules = sum(p * self.interval for p in self.power_readings)

        return avg_power, total_energy_joules


def run_engine(mode, model_path, args, result_dict):
    """
    Runs the vLLM engine inside an isolated process to ensure perfect
    VRAM cleanup between the baseline and compressed runs.
    """
    # Force the isolated process to respect the GPU targeting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Import vLLM ONLY inside the subprocess so it initializes cleanly
    from vllm import LLM, SamplingParams
    import torch

    num_gpus = len(args.gpus.split(","))

    # 1. Get the actual physical memory of the target GPU (e.g., 48GB for A6000)
    total_gpu_vram = torch.cuda.get_device_properties(0).total_memory
    total_allowed_vram = total_gpu_vram * args.vram_util
    weight_budget_bytes = int(total_allowed_vram * 0.80)

    # 4. Broadcast the budget to your custom backend
    os.environ["RANS_WEIGHT_BUDGET_BYTES"] = str(weight_budget_bytes)
    print(f"\n{'='*60}")
    print(f"🚀 STARTING {mode.upper()} BENCHMARK")
    print(f"Model: {model_path}")
    print(f"VRAM Budget: {args.vram_util * 100:.1f}%")
    if mode == "baseline":
        print(f"CPU Offload Pool: {args.cpu_offload} GB")
    print(f"{'='*60}")

    # 1. Initialize Engine
    if mode == "rans":
        llm = LLM(
            model=model_path,
            quantization="rans",
            dtype="bfloat16",
            enforce_eager=True,  # Critical: Disable CUDA graphs
            trust_remote_code=True,
            gpu_memory_utilization=args.vram_util,
            # cpu_offload_gb=args.cpu_offload,
            max_model_len=args.max_len,
            tensor_parallel_size=num_gpus,
        )
    else:
        llm = LLM(
            model=model_path,  # Load from huggingface for baseline to ensure no compression
            dtype="bfloat16",
            enforce_eager=True,  # Disabled to maintain fairness
            trust_remote_code=True,
            gpu_memory_utilization=args.vram_util,
            cpu_offload_gb=args.cpu_offload,
            max_model_len=args.max_len,
            tensor_parallel_size=num_gpus,
        )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # 2. Warm-up Phase
    print("\n[Running Warmup...]")
    for _ in range(args.warmup_runs):
        llm.generate(
            ["Hello world, please reply."],
            SamplingParams(max_tokens=10),
            use_tqdm=False,
        )

    # --- VERIFICATION SNAPSHOT ---
    torch.cuda.synchronize()
    vram_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
    vram_reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"\n[VRAM VERIFICATION]")
    print(f"  Allocated (Weights + KV Cache active): {vram_used_gb:.2f} GB")
    print(f"  Reserved (vLLM total claim):           {vram_reserved_gb:.2f} GB")
    # -----------------------------

    # 3. Evaluation Phase
    print(f"\n[Evaluating for {args.eval_runs} runs...]")
    total_decode_toks_per_sec = 0.0
    final_text = ""

    power_monitor = PowerMonitor(gpu_id=0, interval=0.01)

    total_avg_power = 0.0
    total_energy = 0.0
    total_tokens_generated = 0

    for i in range(args.eval_runs):
        power_monitor.start()

        start_time = time.perf_counter()
        outputs = llm.generate([args.prompt], sampling_params, use_tqdm=False)
        end_time = time.perf_counter()

        avg_power, energy_joules = power_monitor.stop()

        output = outputs[0]
        final_text = output.outputs[0].text
        generated_tokens = len(output.outputs[0].token_ids)
        total_tokens_generated += generated_tokens

        metrics = output.metrics
        if metrics is not None and metrics.first_token_time is not None:
            decode_time = metrics.finished_time - metrics.first_token_time
        else:
            decode_time = end_time - start_time

        decode_toks_sec = generated_tokens / decode_time if decode_time > 0 else 0
        total_decode_toks_per_sec += decode_toks_sec
        total_avg_power += avg_power
        total_energy += energy_joules

        print(
            f"  Run {i+1}: {decode_toks_sec:.2f} tokens/s | Power: {avg_power:.1f} W | Energy: {energy_joules:.1f} Joules"
        )

    avg_toks_sec = total_decode_toks_per_sec / args.eval_runs
    avg_power_overall = total_avg_power / args.eval_runs
    energy_per_token = (
        total_energy / total_tokens_generated if total_tokens_generated > 0 else 0
    )

    # Save results to the shared dictionary
    result_dict["avg_toks_sec"] = avg_toks_sec
    result_dict["avg_power"] = avg_power_overall
    result_dict["energy_per_token"] = energy_per_token
    result_dict["vram_used_gb"] = vram_used_gb
    result_dict["vram_reserved_gb"] = vram_reserved_gb

    print(f"\n✅ {mode.upper()} AVERAGE: {avg_toks_sec:.2f} tokens/s")
    print(f"   Average Power: {avg_power_overall:.1f} W")
    print(f"   Energy per Token: {energy_per_token:.2f} Joules/token")

    # Aggressive teardown (though the process dying will handle most of this)
    del llm
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="rANS vs Baseline CPU Offload Benchmark"
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Path/Hub ID for the baseline model",
    )
    parser.add_argument(
        "--compressed_model",
        type=str,
        required=True,
        help="Path to the rANS compressed model folder",
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES string"
    )
    parser.add_argument(
        "--vram_util",
        type=float,
        default=0.416,
        help="GPU memory utilization (e.g., 0.416 for ~10GB on a 24GB GPU)",
    )
    parser.add_argument(
        "--cpu_offload",
        type=float,
        default=40.0,
        help="GB of CPU RAM for baseline offloading",
    )
    parser.add_argument("--max_len", type=int, default=2048, help="Max model length")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the history and architecture of deep learning accelerators in extreme detail.",
        help="Input text",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128, help="Max output tokens to generate"
    )
    parser.add_argument(
        "--warmup_runs", type=int, default=2, help="Number of warmup runs"
    )
    parser.add_argument(
        "--eval_runs", type=int, default=3, help="Number of evaluated runs to average"
    )
    args = parser.parse_args()

    res_baseline = {}
    res_rans = {}
    run_engine("baseline", args.baseline_model, args, res_baseline)
    run_engine("rans", args.compressed_model, args, res_rans)
    print(f"\n{'='*60}")
    print("🏆 FINAL BENCHMARK SHOWDOWN 🏆")
    print(f"{'='*60}")
    print(f"VRAM Budget:    {args.vram_util * 100:.1f}%")
    print(f'Prompt:         "{args.prompt[:40]}..."')
    print("-" * 60)
    print(
        f"BASELINE Speed: {res_baseline['avg_toks_sec']:>6.2f} tokens/s (Uncompressed Offloading)"
    )
    print(
        f"RANS Speed:     {res_rans['avg_toks_sec']:>6.2f} tokens/s (Compressed + Triton)"
    )
    print("-" * 60)

    if res_baseline["avg_toks_sec"] > 0:
        speedup = res_rans["avg_toks_sec"] / res_baseline["avg_toks_sec"]
        print(f"🚀 MULTIPLIER:  {speedup:.2f}x Faster Inference!")
    else:
        print("Error calculating multiplier: Baseline speed was 0.")
    print(f"{'='*60}\n")

    # Print warmup vram utilization
    print("VRAM Utilization check")
    print(
        f"Baseline: VRAM Allocated: {res_baseline['vram_used_gb']}, VRAM reserverd: {res_baseline['vram_allocated_gb']}"
    )
    print(
        f"RANS: VRAM Allocated: {res_rans['vram_used_gb']}, VRAM reserverd: {res_rans['vram_allocated_gb']}"
    )


if __name__ == "__main__":
    main()
