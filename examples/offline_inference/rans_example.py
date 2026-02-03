#!/usr/bin/env python3

import argparse
import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Run RANS-vLLM Inference")
    parser.add_argument(
        "model_path", type=str, help="Path to the compressed model folder"
    )
    parser.add_argument(
        "--prompt", type=str, default="The capital of Finland is", help="Input text"
    )
    parser.add_argument("--max_tokens", type=int, default=50, help="Max output tokens")
    parser.add_argument(
        "--temp", type=float, default=0.0, help="Temperature (0.0 = greedy)"
    )
    parser.add_argument("--quant", action="store_true", help="Use quantization (rans)")
    args = parser.parse_args()

    print(f"--- Initializing RANS vLLM Engine ---")
    print(f"Model: {args.model_path}")

    # 1. Initialize the Engine
    # This triggers your RansLinearMethod.create_weights logic
    llm = LLM(
        model=args.model_path,
        tokenizer="Qwen/Qwen3-0.6B",
        quantization="rans" if args.quant else None,  # <--- Triggers your backend
        dtype="bfloat16",  # Match your compression dtype
        enforce_eager=True,  # CRITICAL: Disable CUDA Graphs for dynamic loading
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,  # Change if you have multi-gpu
    )

    # 2. Define Sampling
    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.max_tokens)

    # 3. Generate
    print(f"--- Generating ---")
    print(f"Prompt: {args.prompt}")

    outputs = llm.generate([args.prompt], sampling_params)

    # 4. Print Result
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"\n[Output]:\n{generated_text}")

        # Optional: Print stats if you want to verify speed
        # print(f"\nStats: {output.metrics}")


if __name__ == "__main__":
    main()
