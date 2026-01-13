#!/usr/bin/env python3
#
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    print("Starting sanity check...")
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        enforce_eager=True,
    )
    print(llm.generate("Hello world"))
    print("Sanity check passed!")
