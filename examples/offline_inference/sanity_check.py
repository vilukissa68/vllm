#!/usr/bin/env python3
#
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(model="facebook/opt-125m", enforce_eager=False)
    print(llm.generate("Hello world"))
    print("Sanity check passed!")
