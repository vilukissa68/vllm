#PROMPT="Who is Linus Torvalds?"
PROMPT="What is the capital of France?"
MODEL="Qwen/Qwen3-0.6B"

# Define the command as an array
#LAUNCH_CMD=(python examples/offline_inference/rans_example.py ../comp_inference/model/ --prompt "$PROMPT" --quant)
LAUNCH_CMD=(python examples/offline_inference/rans_example.py "$MODEL" --prompt "$PROMPT" --quant)

nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    "${LAUNCH_CMD[@]}"   
