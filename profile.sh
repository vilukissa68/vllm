PROMPT="Who is Linus Torvalds?"

# Define the command as an array
LAUNCH_CMD=(python examples/offline_inference/rans_example.py ../comp_inference/model/ --prompt "$PROMPT" --quant)

nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    "${LAUNCH_CMD[@]}"   
