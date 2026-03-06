#!/bin/bash
set -e

echo "============================================================"
echo " STARTING RANS LLM BENCHMARK SUITE"
echo "============================================================"

# Define your model pairs here: "Baseline_Path|Compressed_Path"
# Add larger models here (e.g., 32B or 70B) to prove the VRAM fitting limits!

MODELS=(
    "Qwen/Qwen3-32B|../comp_inference/Qwen_Qwen3-32B"

)

# --- SWEEP CONFIGURATION ---
# We use comma-separated strings with NO SPACES for argparse
BATCH_SIZES="1"
PROMPT_LENS="64"
GEN_LENS="4096"

# The critical "Memory Wall" sweep.
# Go from heavy constraints (0.4) to almost full memory (0.95)
VRAM_SWEEP="0.62"

# Hardware Config
GPUS="1,2"

MODES="rans_fused,baseline"


# 1. Run the experiments
for MODEL_PAIR in "${MODELS[@]}"; do
    # Split the string by the pipe character logic
    IFS='|' read -r BASELINE COMPRESSED <<< "$MODEL_PAIR"

    # Trim whitespace just in case
    BASELINE=$(echo "$BASELINE" | xargs)
    COMPRESSED=$(echo "$COMPRESSED" | xargs)

    echo ""
    echo "============================================================"
    echo " TESTING MODEL FAMILY: $BASELINE"
    echo "============================================================"

    # ---------------------------------------------------------
    # The Memory Wall Test
    # Sweeps across VRAM limits to show exactly where the baseline
    # spills over to CPU offload while rANS stays entirely on GPU.
    # ---------------------------------------------------------
    echo "Running VRAM Sweep..."
    python3 examples/offline_inference/rans_benchmark.py \
        --baseline_model "$BASELINE" \
        --compressed_model "$COMPRESSED" \
        --batch_sizes "$BATCH_SIZES" \
        --vram_utils "$VRAM_SWEEP" \
        --prompt_lens "$PROMPT_LENS" \
        --gen_lens "$GEN_LENS" \
        --eval_runs 1 \
        --gpus "$GPUS" \
        --modes "$MODES"
done

echo "============================================================"
echo "ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Run 'python3 plot_benchmark.py --plot_batch_sizes \"1,16\"' to generate your publication graphs."
echo "============================================================"
