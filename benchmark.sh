#!/bin/bash
set -e

echo "============================================================"
echo " STARTING RANS LLM BENCHMARK SUITE"
echo "============================================================"

# Define your model pairs here: "Baseline_Path|Compressed_Path"
# Add larger models here (e.g., 32B or 70B) to prove the VRAM fitting limits!
# MODELS=(
#     #"Qwen/Qwen3-14B|../comp_inference/Qwen_Qwen3-14B"
#     "Qwen/Qwen3-32B|../comp_inference/Qwen_Qwen3-32B"
#     # "meta-llama/Meta-Llama-3-8B|../comp_inference/Llama-3-8B"
# )
BASELINE_MODEL="Qwen/Qwen3-32B"
COALESCED_MODEL="../comp_inference/Qwen_Qwen3-32B_coalesced"
UNCOALESCED_MODEL="../comp_inference/Qwen_Qwen3-32B_uncoalesced"
#BASELINE_MODEL="Qwen/Qwen3-0.6B"
#COALESCED_MODEL="../comp_inference/Qwen_Qwen3-0.6B_coalesced"
#UNCOALESCED_MODEL="../comp_inference/Qwen_Qwen3-0.6B_uncoalesced"


# --- SWEEP CONFIGURATION ---
# We use comma-separated strings with NO SPACES for argparse
BATCH_SIZES="1"
PROMPT_LENS="1, 128, 512"
GEN_LENS="1, 128, 512"

# The critical "Memory Wall" sweep.
# Go from heavy constraints (0.4) to almost full memory (0.95)
#VRAM_SWEEP="0.6, 0.95"
VRAM_SWEEP="0.6, 0.95"

# Hardware Config
GPUS="0"

#MODES="rans_fused,baseline,rans_unfused"
MODES="rans_coal,rans_uncoal,baseline,triton_baseline"
#MODES="rans_fused,baseline"
KV_CACHE_GB="1.5"


echo "Running VRAM Sweep..."
python3 examples/offline_inference/rans_benchmark.py \
    --baseline_model "$BASELINE_MODEL" \
    --model_coal "$COALESCED_MODEL" \
    --model_uncoal "$UNCOALESCED_MODEL" \
    --vram_utils "$VRAM_SWEEP" \
    --batch_sizes "$BATCH_SIZES" \
    --prompt_lens "$PROMPT_LENS" \
    --gen_lens "$GEN_LENS" \
    --eval_runs 3 \
    --gpus "$GPUS" \
    --modes "$MODES"

# # 1. Run the experiments
# for MODEL_PAIR in "${MODELS[@]}"; do
#     # Split the string by the pipe character logic
#     IFS='|' read -r BASELINE COMPRESSED <<< "$MODEL_PAIR"

#     # Trim whitespace just in case
#     BASELINE=$(echo "$BASELINE" | xargs)
#     COMPRESSED=$(echo "$COMPRESSED" | xargs)

#     echo ""
#     echo "============================================================"
#     echo " TESTING MODEL FAMILY: $BASELINE"
#     echo "============================================================"

#     # ---------------------------------------------------------
#     # The Memory Wall Test
#     # Sweeps across VRAM limits to show exactly where the baseline
#     # spills over to CPU offload while rANS stays entirely on GPU.
#     # ---------------------------------------------------------
#     echo "Running VRAM Sweep..."
#     python3 examples/offline_inference/rans_benchmark.py \
#         --baseline_model "$BASELINE" \
#         --model_coal "$COALESCED_MODEL" \
#         --model_uncoal "$UNCOALESCED_MODEL" \
#         --vram_utils "$VRAM_SWEEP" \
#         --batch_sizes "$BATCH_SIZES" \
#         --prompt_lens "$PROMPT_LENS" \
#         --gen_lens "$GEN_LENS" \
#         --eval_runs 3 \
#         --gpus "$GPUS" \
#         --modes "$MODES" \
#         #--kv_cache "$KV_CACHE_GB"

# done

echo "============================================================"
echo "ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Run 'python3 plot_benchmark.py --plot_batch_sizes \"1,16\"' to generate your publication graphs."
echo "============================================================"
