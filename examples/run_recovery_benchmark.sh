#!/bin/bash
# PipeLayer Recovery Benchmark: multiple runs for PipeLayer and standard PyTorch resume
# Usage: bash run_recovery_benchmark.sh [num_runs]
#
# Prerequisites:
#   - A valid MultiStream checkpoint must exist (checkpoint .chk + metadata .json)
#   - A valid standard PyTorch checkpoint must exist (accelerator.save_state format)
#   - Adjust paths below to match your environment

NUM_RUNS=${1:-5}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOME_DIR="$HOME"
MODEL="facebook/opt-1.3b"
DATASET="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
BATCH_SIZE=1
EPOCHS=1

# ============ Paths - ADJUST THESE ============
# PipeLayer with MultiStream loader
PIPELAYER_OUTPUT_DIR="${HOME_DIR}/download/ckpt"
PIPELAYER_RESUME_DIR="${PIPELAYER_OUTPUT_DIR}/step_1"
MULTISTREAM_LIB="${HOME_DIR}/code/pccheck/checkpoint_eval/pccheck/libtest_ssd.so"
MULTISTREAM_METADATA="${HOME_DIR}/code/pccheck/checkpoint_eval/pccheck/metadata_latest.json"
MULTISTREAM_CHECKPOINT="${HOME_DIR}/code/pccheck/checkpoint_eval/pccheck/checkpoint-0-0.chk"

# Standard PyTorch resume
STANDARD_OUTPUT_DIR="${HOME_DIR}/download/ckpt_standard_resume"
STANDARD_RESUME_DIR="${STANDARD_OUTPUT_DIR}/step_1"

# Result directories
RESULT_DIR="${SCRIPT_DIR}/recovery_benchmark_results"
mkdir -p "$RESULT_DIR"

PIPELAYER_CSV="${RESULT_DIR}/pipelayer_results.csv"
STANDARD_CSV="${RESULT_DIR}/standard_results.csv"

echo "========================================="
echo "PipeLayer Recovery Benchmark"
echo "Runs per method: ${NUM_RUNS}"
echo "========================================="

# Detect python in pccheck conda env
PYTHON_CMD="python"
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -d "${HOME_DIR}/anaconda3/envs/pccheck" ]; then
        PYTHON_CMD="${HOME_DIR}/anaconda3/envs/pccheck/bin/python"
    fi
fi
echo "Using Python: ${PYTHON_CMD}"

# ---- PipeLayer MultiStream Recovery ----
echo ""
echo "===== PipeLayer (MultiStream) Recovery ====="

# Remove old CSV to start fresh
rm -f "${PIPELAYER_OUTPUT_DIR}/training_time_log.csv"

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "--- PipeLayer Run $i / $NUM_RUNS ---"
    $PYTHON_CMD run_clm_pipelayer2.py \
        --model_name_or_path "$MODEL" \
        --dataset_name "$DATASET" \
        --dataset_config_name "$DATASET_CONFIG" \
        --per_device_train_batch_size $BATCH_SIZE \
        --num_train_epochs $EPOCHS \
        --output_dir "$PIPELAYER_OUTPUT_DIR" \
        --use_pipelayer \
        --pipelayer_loader multistream \
        --multistream_lib_path "$MULTISTREAM_LIB" \
        --multistream_metadata_file "$MULTISTREAM_METADATA" \
        --multistream_checkpoint_file "$MULTISTREAM_CHECKPOINT" \
        --resume_from_checkpoint "$PIPELAYER_RESUME_DIR" \
        --do_train 2>&1 | tee "${RESULT_DIR}/pipelayer_run_${i}.log"
    
    echo "[OK] PipeLayer run $i complete"
done

# Copy results
if [ -f "${PIPELAYER_OUTPUT_DIR}/training_time_log.csv" ]; then
    cp "${PIPELAYER_OUTPUT_DIR}/training_time_log.csv" "$PIPELAYER_CSV"
    echo "PipeLayer results saved to: $PIPELAYER_CSV"
fi

# ---- Standard PyTorch Recovery ----
echo ""
echo "===== Standard PyTorch Recovery ====="

rm -f "${STANDARD_OUTPUT_DIR}/training_time_log.csv"

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "--- Standard Run $i / $NUM_RUNS ---"
    $PYTHON_CMD run_clm_pipelayer2.py \
        --model_name_or_path "$MODEL" \
        --dataset_name "$DATASET" \
        --dataset_config_name "$DATASET_CONFIG" \
        --per_device_train_batch_size $BATCH_SIZE \
        --num_train_epochs $EPOCHS \
        --output_dir "$STANDARD_OUTPUT_DIR" \
        --resume_from_checkpoint "$STANDARD_RESUME_DIR" \
        --do_train 2>&1 | tee "${RESULT_DIR}/standard_run_${i}.log"
    
    echo "[OK] Standard run $i complete"
done

if [ -f "${STANDARD_OUTPUT_DIR}/training_time_log.csv" ]; then
    cp "${STANDARD_OUTPUT_DIR}/training_time_log.csv" "$STANDARD_CSV"
    echo "Standard results saved to: $STANDARD_CSV"
fi

echo ""
echo "========================================="
echo "Benchmark complete. Generating plots..."
echo "========================================="

$PYTHON_CMD plot_recovery_results.py "$RESULT_DIR"
