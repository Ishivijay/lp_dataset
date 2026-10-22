#!/bin/bash
# vggt_run_zero_shot.sh
# Complete pipeline for VGGT zero-shot learning on prosthesis data


#################################################################################
# CONFIGURATION
#################################################################################

# Model checkpoint directory 
CHECKPOINT_DIR="checkpoint"
# mkdir -p "$CHECKPOINT_DIR"

MODEL_PATH="$CHECKPOINT_DIR/vggt_1b.pt"

# #################################################################################
# # Download Model
# #################################################################################

# echo "================================================"
# echo "Downloading VGGT-1B Model Weights"
# echo "================================================"
# echo ""
# echo "Checkpoint directory: $CHECKPOINT_DIR"
# echo "Model will be saved to: $MODEL_PATH"
# echo ""

# # Check if already downloaded
# if [ -f "$MODEL_PATH" ]; then
#     echo "Model already exists at $MODEL_PATH"
#     echo "Size: $(du -h $MODEL_PATH | cut -f1)"
#     echo ""
#     read -p "Download again? (y/n) " -n 1 -r
#     echo
#     if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#         echo "Using existing model."
#         exit 0
#     fi
#     echo "Re-downloading..."
# fi

# # Download with wget
# echo "Downloading with wget..."
# wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -O "$MODEL_PATH"

# # Check if download was successful
# if [ $? -eq 0 ] && [ -f "$MODEL_PATH" ]; then
#     echo ""
#     echo "================================================"
#     echo "Download Successful!"
#     echo "================================================"
#     echo "Model saved to: $MODEL_PATH"
#     echo "Size: $(du -h $MODEL_PATH | cut -f1)"
#     echo ""
#     echo "Model ready for use!"
# else
#     echo ""
#     echo "================================================"
#     echo "ERROR: Download failed"
#     echo "================================================"
#     echo ""
#     echo "Manual download instructions:"
#     echo "1. Go to: https://huggingface.co/facebook/VGGT-1B/tree/main"
#     echo "2. Click on 'model.pt' file"
#     echo "3. Click the download button"
#     echo "4. Save to: $MODEL_PATH"
#     echo ""
#     exit 1
# fi


LOG_DIR="${HOME}/log"
mkdir -p "$LOG_DIR"

TS="$(date +'%Y%m%d-%H%M%S')"
LOG_FILE="$LOG_DIR/vggt_zero_shot_${TS}.log"

BASE_DIR="${WORK}/mlp_dataset/prospthesisproject-Data/Code/Data"

#################################################################################
# SETUP ENVIRONMENT
#################################################################################

echo "================================================" | tee -a "$LOG_FILE"
echo "Running VGGT Inference (Zero-Shot)" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python "$(dirname "$0")/vggt_inference.py" \
    --base_dir "$BASE_DIR" \
    --model_path "$MODEL_PATH" \
    --num_workers 1 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Inference completed at: $(date)" | tee -a "$LOG_FILE"

#################################################################################
# RUN EVALUATION
#################################################################################

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "Running VGGT Evaluation" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Evaluation without alignment
echo "1. Evaluation WITHOUT alignment:" | tee -a "$LOG_FILE"
python "$(dirname "$0")/vggt_eval.py" \
    --base_dir "$BASE_DIR" \
    --min_depth 0.1 \
    --max_depth 10.0 \
    --output_dir "$BASE_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"

# Evaluation with alignment
echo "2. Evaluation WITH alignment:" | tee -a "$LOG_FILE"
python "$(dirname "$0")/vggt_eval.py" \
    --base_dir "$BASE_DIR" \
    --use_alignment \
    --min_depth 0.1 \
    --max_depth 10.0 \
    --output_dir "$BASE_DIR" 2>&1 | tee -a "$LOG_FILE"

#################################################################################
# COMPLETION
#################################################################################

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "VGGT Zero-Shot Pipeline Completed!" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"