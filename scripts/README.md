# Stage 1 - Zero-Shot Depth Evaluation

Benchmarks four foundation models (Marigold, Depth Anything V2, Depth Pro,
VGGT) on the internal prosthesis dataset without fine-tuning.

Every model follows the same protocol: zero-shot inference on the raw RGB
frames → least-squares affine alignment to ground truth (`GT = s·pred + t`)
→ AbsRel / SqRel / RMSE / RMSE log / δ1 / δ2 / δ3.

## Common Environment Setup

* Run all commands from the repository root.
* Dataset at `dataset` with the
  six sequence folders `lms_kamal_*` - the default `--base_dir` of all
  scripts (`BASE_DIR` below).
* One venv per model. 

## Marigold

```bash
git clone https://github.com/prs-eth/Marigold.git
# apply the dataset registration for the LP dataset (`mlp_depth`), required by
# the evaluation configs in config/
cp -r model_customizations/marigold/src/. Marigold/src/
# download the checkpoint
Marigold/script/download_weights.sh marigold-depth-v1-1

# 1. inference - once per sequence folder
python Marigold/script/depth/run.py \
    --checkpoint <checkpoint_dir>/marigold-depth-v1-1 \
    --input_rgb_dir "$BASE_DIR/<sequence>/sharpen_rgb/PNG" \
    --output_dir   "$BASE_DIR/<sequence>/marigold_zero_shot" \
    --fp16

# 2. organise per-sequence outputs into one prediction tree
bash scripts/marigold/mlp_prosthesis_structured_predictions.sh

# 3. resize predictions to ground-truth resolution
python scripts/marigold/mlp_prosthesis_resize_predictions.py

# 4. evaluate on the full test set
python Marigold/script/depth/eval.py \
    --base_data_dir "$PWD" \
    --dataset_config config/lp_all.yaml \
    --alignment least_square \
    --prediction_dir "$BASE_DIR/marigold_predictions" \
    --output_dir "$BASE_DIR/eval_results_zero_shot"
```

## Depth Anything V2

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
# apply the author's overlay (adds --save-numpy raw-depth export, required for evaluation)
cp -r model_customizations/depth_anything_v2/. Depth-Anything-V2/
# checkpoint (vitl used in the thesis)
wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" \
     -O checkpoint/depth_anything_v2_vitl.pth

# 1. inference - loop over the six sequence folders
for SEQ in lms_kamal_LA_downstairs_Nopeople_1 lms_kamal_LA_upstairs_Nopeople_1 \
           lms_kamal_LB_upstairs_Nopeople_2  lms_kamal_RA_downstairs_Nopeople_1 \
           lms_kamal_RA_upstairs_Nopeople_1  lms_kamal_RB_downstairs_Nopeople_1; do
  python Depth-Anything-V2/run.py --encoder vitl \
      --img-path "$BASE_DIR/$SEQ/sharpen_rgb/PNG" \
      --outdir   "$BASE_DIR/$SEQ/depth_anything_v2_large" \
      --save-numpy --pred-only
done

# 2. evaluate (alignment on by default)
python scripts/depth_anything_v2/depth_anything_v2_eval.py \
    --base_dir "$BASE_DIR" --encoder large \
    --num_workers 16 --min_depth 0.1 --max_depth 10.0 --output_dir "$BASE_DIR"
```

## Depth Pro

```bash
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro && pip install -e .
# download checkpoint (upstream helper), then point depth_pro.py at it:
# set checkpoint_uri in src/depth_pro/depth_pro.py (DEFAULT_MONODEPTH_CONFIG_DICT)
# to your location (author: checkpoint/ml_depth_pro.pt), or keep the
# upstream default and place it at ./checkpoints/depth_pro.pt
source get_pretrained_models.sh

# 1. inference 
python scripts/ml_depth_pro/ml_depth_pro_inference.py

# 2. evaluate
python scripts/ml_depth_pro/ml_depth_pro_eval.py \
    --base_dir "$BASE_DIR" --num_workers 16 \
    --min_depth 0.1 --max_depth 10.0 --use_alignment --output_dir "$BASE_DIR"

# optional: regenerate depth visualisations
python scripts/ml_depth_pro/regenerate_mldepthpro_vis.py
```

## VGGT

```bash
git clone https://github.com/facebookresearch/vggt.git
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r vggt/requirements.txt
# checkpoint: vggt_1b.pt in checkpoint

# inference + evaluation (with and without alignment) in one job:
sbatch scripts/vggt/vggt_run_zero_shot.sh

# or step by step:
python scripts/vggt/vggt_inference.py --model_path checkpoint/vggt_1b.pt
python scripts/vggt/vggt_eval.py --use_alignment --min_depth 0.1 --max_depth 10.0
python scripts/vggt/vggt_eval_results_to_txt.py   # convert result CSV to text table
python scripts/vggt/regenerate_vggt_vis.py        # regenerate depth visualisations
```

---
