# Zero-Shot Monocular Depth Estimation for Visual Perception in Lower-Limb Prostheses 

**Zero-Shot Monocular Depth Estimation for Visual Perception in Lower-Limb Prostheses**
<br>
Ishi Vijayvergiya, Marc Hölle, Youssef Dawoud, Amir El-Ghoussani, Michele De Vita, Rohan Asthana, Philipp Beckerle and Vasileios Belagiannis
<br>
*Accepted at [ISMT 2026](https://www.ismt.fau.de/).*

The LP dataset is a benchmark for monocular depth estimation from the viewpoint
of a prosthesis-mounted camera. It contains 3,144 synchronised RGB–depth pairs
recorded during stair ascent, stair descent and corridor walking, with the
camera attached above or below the left or right knee. This repository provides
the dataset layout, the split definitions, the evaluation protocol and the code
used to benchmark four depth foundation models without any fine-tuning.

## Repository layout

| Path | Content |
|---|---|
| `dataset/` | the six recording sessions and the split definitions in `dataset/splits/` |
| `config/` | one evaluation config per split (`lp_all`, `lp_staircase`, `lp_corridor`) |
| `scripts/` | inference and evaluation code for each benchmarked model, plus result comparison |
| `model_customizations/` | files applied on top of a model checkout: the LP dataset registration for Marigold, the raw-depth export for Depth Anything V2 |
| `LICENSE` | Creative Commons Attribution-NonCommercial 4.0 International license text |

## Dataset

### Recording setup

The data were captured with an Orbbec Femto Bolt, which combines a 1-megapixel
indirect Time-of-Flight depth sensor (Microsoft Azure Kinect technology, wide
field-of-view mode, 120° × 120°, 1024 × 1024 depth maps) with a co-registered 4K
RGB sensor synchronised at hardware level. The sensor was strapped to the lower
limb above or below the knee and connected to a laptop over USB-C 3.2.

### Sessions

| Sequence | Mount | Direction | Images |
|---|---|---|---|
| `lms_subject1_LA_downstairs_Nopeople_1` | left, above knee | down | 428 |
| `lms_subject1_LA_upstairs_Nopeople_1` | left, above knee | up | 629 |
| `lms_subject1_LB_upstairs_Nopeople_2` | left, below knee | up | 591 |
| `lms_subject1_RA_downstairs_Nopeople_1` | right, above knee | down | 375 |
| `lms_subject1_RA_upstairs_Nopeople_1` | right, above knee | up | 622 |
| `lms_subject1_RB_downstairs_Nopeople_1` | right, below knee | down | 499 |
| **Total** | | | **3,144** |

### Folder and file format

```
dataset/
└── lms_subject1_LA_downstairs_Nopeople_1/
    ├── sharpen_rgb/PNG/image100.png      
    ├── raw_depth/PNG/image100.png        
    ├── colored_depth/PNG/                
    ├── smooth_depth/PNG/                
    └── transformed_depth/PNG/        
```

Frames share the same names across representations within a session, so an RGB
frame and its ground truth can always be paired by filename. Only `sharpen_rgb`
and `raw_depth` are needed by the benchmark; the other folders are provided for
completeness.

Not every session ships all representations: `lms_subject1_RB_downstairs_Nopeople_1`
contains only `sharpen_rgb`, `raw_depth` and `colored_depth`, without
`smooth_depth` and `transformed_depth`. In that session each of the three folders
also holds a single `image.gif` preview next to its `PNG/` directory. None of
this affects the evaluation, since the scripts operate on the `PNG/` folders.

### Splits

| Split | Frames | Definition |
|---|---|---|
| `all` | 3,144 | every frame of the six sessions |
| `staircase` | 2,862 | frames recorded while ascending or descending stairs |
| `corridor` | 282 | frames recorded while walking a corridor |

Each line of a split file holds an RGB frame and its depth map, relative to
`dataset/`:

```
lms_subject1_LA_downstairs_Nopeople_1/sharpen_rgb/PNG/image100.png lms_subject1_LA_downstairs_Nopeople_1/raw_depth/PNG/image100.png
```

### Availability

To optain the dataset, please send an email to *mlisp.lms@fau.de* with subject matter
'[ISMT2026] LP Dataset - Access Request' stating the following:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Your name, title and affiliation*
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Your intended use of the data*
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The following license statement:
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *With this e-mail I agree that I will use the LP dataset following the Creative 
Common Attribution-NonCommercial 4.0 International License.*

We will promptly reply with a download link. Extract it into
`dataset/`:

```sh
unzip lp_dataset.zip -d dataset
```

You can verify the number of frames per session with

```sh
for d in dataset/lms_subject1_*; do echo "$d: $(ls "$d/sharpen_rgb/PNG" | wc -l)"; done
```

### License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />The LP dataset and this repository are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>. The full license text is in [LICENSE](LICENSE).

## Running the benchmark

### Environment

The evaluation scripts need `numpy`, `pillow` and `pyyaml`:

```sh
conda create -n lp_dataset python=3.10
conda activate lp_dataset
pip install numpy pillow pyyaml
```

Each foundation model additionally needs its upstream environment (PyTorch, CUDA
and model-specific packages such as `diffusers`); the commands are listed in
[`scripts/README.md`](scripts/README.md).

### Zero-shot inference

Clone the upstream repository and run inference per session, for example
Marigold

```sh
# register the LP dataset in Marigold (`mlp_depth`), required by config/
cp -r model_customizations/marigold/src/. Marigold/src/
python Marigold/script/depth/run.py \
    --checkpoint <checkpoint_dir>/marigold-depth-v1-1 \
    --input_rgb_dir dataset/<sequence>/sharpen_rgb/PNG \
    --output_dir dataset/<sequence>/marigold_zero_shot \
    --fp16
```

and Depth Anything V2, after applying the patched `run.py` that adds the
`--save-numpy` export needed for the evaluation

```sh
cp -r model_customizations/depth_anything_v2/. Depth-Anything-V2/
python Depth-Anything-V2/run.py --encoder vitl --save-numpy --pred-only \
    --img-path dataset/<sequence>/sharpen_rgb/PNG \
    --outdir dataset/<sequence>/depth_anything_v2_large
```

Depth Pro and VGGT have their own inference scripts under `scripts/ml_depth_pro`
and `scripts/vggt`; the README in `scripts/` covers all four models, their
checkpoints and the expected prediction folders.

### Evaluating a model

Pick the split through the config, point `--prediction_dir` at the predictions,
and choose the alignment (`least_square` for all benchmarked models, `none` to
score metric depth as it is):

```sh
python Marigold/script/depth/eval.py \
    --base_data_dir "$PWD" \
    --dataset_config config/lp_staircase.yaml \
    --alignment least_square \
    --prediction_dir dataset/marigold_predictions \
    --output_dir results/marigold_staircase
```

Each run writes per-sample metrics, aggregated metrics and a text report into
the output directory. To put several models side by side, use the scripts in
`scripts/comparison`.

## Citation

If you use the LP dataset or this code, please cite:

```bibtex
@InProceedings{vijayvergiya2026lpdataset,
  author={Ishi Vijayvergiya and Marc H{\"o}lle and Youssef Dawoud and Amir El-Ghoussani and Michele De Vita and Rohan Asthana and Philipp Beckerle and Vasileios Belagiannis},
  booktitle={},
  title={Zero-Shot Monocular Depth Estimation for Visual Perception in Lower-Limb Prostheses},
  year={2026},
  pages={},}
```

Please also cite the benchmarked models: Marigold, Depth Anything V2, Depth Pro
and VGGT.
