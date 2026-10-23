#!/usr/bin/env python3
"""Regenerate ML-Depth-Pro depth visualizations with Spectral colormap.

This script reads existing .npy depth predictions and regenerates the
visualization JPGs with the Spectral colormap to match Marigold and
Depth Anything V2 visualizations.

Evaluation results are NOT affected, only the .jpg visualization files
are overwritten. The .npy and .txt files are never modified.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import argparse


def regenerate_vis(npy_path):
    """Regenerate a single visualization with Spectral colormap (matching Marigold)."""
    try:
        depth = np.load(npy_path)

        # Normalize depth to [0, 255] (same as Depth Anything V2)
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Apply Spectral colormap (matching Marigold)
        cmap = matplotlib.colormaps.get_cmap("Spectral")
        color_depth = (cmap(depth_uint8)[..., :3] * 255).astype(np.uint8)

        # Save over the old turbo JPG
        # ML-Depth-Pro saves vis as: image100.jpg (stem + .jpg)
        stem = npy_path.stem.replace("_depth", "")
        vis_path = npy_path.parent / (stem + ".jpg")
        Image.fromarray(color_depth).save(vis_path, format="JPEG", quality=90)
        return f"OK: {npy_path.name}"
    except Exception as e:
        return f"FAIL: {npy_path.name}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Regenerate ML-Depth-Pro visualizations with Spectral colormap")
    parser.add_argument("--base_dir", type=str,
                        default="dataset",
                        help="Base directory containing sequence folders")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    sequences = [
        "lms_subject1_LA_downstairs_Nopeople_1",
        "lms_subject1_LA_upstairs_Nopeople_1",
        "lms_subject1_LB_upstairs_Nopeople_2",
        "lms_subject1_RA_downstairs_Nopeople_1",
        "lms_subject1_RA_upstairs_Nopeople_1",
        "lms_subject1_RB_downstairs_Nopeople_1",
    ]

    all_npy = []
    for seq in sequences:
        pred_dir = base_dir / seq / "ml_depth_pro_predictions"
        if pred_dir.exists():
            npy_files = sorted(pred_dir.glob("*_depth.npy"))
            all_npy.extend(npy_files)
            print(f"  {seq}: {len(npy_files)} .npy files")
        else:
            print(f"  {seq}: directory not found — skipping")

    print(f"\nTotal: {len(all_npy)} .npy files to regenerate visualizations for")
    print("Regenerating with Spectral colormap...\n")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(regenerate_vis, all_npy))

    ok_count = sum(1 for r in results if r.startswith("OK"))
    fail_count = sum(1 for r in results if r.startswith("FAIL"))

    for r in results:
        if r.startswith("FAIL"):
            print(r)

    print(f"\nDone! {ok_count} succeeded, {fail_count} failed.")


if __name__ == "__main__":
    main()
