#!/usr/bin/env python3
"""
vggt_eval.py
Evaluate VGGT zero-shot predictions on prosthesis dataset
"""

import numpy as np
import os
from pathlib import Path
from PIL import Image
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from tqdm import tqdm


def load_depth_image(path):
    """Load depth image from .npy or .png file"""
    if path.endswith('.npy'):
        depth = np.load(path)
    elif path.endswith('.png'):
        img = Image.open(path)
        depth = np.array(img, dtype=np.float32)
        
        # Handle 16-bit depth images
        if img.mode == 'I;16' or np.max(depth) > 256:
            depth = depth / 1000.0  # Convert mm to meters
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    if len(depth.shape) == 3:
        depth = depth.squeeze()
    
    return depth


def align_depth_least_square(gt_arr, pred_arr, valid_mask_arr):
    """
    Align prediction to ground truth using least squares (scale and shift)
    
    Args:
        gt_arr: Ground truth depth array
        pred_arr: Predicted depth array
        valid_mask_arr: Valid pixel mask
        
    Returns:
        Aligned prediction array
    """
    valid_mask_arr = valid_mask_arr.astype(bool)
    
    # Extract valid pixels
    gt_valid = gt_arr[valid_mask_arr]
    pred_valid = pred_arr[valid_mask_arr]
    
    if len(gt_valid) < 10:
        return pred_arr
    
    # Solve: gt = scale * pred + shift
    A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=-1)
    
    try:
        X = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
        scale, shift = X
    except np.linalg.LinAlgError:
        return pred_arr
    
    # Apply alignment
    aligned = scale * pred_arr + shift
    
    return aligned


def compute_metrics(gt, pred, valid_mask, min_depth=0.1, max_depth=10.0):
    """
    Compute depth estimation metrics
    
    Args:
        gt: Ground truth depth
        pred: Predicted depth
        valid_mask: Valid pixel mask
        min_depth: Minimum depth threshold
        max_depth: Maximum depth threshold
        
    Returns:
        Dictionary of metrics
    """
    # Apply valid mask and depth range
    valid_mask = valid_mask & (gt > min_depth) & (gt < max_depth)
    valid_mask = valid_mask & (pred > 0) & np.isfinite(pred)
    
    if valid_mask.sum() < 10:
        return None
    
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    
    # Compute metrics
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    
    # Delta metrics
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'd1': d1,
        'd2': d2,
        'd3': d3
    }


def evaluate_single_image(pred_path, gt_path, use_alignment=False, 
                         min_depth=0.1, max_depth=10.0):
    """Evaluate a single image pair"""
    try:
        # Load predictions and ground truth
        pred = load_depth_image(str(pred_path))
        gt = load_depth_image(str(gt_path))
        
        # Ensure same shape
        if pred.shape != gt.shape:
            from PIL import Image as PILImage
            pred_img = PILImage.fromarray(pred)
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), PILImage.BILINEAR)
            pred = np.array(pred_img)
        
        # Create valid mask
        valid_mask = (gt > 0) & np.isfinite(gt) & (pred > 0) & np.isfinite(pred)
        
        # Align if requested
        if use_alignment:
            pred = align_depth_least_square(gt, pred, valid_mask)
        
        # Compute metrics
        metrics = compute_metrics(gt, pred, valid_mask, min_depth, max_depth)
        
        if metrics is None:
            return None
        
        metrics['filename'] = pred_path.name
        return metrics
        
    except Exception as e:
        print(f"Error evaluating {pred_path.name}: {e}")
        return None


def evaluate_sequence(sequence_path, use_alignment=False, 
                     min_depth=0.1, max_depth=10.0):
    """Evaluate all predictions in a sequence"""
    pred_dir = sequence_path / "vggt_predictions"
    gt_dir = sequence_path / "raw_depth" / "PNG"
    
    if not pred_dir.exists():
        print(f"Prediction directory not found: {pred_dir}")
        return None
    
    if not gt_dir.exists():
        print(f"Ground truth directory not found: {gt_dir}")
        return None
    
    # Get prediction files
    pred_files = sorted(pred_dir.glob("*_vggt_depth.npy"))
    
    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return None
    
    print(f"\nEvaluating {len(pred_files)} predictions from {sequence_path.name}")
    
    # Evaluate each image
    all_metrics = []
    
    for pred_path in tqdm(pred_files, desc=sequence_path.name):
        # Find corresponding ground truth
        # Assuming naming convention: frame_XXXX_vggt_depth.npy -> frame_XXXX.png
        base_name = pred_path.name.replace("_vggt_depth.npy", ".png")
        gt_path = gt_dir / base_name
        
        if not gt_path.exists():
            print(f"Ground truth not found for {pred_path.name}")
            continue
        
        metrics = evaluate_single_image(pred_path, gt_path, use_alignment, 
                                       min_depth, max_depth)
        
        if metrics is not None:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print(f"No valid metrics for {sequence_path.name}")
        return None
    
    # Aggregate metrics
    aggregated = {
        'sequence': sequence_path.name,
        'num_images': len(all_metrics)
    }
    
    metric_keys = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3']
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        aggregated[key] = np.mean(values)
    
    return aggregated, all_metrics


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate VGGT zero-shot predictions on prosthesis dataset'
    )
    parser.add_argument('--base_dir', type=str,
                       default='dataset',
                       help='Base directory containing sequences')
    parser.add_argument('--use_alignment', action='store_true',
                       help='Use least-squares alignment')
    parser.add_argument('--min_depth', type=float, default=0.1,
                       help='Minimum depth threshold')
    parser.add_argument('--max_depth', type=float, default=10.0,
                       help='Maximum depth threshold')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    
    # Sequences to evaluate
    sequences = [
        "lms_subject1_LA_downstairs_Nopeople_1",
        "lms_subject1_LA_upstairs_Nopeople_1",
        "lms_subject1_LB_upstairs_Nopeople_2",
        "lms_subject1_RA_downstairs_Nopeople_1",
        "lms_subject1_RA_upstairs_Nopeople_1",
        "lms_subject1_RB_downstairs_Nopeople_1"
    ]
    
    print("=" * 80)
    print("VGGT Zero-Shot Evaluation on Prosthesis Dataset")
    print(f"Alignment: {'ON' if args.use_alignment else 'OFF'}")
    print(f"Depth range: [{args.min_depth}, {args.max_depth}]")
    print("=" * 80)
    
    # Evaluate all sequences
    all_results = []
    detailed_results = []
    
    for seq_name in sequences:
        seq_path = base_dir / seq_name
        
        if not seq_path.exists():
            print(f"Warning: Sequence not found: {seq_path}")
            continue
        
        result = evaluate_sequence(seq_path, args.use_alignment,
                                  args.min_depth, args.max_depth)
        
        if result is not None:
            aggregated, detailed = result
            all_results.append(aggregated)
            detailed_results.extend(detailed)
    
    if not all_results:
        print("No evaluation results generated!")
        return
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for result in all_results:
        print(f"\n{result['sequence']} ({result['num_images']} images)")
        print(f"  AbsRel: {result['abs_rel']:.4f}")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  δ1: {result['d1']:.4f}")
        print(f"  δ2: {result['d2']:.4f}")
        print(f"  δ3: {result['d3']:.4f}")
    
    # Overall average
    print("\n" + "=" * 80)
    print("OVERALL AVERAGE")
    print("=" * 80)
    
    total_images = sum(r['num_images'] for r in all_results)
    metric_keys = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3']
    for key in metric_keys:
        weighted_sum = sum(r[key] * r['num_images'] for r in all_results)
        avg = weighted_sum / total_images if total_images > 0 else 0.0
        print(f"{key}: {avg:.4f}")
    
    # Save results
    alignment_suffix = "_aligned" if args.use_alignment else ""
    
    # Save aggregated results
    csv_path = output_dir / f"vggt_evaluation_results{alignment_suffix}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence', 'num_images'] + metric_keys)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nResults saved to {csv_path}")
    
    # Save detailed results
    detailed_csv = output_dir / f"vggt_evaluation_detailed{alignment_suffix}.csv"
    with open(detailed_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename'] + metric_keys)
        writer.writeheader()
        writer.writerows(detailed_results)
    
    print(f"Detailed results saved to {detailed_csv}")
    
    # Save summary text file (same format as ML-Depth-Pro)
    txt_path = output_dir / f"vggt_eval_results{alignment_suffix}.txt"
    with open(txt_path, 'w') as f:
        alignment_text = "with alignment (least-squares)" if args.use_alignment else "without alignment (metric depth)"
        f.write(f"VGGT Evaluation Results\n")
        f.write(f"Evaluation mode: {alignment_text}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Overall Results ({total_images} images):\n")
        for key in metric_keys:
            weighted_sum = sum(r[key] * r['num_images'] for r in all_results)
            avg = weighted_sum / total_images if total_images > 0 else 0.0
            f.write(f"  {key}: {avg:.4f}\n")
        f.write("\n")
        
        f.write("Per-sequence results:\n")
        for r in all_results:
            f.write(f"\n{r['sequence']} ({r['num_images']} images):\n")
            for key in metric_keys:
                f.write(f"  {key}: {r[key]:.4f}\n")
    
    print(f"Summary text saved to {txt_path}")


if __name__ == '__main__':
    main()