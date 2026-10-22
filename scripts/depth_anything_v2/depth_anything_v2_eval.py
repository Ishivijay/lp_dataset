import numpy as np
import os
from pathlib import Path
from PIL import Image
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def load_depth_image(path):
    """Load depth image from .npy or .png file"""
    if path.endswith('.npy'):
        # Load raw depth from .npy file
        depth = np.load(path)
    elif path.endswith('.png'):
        img = Image.open(path)
        depth = np.array(img, dtype=np.float32)
        
        # If it's a 16-bit image, convert to meters
        if img.mode == 'I;16' or np.max(depth) > 256:
            depth = depth / 1000.0  # Convert mm to meters
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    # Ensure 2D array
    if len(depth.shape) == 3:
        depth = depth.squeeze()
    
    return depth


def align_depth_least_square(gt_arr, pred_arr, valid_mask_arr):
    """
    Align prediction to ground truth using least squares (scale and shift).
    This matches Marigold's alignment approach.
    
    Args:
        gt_arr: Ground truth depth array
        pred_arr: Predicted depth array  
        valid_mask_arr: Valid pixel mask
        
    Returns:
        Aligned prediction array
    """
    # Only use valid pixels for alignment
    valid_gt = gt_arr[valid_mask_arr]
    valid_pred = pred_arr[valid_mask_arr]
    
    # Solve: gt = scale * pred + shift
    # Using least squares: [pred, 1] @ [scale, shift]^T = gt
    A = np.stack([valid_pred, np.ones_like(valid_pred)], axis=-1)
    b = valid_gt
    
    # Solve least squares
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    scale, shift = x[0], x[1]
    
    # Apply alignment
    aligned_pred = scale * pred_arr + shift
    
    return aligned_pred


def compute_depth_metrics(pred, gt, valid_mask):
    """
    Compute depth evaluation metrics following Marigold protocol.
    
    Args:
        pred: Predicted depth map (after alignment)
        gt: Ground truth depth map
        valid_mask: Boolean mask of valid pixels
    
    Returns:
        Dictionary of metrics matching Marigold's output
    """
    valid_count = valid_mask.sum()
    
    if valid_count == 0:
        return None
    
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Threshold accuracies (δ < 1.25^i)
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    delta1_acc = (thresh < 1.25).mean()
    delta2_acc = (thresh < 1.25 ** 2).mean()
    delta3_acc = (thresh < 1.25 ** 3).mean()
    
    # Error metrics
    diff = pred_valid - gt_valid
    diff_log = np.log(pred_valid) - np.log(gt_valid)
    
    abs_rel = np.mean(np.abs(diff) / gt_valid)
    sq_rel = np.mean((diff ** 2) / gt_valid)
    
    rmse = np.sqrt(np.mean(diff ** 2))
    rmse_log = np.sqrt(np.mean(diff_log ** 2))
    
    log10 = np.mean(np.abs(np.log10(pred_valid) - np.log10(gt_valid)))
    
    # Inverse RMSE
    i_rmse = np.sqrt(np.mean((1.0 / gt_valid - 1.0 / pred_valid) ** 2))
    
    # Scale-invariant log RMSE (matching Marigold's formula)
    silog = np.sqrt(np.mean(diff_log ** 2) - (np.mean(diff_log) ** 2)) * 100
    
    return {
        'abs_relative_difference': abs_rel,
        'squared_relative_difference': sq_rel,
        'rmse_linear': rmse,
        'rmse_log': rmse_log,
        'log10': log10,
        'delta1_acc': delta1_acc,
        'delta2_acc': delta2_acc,
        'delta3_acc': delta3_acc,
        'i_rmse': i_rmse,
        'silog_rmse': silog
    }


def evaluate_single_image(args):
    """Evaluate a single image pair - designed for parallel processing"""
    pred_path, gt_path, min_depth, max_depth, use_alignment = args
    
    try:
        # Load data
        predicted_depth = load_depth_image(pred_path)
        ground_truth = load_depth_image(gt_path)
        
        # Resize if shapes don't match
        if predicted_depth.shape != ground_truth.shape:
            pred_img = Image.fromarray(predicted_depth)
            pred_img = pred_img.resize((ground_truth.shape[1], ground_truth.shape[0]), Image.BILINEAR)
            predicted_depth = np.array(pred_img)
        
        # Create valid mask (matching Marigold's approach)
        valid_mask = (ground_truth > min_depth) & (ground_truth < max_depth) & \
                     np.isfinite(ground_truth) & np.isfinite(predicted_depth) & \
                     (predicted_depth > 0)
        
        if valid_mask.sum() < 10:
            return os.path.basename(pred_path), None, "Insufficient valid pixels"
        
        # Apply least-squares alignment (like Marigold)
        if use_alignment:
            predicted_depth = align_depth_least_square(ground_truth, predicted_depth, valid_mask)
        
        # Clip prediction to valid range
        predicted_depth = np.clip(predicted_depth, min_depth, max_depth)
        predicted_depth = np.clip(predicted_depth, 1e-6, None)  # Avoid division by zero
        
        # Compute metrics
        metrics = compute_depth_metrics(predicted_depth, ground_truth, valid_mask)
        
        if metrics is None:
            return os.path.basename(pred_path), None, "Metric computation failed"
        
        return os.path.basename(pred_path), metrics, None
        
    except Exception as e:
        return os.path.basename(pred_path), None, str(e)


def evaluate_sequence(predictions_dir, ground_truth_dir, num_workers=16, 
                      min_depth=0.1, max_depth=10.0, use_alignment=True):
    """Evaluate using .npy prediction files (raw depth values)"""
    
    # Get all prediction .npy files
    pred_files = sorted([f for f in os.listdir(predictions_dir) if f.endswith('_depth.npy')])
    
    if len(pred_files) == 0:
        print(f"Warning: No .npy prediction files found in {predictions_dir}")
        return []
    
    args_list = []
    for pred_file in pred_files:
        # Extract base filename
        base_name = pred_file.replace('_depth.npy', '')
        gt_file = base_name + '.png'
        
        pred_path = os.path.join(predictions_dir, pred_file)
        gt_path = os.path.join(ground_truth_dir, gt_file)
        
        if os.path.exists(gt_path):
            args_list.append((pred_path, gt_path, min_depth, max_depth, use_alignment))
        else:
            print(f"  Warning: Ground truth not found for {pred_file}, expected {gt_file}")
    
    if len(args_list) == 0:
        print(f"Warning: No matching ground truth files found")
        return []
    
    # Process images in parallel
    results = []
    processed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_single_image, args): args for args in args_list}
        
        for future in as_completed(futures):
            filename, metrics, error = future.result()
            processed_count += 1
            
            if error is None and metrics is not None:
                results.append((filename, metrics))
            else:
                error_count += 1
            
            if processed_count % 50 == 0:
                print(f"  Progress: {processed_count}/{len(args_list)} images ({len(results)} successful, {error_count} errors)")
    
    print(f"  Final: {len(results)} successful, {error_count} errors out of {len(args_list)} total")
    
    return results


def evaluate_all_sequences(base_dir, encoder='vitl', num_workers=16, 
                          min_depth=0.1, max_depth=10.0, use_alignment=True):
    """
    Evaluate all sequences in the prosthesis dataset.
    
    Args:
        base_dir: Base directory containing all sequences
        encoder: Encoder size (vits, vitb, vitl, vitg)
        num_workers: Number of parallel workers per sequence
        min_depth: Minimum valid depth (in meters)
        max_depth: Maximum valid depth (in meters)
        use_alignment: Whether to use least-squares scale-shift alignment
    """
    
    sequences = [
        "lms_kamal_LA_downstairs_Nopeople_1",
        "lms_kamal_LA_upstairs_Nopeople_1", 
        "lms_kamal_LB_upstairs_Nopeople_2",
        "lms_kamal_RA_downstairs_Nopeople_1", 
        "lms_kamal_RA_upstairs_Nopeople_1",
        "lms_kamal_RB_downstairs_Nopeople_1"
    ]
    
    all_results = []
    sequence_results = {}
    
    # Map encoder to output directory name
    encoder_to_size = {
        'vits': 'small',
        'vitb': 'base',
        'vitl': 'large',
        'vitg': 'giant'
    }
    size_name = encoder_to_size.get(encoder, encoder)
    output_dirname = f"depth_anything_v2_{size_name}"
    
    alignment_text = " (with least-squares alignment)" if use_alignment else " (no alignment)"
    print(f"Looking for predictions in: {output_dirname}")
    print(f"Evaluation mode: {alignment_text}\n")
    
    for seq in sequences:
        print(f"\n{'='*60}")
        print(f"Evaluating sequence: {seq}")
        print(f"{'='*60}")
        
        predictions_dir = base_dir / seq / output_dirname
        ground_truth_dir = base_dir / seq / "raw_depth" / "PNG"
        
        if not predictions_dir.exists():
            print(f"Warning: Predictions directory not found: {predictions_dir}")
            continue
        
        if not ground_truth_dir.exists():
            print(f"Warning: Ground truth directory not found: {ground_truth_dir}")
            continue
        
        # Evaluate this sequence
        results = evaluate_sequence(str(predictions_dir), str(ground_truth_dir), 
                                   num_workers, min_depth, max_depth, use_alignment)
        
        if len(results) > 0:
            # Add sequence name to results
            for filename, metrics in results:
                all_results.append((seq, filename, metrics))
            
            # Compute sequence statistics
            metrics_list = [m for _, m in results]
            mean_metrics = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if not np.isnan(m[key])]
                if len(values) > 0:
                    mean_metrics[key] = np.mean(values)
                else:
                    mean_metrics[key] = np.nan
            
            sequence_results[seq] = {
                'mean_metrics': mean_metrics,
                'count': len(results)
            }
            
            print(f"\nSequence {seq} - Results ({len(results)} images):")
            print(f"  δ1: {mean_metrics.get('delta1_acc', np.nan):.4f}")
            print(f"  δ2: {mean_metrics.get('delta2_acc', np.nan):.4f}")
            print(f"  δ3: {mean_metrics.get('delta3_acc', np.nan):.4f}")
            print(f"  AbsRel: {mean_metrics.get('abs_relative_difference', np.nan):.4f}")
            print(f"  SqRel: {mean_metrics.get('squared_relative_difference', np.nan):.4f}")
            print(f"  RMSE: {mean_metrics.get('rmse_linear', np.nan):.4f}")
            print(f"  RMSE_log: {mean_metrics.get('rmse_log', np.nan):.4f}")
            print(f"  log10: {mean_metrics.get('log10', np.nan):.4f}")
            print(f"  SILog: {mean_metrics.get('silog_rmse', np.nan):.4f}")
            print(f"  iRMSE: {mean_metrics.get('i_rmse', np.nan):.4f}")
        else:
            print(f"\nNo valid results for sequence {seq}")
    
    return all_results, sequence_results


def save_results_to_csv(all_results, output_file):
    """Save evaluation results to CSV file"""
    
    if len(all_results) == 0:
        print("No results to save!")
        return
    
    # Define CSV columns (matching Marigold's metric names)
    fieldnames = [
        'filename',
        'abs_relative_difference',
        'squared_relative_difference',
        'rmse_linear',
        'rmse_log',
        'log10',
        'delta1_acc',
        'delta2_acc',
        'delta3_acc',
        'i_rmse',
        'silog_rmse'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for seq, filename, metrics in all_results:
            # Construct full path: sequence/sharpen_rgb/PNG/filename
            full_path = f"{seq}/sharpen_rgb/PNG/{filename}"
            row = {'filename': full_path}
            row.update(metrics)
            writer.writerow(row)
    
    print(f"\nDetailed results saved to: {output_file}")


def save_summary_results(all_results, sequence_results, output_file, encoder, use_alignment):
    """Save summary results to text file"""
    
    if len(all_results) == 0:
        print("No results to save!")
        return
    
    with open(output_file, 'w') as f:
        alignment_text = "with least-squares alignment" if use_alignment else "without alignment"
        f.write(f"Depth-Anything-V2 ({encoder}) Evaluation Results\n")
        f.write(f"Evaluation mode: {alignment_text}\n")
        f.write("="*60 + "\n\n")
        
        # Overall statistics
        all_metrics = [metrics for _, _, metrics in all_results]
        overall_means = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if len(values) > 0:
                overall_means[key] = np.mean(values)
            else:
                overall_means[key] = np.nan
        
        f.write(f"Overall Results ({len(all_results)} images):\n")
        f.write(f"  abs_relative_difference: {overall_means.get('abs_relative_difference', np.nan):.4f}\n")
        f.write(f"  squared_relative_difference: {overall_means.get('squared_relative_difference', np.nan):.4f}\n")
        f.write(f"  rmse_linear: {overall_means.get('rmse_linear', np.nan):.4f}\n")
        f.write(f"  rmse_log: {overall_means.get('rmse_log', np.nan):.4f}\n")
        f.write(f"  log10: {overall_means.get('log10', np.nan):.4f}\n")
        f.write(f"  delta1_acc: {overall_means.get('delta1_acc', np.nan):.4f}\n")
        f.write(f"  delta2_acc: {overall_means.get('delta2_acc', np.nan):.4f}\n")
        f.write(f"  delta3_acc: {overall_means.get('delta3_acc', np.nan):.4f}\n")
        f.write(f"  i_rmse: {overall_means.get('i_rmse', np.nan):.4f}\n")
        f.write(f"  silog_rmse: {overall_means.get('silog_rmse', np.nan):.4f}\n")
        f.write("\n")
        
        # Per-sequence results
        f.write("Per-sequence results:\n")
        for seq, seq_data in sequence_results.items():
            f.write(f"\n{seq} ({seq_data['count']} images):\n")
            mean_metrics = seq_data['mean_metrics']
            f.write(f"  abs_relative_difference: {mean_metrics.get('abs_relative_difference', np.nan):.4f}\n")
            f.write(f"  delta1_acc: {mean_metrics.get('delta1_acc', np.nan):.4f}\n")
            f.write(f"  rmse_linear: {mean_metrics.get('rmse_linear', np.nan):.4f}\n")
            f.write(f"  silog_rmse: {mean_metrics.get('silog_rmse', np.nan):.4f}\n")
    
    print(f"Summary results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Depth-Anything-V2 on prosthesis dataset (Marigold-aligned)')
    parser.add_argument('--base_dir', type=str, 
                        default='/path/to/work/mlp_dataset/prospthesisproject-Data/Code/Data',
                        help='Base directory containing all sequences')
    parser.add_argument('--encoder', type=str, default='large', 
                        choices=['vits', 'vitb', 'vitl', 'vitg', 'small', 'base', 'large', 'giant'],
                        help='Encoder size used for predictions')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum valid depth in meters')
    parser.add_argument('--max_depth', type=float, default=10.0,
                        help='Maximum valid depth in meters')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: base_dir)')
    parser.add_argument('--no_alignment', action='store_true',
                        help='Disable least-squares scale-shift alignment')
    
    args = parser.parse_args()
    
    # Map encoder names
    encoder_map = {
        'small': 'vits',
        'base': 'vitb',
        'large': 'vitl',
        'giant': 'vitg'
    }
    encoder = encoder_map.get(args.encoder, args.encoder)
    
    use_alignment = not args.no_alignment
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    
    num_workers = min(args.num_workers, mp.cpu_count())
    print(f"Using {num_workers} parallel workers")
    print(f"Encoder: {encoder}")
    print(f"Depth range: [{args.min_depth}, {args.max_depth}] meters")
    print(f"Alignment: {'enabled (least-squares)' if use_alignment else 'disabled'}\n")
    
    # Run evaluation
    print("Starting Depth-Anything-V2 evaluation (Marigold-aligned protocol)...")
    all_results, sequence_results = evaluate_all_sequences(
        base_dir, 
        encoder=encoder,
        num_workers=num_workers,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_alignment=use_alignment
    )
    
    if len(all_results) > 0:
        # Save detailed CSV
        alignment_suffix = "_aligned" if use_alignment else "_noalign"
        csv_output = output_dir / f"depth_anything_v2_{encoder}_detailed_results{alignment_suffix}.csv"
        save_results_to_csv(all_results, csv_output)
        
        # Save summary text file
        txt_output = output_dir / f"depth_anything_v2_{encoder}_eval_results{alignment_suffix}.txt"
        save_summary_results(all_results, sequence_results, txt_output, encoder, use_alignment)
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*60}")
    else:
        print("\nNo results to save.")