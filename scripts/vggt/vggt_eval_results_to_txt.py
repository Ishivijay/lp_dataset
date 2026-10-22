#!/usr/bin/env python3
"""
vggt_eval_results_to_txt.py

Read VGGT evaluation CSV files and generate nicely formatted TXT summary files,
similar to ML-Depth-Pro's ml_depth_pro_eval_results_noalign.txt / _aligned.txt
"""

import numpy as np
import csv
from pathlib import Path
import argparse


def load_csv_results(csv_path: Path):
    """Load evaluation results from CSV file."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_name = row['sequence']
            num_images = int(row['num_images'])
            metrics = {}
            for key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3']:
                metrics[key] = float(row[key])
            results.append({'sequence': seq_name, 'num_images': num_images, 'metrics': metrics})
    return results


def save_summary_txt(results, output_file: Path, use_alignment: bool):
    """Save evaluation results to a formatted text file."""
    
    if len(results) == 0:
        print("No results to save!")
        return
    
    metric_keys = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3']
    metric_names = {
        'abs_rel': 'abs_rel',
        'sq_rel': 'sq_rel',
        'rmse': 'rmse',
        'rmse_log': 'rmse_log',
        'd1': 'd1',
        'd2': 'd2',
        'd3': 'd3',
    }
    
    alignment_text = "with alignment (least-squares)" if use_alignment else "without alignment (metric depth)"
    
    with open(output_file, 'w') as f:
        f.write("VGGT Evaluation Results\n")
        f.write(f"Evaluation mode: {alignment_text}\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall statistics (weighted by image count to match other models)
        total_images = sum(r['num_images'] for r in results)
        overall_means = {}
        for key in metric_keys:
            weighted_sum = sum(r['metrics'][key] * r['num_images'] for r in results)
            overall_means[key] = weighted_sum / total_images if total_images > 0 else 0.0
        f.write(f"Overall Results ({total_images} images):\n")
        for key in metric_keys:
            f.write(f"  {key}: {overall_means[key]:.4f}\n")
        f.write("\n")
        
        # Per-sequence results
        f.write("Per-sequence results:\n")
        for r in results:
            f.write(f"\n{r['sequence']} ({r['num_images']} images):\n")
            for key in metric_keys:
                f.write(f"  {key}: {r['metrics'][key]:.4f}\n")
    
    print(f"Summary results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate VGGT evaluation TXT summary from CSV results'
    )
    parser.add_argument('--base_dir', type=str,
                        default='dataset',
                        help='Directory containing VGGT evaluation CSV files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for TXT files (default: base_dir)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    
    # Generate unaligned TXT
    csv_unaligned = base_dir / "vggt_evaluation_results.csv"
    if csv_unaligned.exists():
        print(f"Reading: {csv_unaligned}")
        results = load_csv_results(csv_unaligned)
        output_txt = output_dir / "vggt_eval_results_noalign.txt"
        save_summary_txt(results, output_txt, use_alignment=False)
    else:
        print(f"Warning: {csv_unaligned} not found")
    
    # Generate aligned TXT
    csv_aligned = base_dir / "vggt_evaluation_results_aligned.csv"
    if csv_aligned.exists():
        print(f"Reading: {csv_aligned}")
        results = load_csv_results(csv_aligned)
        output_txt = output_dir / "vggt_eval_results_aligned.txt"
        save_summary_txt(results, output_txt, use_alignment=True)
    else:
        print(f"Warning: {csv_aligned} not found")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
