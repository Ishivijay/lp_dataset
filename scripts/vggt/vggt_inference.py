#!/usr/bin/env python3
"""
vggt_inference.py
Zero-shot inference using VGGT model on prosthesis dataset

Note: VGGT is designed for multi-view 3D reconstruction, but can also work on single images.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import json

# Add VGGT to path
sys.path.insert(0, '/path/to/home/vggt')

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


class VGGTInference:
    """Wrapper for VGGT inference on prosthesis dataset"""
    
    def __init__(self, model_path=None, device='cuda', use_commercial=False):
        """
        Initialize VGGT model for inference
        
        Args:
            model_path: Path to pretrained model checkpoint (optional)
            device: Device to run inference on ('cuda' or 'cpu')
            use_commercial: Use commercial license model (VGGT-1B-Commercial)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Determine dtype based on GPU capability (per official README)
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        
        print(f"Using dtype: {self.dtype}")
        
        # Load model
        print("Loading VGGT model...")
        self.model = self._load_model(model_path, use_commercial)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print("VGGT model loaded successfully")
    
    def _load_model(self, model_path, use_commercial):
        """Load VGGT model from checkpoint or Hugging Face (per official README)"""
        if model_path and os.path.exists(model_path):
            # Load from local checkpoint
            print(f"Loading model from {model_path}")
            model = VGGT()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            # Load from Hugging Face (per official README)
            if use_commercial:
                print("Loading VGGT-1B-Commercial from Hugging Face...")
                print("Note: This requires access approval from Meta")
                model = VGGT.from_pretrained("facebook/VGGT-1B-Commercial")
            else:
                print("Loading VGGT-1B from Hugging Face (non-commercial)...")
                try:
                    # Method 1: from_pretrained (per official README)
                    model = VGGT.from_pretrained("facebook/VGGT-1B")
                except Exception as e:
                    print(f"from_pretrained failed: {e}")
                    print("Trying alternative loading method...")
                    # Method 2: Direct URL download (per official README)
                    model = VGGT()
                    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
                    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        
        return model
    
    def predict(self, image_path):
        """
        Run VGGT prediction on a single image (per official README)
        
        Args:
            image_path: Path to input RGB image
            
        Returns:
            dict: Predictions containing depth, world_points, cameras, etc.
        """
        # Load and preprocess image using official VGGT function
        images = load_and_preprocess_images([str(image_path)], mode="crop")
        images = images.to(self.device)
        
        # Run inference (per official README)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                predictions = self.model(images)
        
        
        return predictions


def process_single_image(args):
    """Process a single image with VGGT"""
    image_path, output_dir, model, lock = args
    
    try:
        # Check if already processed
        output_filename = image_path.stem + "_vggt_depth.npy"
        output_path = output_dir / output_filename
        
        if output_path.exists():
            return f"Skipped (exists): {image_path.name}", True
        
        # Run inference (lock for thread safety)
        with lock:
            predictions = model.predict(str(image_path))
        
        # Extract depth map (shape: [1, S, H, W, 1] or [B, S, H, W, 1])
        depth = predictions["depth"].squeeze().cpu().numpy()
        
        # Save depth map
        np.save(output_path, depth)
        
        # Save visualization
        vis_path = output_dir / (image_path.stem + "_vggt_depth_vis.png")
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(vis_path)
        
        return f"Processed: {image_path.name}", True
        
    except Exception as e:
        return f"Error processing {image_path.name}: {str(e)}", False


def process_sequence(sequence_path, model, num_workers=1):
    """
    Process all images in a sequence
    
    Args:
        sequence_path: Path to sequence directory
        model: VGGTInference instance
        num_workers: Number of worker threads (use 1 for GPU to avoid conflicts)
    """
    # Input and output directories
    rgb_dir = sequence_path / "sharpen_rgb" / "PNG"
    output_dir = sequence_path / "vggt_predictions"
    output_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = sorted(rgb_dir.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {rgb_dir}")
        return
    
    print(f"\nProcessing {len(image_files)} images from {sequence_path.name}")
    
    # Thread lock for model access
    lock = threading.Lock()
    
    # Process images
    tasks = [(img_path, output_dir, model, lock) for img_path in image_files]
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=sequence_path.name):
            message, success = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                print(f"\n{message}")
    
    print(f"\nCompleted {sequence_path.name}:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='VGGT Zero-Shot Inference on Prosthesis Dataset'
    )
    parser.add_argument('--base_dir', type=str,
                       default='/path/to/work/mlp_dataset/prospthesisproject-Data/Code/Data',
                       help='Base directory containing sequences')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to local model checkpoint (optional)')
    parser.add_argument('--use_commercial', action='store_true',
                       help='Use VGGT-1B-Commercial model (requires access)')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of worker threads (use 1 for GPU)')
    parser.add_argument('--sequences', nargs='+', default=None,
                       help='Specific sequences to process (default: all)')
    
    args = parser.parse_args()
    
    BASE_DIR = Path(args.base_dir)
    
    # Prosthesis sequences
    ALL_SEQUENCES = [
        "lms_kamal_LA_downstairs_Nopeople_1",
        "lms_kamal_LA_upstairs_Nopeople_1",
        "lms_kamal_LB_upstairs_Nopeople_2",
        "lms_kamal_RA_downstairs_Nopeople_1",
        "lms_kamal_RA_upstairs_Nopeople_1",
        "lms_kamal_RB_downstairs_Nopeople_1"
    ]
    
    SEQUENCES = args.sequences if args.sequences else ALL_SEQUENCES
    
    print("=" * 80)
    print("VGGT Zero-Shot Inference on Prosthesis Dataset")
    print("=" * 80)
    print(f"\nNote: VGGT is primarily designed for multi-view 3D reconstruction.")
    print(f"It can work on single images, but models like Depth Anything V2 may be better.")
    print("=" * 80)
    
    # Initialize model
    try:
        model = VGGTInference(
            model_path=args.model_path,
            use_commercial=args.use_commercial
        )
    except Exception as e:
        print(f"\nError initializing VGGT model: {e}")
        print("\nPlease ensure:")
        print("1. VGGT repository is cloned to /path/to/home/vggt")
        print("2. All dependencies are installed (pip install -r requirements.txt)")
        print("3. You have internet access to download model weights from Hugging Face")
        return
    
    # Process all sequences
    for seq_name in SEQUENCES:
        seq_path = BASE_DIR / seq_name
        if not seq_path.exists():
            print(f"Warning: Sequence not found: {seq_path}")
            continue
        
        try:
            process_sequence(seq_path, model, num_workers=args.num_workers)
        except Exception as e:
            print(f"Error processing sequence {seq_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("VGGT inference completed!")
    print("=" * 80)
    
    # Save run info
    run_info = {
        "model": "VGGT" + ("-Commercial" if args.use_commercial else ""),
        "sequences": SEQUENCES,
        "base_dir": str(BASE_DIR),
        "num_workers": args.num_workers
    }
    
    log_dir = Path("/path/to/home/log")
    log_dir.mkdir(exist_ok=True)
    
    with open(log_dir / "vggt_inference_info.json", 'w') as f:
        json.dump(run_info, f, indent=2)


if __name__ == '__main__':
    main()