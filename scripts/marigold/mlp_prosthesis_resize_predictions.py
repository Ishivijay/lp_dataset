import numpy as np
import os
from PIL import Image
import cv2
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def resize_single_file(args):
    """Resize a single prediction file - designed for parallel execution"""
    pred_file, target_shape = args
    try:
        # Load prediction
        pred = np.load(pred_file)
        
        # Handle different array types
        if pred.ndim == 0:
            return ('skipped', pred_file)
        elif pred.ndim == 1:
            # Try to make square
            size = int(np.sqrt(len(pred)))
            if size * size == len(pred):
                pred = pred.reshape((size, size))
            else:
                return ('skipped', pred_file)
        
        # Ensure 2D
        if pred.ndim == 3:
            pred = pred[:, :, 0] if pred.shape[2] == 1 else pred[:, :, 0]
        
        # Resize if needed
        if pred.shape != target_shape:
            # Convert to float32 for OpenCV
            if pred.dtype != np.float32:
                pred = pred.astype(np.float32)
            
            # Resize
            target_size = (target_shape[1], target_shape[0])  # (width, height)
            resized = cv2.resize(pred, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Save back
            np.save(pred_file, resized)
            return ('resized', pred_file)
        else:
            return ('skipped', pred_file)
            
    except Exception as e:
        return ('error', pred_file, str(e))

def resize_all_predictions():
    base_dir = "dataset"
    prediction_base = os.path.join(base_dir, "marigold_predictions")
    
    sequences = [
        "lms_kamal_LA_downstairs_Nopeople_1",
        "lms_kamal_LA_upstairs_Nopeople_1", 
        "lms_kamal_LB_upstairs_Nopeople_2",
        "lms_kamal_RA_downstairs_Nopeople_1", 
        "lms_kamal_RA_upstairs_Nopeople_1",
        "lms_kamal_RB_downstairs_Nopeople_1"
    ]
    
    num_workers = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_workers} parallel workers\n")
    
    for seq in sequences:
        print(f"\nProcessing {seq}...")
        
        pred_dir = os.path.join(prediction_base, seq, "sharpen_rgb", "PNG")
        depth_dir = os.path.join(base_dir, seq, "raw_depth", "PNG")
        
        if not os.path.exists(pred_dir):
            print(f"  No predictions found in {pred_dir}")
            continue
            
        # Get target shape from first ground truth
        depth_files = glob.glob(os.path.join(depth_dir, "*.png"))
        if not depth_files:
            print(f"  No ground truth files found")
            continue
            
        # Load first ground truth
        gt = np.array(Image.open(depth_files[0]))
        target_shape = gt.shape
        print(f"  Target shape: {target_shape}")
        
        # Get all prediction files
        pred_files = glob.glob(os.path.join(pred_dir, "pred_*.npy"))
        print(f"  Found {len(pred_files)} prediction files")
        
        if not pred_files:
            continue
        
        # Prepare arguments for parallel processing
        args_list = [(pred_file, target_shape) for pred_file in pred_files]
        
        resized_count = 0
        skipped_count = 0
        error_count = 0
        
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(resize_single_file, args_list),
                total=len(args_list),
                desc=f"  Resizing {seq}"
            ))
        
        for result in results:
            if result[0] == 'resized':
                resized_count += 1
            elif result[0] == 'skipped':
                skipped_count += 1
            elif result[0] == 'error':
                error_count += 1
                print(f"    ERROR with {os.path.basename(result[1])}: {result[2]}")
        
        print(f"  ✓ Resized: {resized_count}, Skipped: {skipped_count}, Errors: {error_count}")

if __name__ == "__main__":
    resize_all_predictions()
    print("\n✓ All sequences processed!")