import os
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, '/path/to/home/ml-depth-pro/src')
import depth_pro
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import matplotlib.pyplot as plt

def process_single_image(args):
    """Process a single image using shared model"""
    image_path, output_dir, model, transform, device, lock = args
    
    try:
        # Check if already processed
        output_filename = image_path.stem + "_depth.npy"
        output_path = output_dir / output_filename
        
        if output_path.exists():
            return f"Skipped (exists): {image_path.name}", True
        
        # Load and preprocess image
        image, _, f_px = depth_pro.load_rgb(str(image_path))
        image = transform(image)
        image = image.to(device)
        
        # Run inference (lock to prevent concurrent GPU access issues)
        with lock:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    prediction = model.infer(image, f_px=f_px)
        
        depth = prediction["depth"]
        focallength_px = prediction["focallength_px"]
        
        # Convert to numpy
        depth_np = depth.cpu().numpy()
        
        # Save depth as numpy (.npy format)
        np.save(output_path, depth_np)
        
        # Save focal length
        focal_output_path = output_dir / (image_path.stem + "_focal.txt")
        with open(focal_output_path, 'w') as f:
            f.write(f"{focallength_px}\n")
        
        # Create colorized visualization using inverse depth (same as ML-Depth-Pro)
        inverse_depth = 1.0 / depth_np
        
        # Normalize inverse depth for visualization (same range as ML-Depth-Pro)
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )
        
        # Apply "turbo" colormap (same as ML-Depth-Pro)
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
        
        # Save as JPG
        jpg_output_path = output_dir / (image_path.stem + ".jpg")
        Image.fromarray(color_depth).save(jpg_output_path, format="JPEG", quality=90)
        
        return f"Processed: {image_path.name}", True
        
    except Exception as e:
        return f"Error processing {image_path.name}: {str(e)}", False

def main():
    # Base directory containing all sequences
    base_dir = Path("/path/to/work/mlp_dataset/prospthesisproject-Data/Code/Data")
    
    # Sequences to process
    sequences = [
        "lms_kamal_LA_downstairs_Nopeople_1",
        "lms_kamal_LA_upstairs_Nopeople_1", 
        "lms_kamal_LB_upstairs_Nopeople_2",
        "lms_kamal_RA_downstairs_Nopeople_1", 
        "lms_kamal_RA_upstairs_Nopeople_1",
        "lms_kamal_RB_downstairs_Nopeople_1"
    ]
    
    # Load model once (shared across threads)
    print("Loading ML-Depth-Pro model...")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Lock for thread-safe GPU access
    lock = threading.Lock()
    
    # Number of threads - keep low for GPU to avoid memory issues
    num_workers = 4
    
    print(f"Using {num_workers} worker threads")
    print("Generating: .npy (depth), .txt (focal), .jpg (turbo colormap visualization)\n")
    
    # Process each sequence
    for seq in sequences:
        print(f"\n{'='*60}")
        print(f"Processing sequence: {seq}")
        print(f"{'='*60}")
        
        # Input directory
        rgb_dir = base_dir / seq / "sharpen_rgb" / "PNG"
        
        # Output directory (create if doesn't exist)
        output_dir = base_dir / seq / "ml_depth_pro_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not rgb_dir.exists():
            print(f"Warning: RGB directory not found: {rgb_dir}")
            continue
        
        # Get all PNG files
        image_files = sorted(rgb_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"Warning: No PNG files found in {rgb_dir}")
            continue
        
        print(f"Found {len(image_files)} images to process")
        
        # Prepare arguments for parallel processing
        args_list = [(img_path, output_dir, model, transform, device, lock) 
                     for img_path in image_files]
        
        # Process images with threading
        processed_count = 0
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_image, args): args for args in args_list}
            
            for future in as_completed(futures):
                result, success = future.result()
                processed_count += 1
                if success:
                    success_count += 1
                
                if processed_count % 50 == 0:
                    print(f"  Progress: {processed_count}/{len(image_files)} images ({success_count} successful)")
        
        print(f"Completed {seq}: {success_count}/{len(image_files)} images processed successfully")
        print(f"Output saved to: {output_dir}")
    
    print(f"\n{'='*60}")
    print("All sequences processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()