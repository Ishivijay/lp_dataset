import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save raw depth values as .npy for evaluation')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'/path/to/work/checkpoint/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        # Get RAW depth prediction (NOT normalized)
        depth_raw = depth_anything.infer_image(raw_image, args.input_size)
        
        # Save RAW depth values for evaluation if requested
        if args.save_numpy:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            raw_depth_path = os.path.join(args.outdir, base_name + '_depth.npy')
            np.save(raw_depth_path, depth_raw.astype(np.float32))
        
        # Create visualization (normalized for display)
        depth_viz = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min()) * 255.0
        depth_viz = depth_viz.astype(np.uint8)
        
        if args.grayscale:
            depth_viz = np.repeat(depth_viz[..., np.newaxis], 3, axis=-1)
        else:
            depth_viz = (cmap(depth_viz)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, base_name + '.png'), depth_viz)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth_viz])
            
            cv2.imwrite(os.path.join(args.outdir, base_name + '.png'), combined_result)