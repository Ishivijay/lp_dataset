import torch
import numpy as np

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class MLPDepthDataset(BaseDepthDataset):
    def __init__(
        self,
        eigen_valid_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            # MLP prosthesis dataset parameters
            min_depth=0.1,  
            max_depth=10.0,  
            has_filled_depth=False,  
            name_mode=DepthFileNameMode.id,  
            **kwargs,
        )
        
        self.eigen_valid_mask = eigen_valid_mask

    def _read_depth_file(self, rel_path):
        """Read depth file for MLP prosthesis dataset"""
        depth_in = self._read_image(rel_path)

        if depth_in.dtype == np.uint16:
            depth_decoded = depth_in.astype(np.float32) / 1000.0  
        elif depth_in.dtype == np.uint8:
            depth_decoded = depth_in.astype(np.float32) / 255.0 * self.max_depth
        else:
            depth_decoded = depth_in.astype(np.float32)
        
        depth_decoded = np.clip(depth_decoded, self.min_depth, self.max_depth)
        
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        """Create valid mask for MLP dataset"""
        valid_mask = super()._get_valid_mask(depth)

        if self.eigen_valid_mask:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            H, W = eval_mask.shape
            
            h_start = int(0.0 * H)  
            h_end = int(1.0 * H)    
            w_start = int(0.0 * W)  
            w_end = int(1.0 * W)    
            
            eval_mask[h_start:h_end, w_start:w_end] = 1
            eval_mask = eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)

        return valid_mask