import math
import pathlib
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


class CharacterMatteExtractorBase:
    def _parse_hex_color(self, hex_color: str) -> torch.Tensor:
        hex_color = hex_color.strip().lstrip("#")
        if not hex_color:
            return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32) # Default white
            
        try:
            if len(hex_color) == 6:
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif len(hex_color) == 3:
                r, g, b = tuple(int(hex_color[i]*2, 16) for i in (0, 1, 2))
            else:
                 return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            
            return torch.tensor([r, g, b], dtype=torch.float32) / 255.0
        except ValueError:
            return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    def _make_bg_candidate(
        self,
        img: torch.Tensor,
        bg_color: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        # Move bg_color to same device as img
        bg_color = bg_color.to(img.device)
        
        # Calculate Euclidean distance in RGB space
        diff = torch.sqrt(torch.sum((img - bg_color) ** 2, dim=-1))
        
        # Simple thresholding based on color distance
        is_bg_candidate = (diff <= threshold)
        return is_bg_candidate

    def _flood_fill_from_border(self, bg_candidate: torch.Tensor) -> torch.Tensor:
        """
        bg_candidate: bool [H, W] – paper-like candidate pixels.
        Returns a bool mask [H, W] where True means “background connected to image border”.
        """
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for this node. Please install it with: pip install opencv-python")

        # Convert to uint8 for OpenCV
        mask = bg_candidate.cpu().numpy().astype(np.uint8) * 255
        h, w = mask.shape
        
        # Mask for floodFill must be 2 pixels larger in both dimensions
        # and initialized to 0.
        fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        
        # Flood fill from all border pixels
        # Top and bottom
        for x in range(w):
            if mask[0, x] == 255:
                cv2.floodFill(mask, fill_mask, (x, 0), 128)
            if mask[h - 1, x] == 255:
                cv2.floodFill(mask, fill_mask, (x, h - 1), 128)
                
        # Left and right
        for y in range(h):
            if mask[y, 0] == 255:
                cv2.floodFill(mask, fill_mask, (0, y), 128)
            if mask[y, w - 1] == 255:
                cv2.floodFill(mask, fill_mask, (w - 1, y), 128)

        # Pixels that were filled are now 128. Returns True where filled.
        # Note: We modified 'mask' in-place with 128 where filled.
        # But 'fill_mask' also tracks where it filled (roi).
        # To be safe and consistent with logic: 
        # The filled area (connected to border) is now 128. Original bg_candidate was 255.
        
        return torch.from_numpy(mask == 128).to(bg_candidate.device)

    def _close_gaps(self, alpha: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return alpha
        k = 2 * radius + 1
        pad = radius
        a = alpha.unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(a, kernel_size=k, stride=1, padding=pad)
        eroded = -F.max_pool2d(-dilated, kernel_size=k, stride=1, padding=pad)
        return eroded.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    def _feather(self, alpha: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return alpha
        k = 2 * radius + 1
        kernel = torch.ones((1, 1, k, k), device=alpha.device, dtype=alpha.dtype) / float(k * k)
        a = alpha.unsqueeze(0).unsqueeze(0)
        blurred = F.conv2d(a, kernel, padding=radius)
        return blurred.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    def _shift_matte(self, alpha: torch.Tensor, pixels: int) -> torch.Tensor:
        """
        Shift the matte by a given number of pixels.
        pixels > 0: expand (dilate) opaque area outward.
        pixels < 0: shrink (erode) opaque area inward.
        pixels = 0: no change.
        """
        if pixels == 0:
            return alpha

        radius = abs(int(pixels))
        if radius == 0:
            return alpha

        k = 2 * radius + 1
        a = alpha.unsqueeze(0).unsqueeze(0)

        if pixels > 0:
            shifted = F.max_pool2d(a, kernel_size=k, stride=1, padding=radius)
        else:
            shifted = -F.max_pool2d(-a, kernel_size=k, stride=1, padding=radius)

        return shifted.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    def _remove_small_components(self, alpha: torch.Tensor, min_area: int) -> torch.Tensor:
        if min_area <= 0:
            return alpha
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for this node.")

        # alpha is float 0.0-1.0. Convert to uint8 binary mask.
        mask = (alpha.cpu().numpy() >= 0.5).astype(np.uint8) * 255
        
        # Connected components with stats
        # connectivity=8 is standard
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # stats: [x, y, width, height, area]
        
        new_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels): # Skip label 0 (black background)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                # Keep this component
                new_mask[labels == i] = 255
                
            # Extra check: In original logic, we preserved components touching borders?
            # Original code: "if len(coords) < min_area and not touches_border: remove"
            # It means: if it touches border, we keep it EVEN IF small.
            # Let's replicate this logic using bounding box.
            else:
                x, y, w, h = stats[i, :4]
                img_h, img_w = mask.shape
                touches_border = (x == 0) or (y == 0) or (x + w == img_w) or (y + h == img_h)
                if touches_border:
                    new_mask[labels == i] = 255

        return torch.from_numpy(new_mask.astype(np.float32) / 255.0).to(alpha.device)

    def _remove_large_holes(
        self,
        alpha: torch.Tensor,
        holes_mask: torch.Tensor,
        min_area: int,
    ) -> torch.Tensor:
        if min_area <= 0:
            return alpha
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for this node.")

        mask = holes_mask.cpu().numpy().astype(np.uint8) * 255
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # alpha is currently tensor.
        alpha_np = alpha.cpu().numpy().copy()
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                # This is a large hole. Punch it out (set to 0).
                alpha_np[labels == i] = 0.0
                
        return torch.from_numpy(alpha_np).to(alpha.device)

    def _process_image(
        self,
        img: torch.Tensor,
        bg_color_tensor: torch.Tensor,
        threshold: float,
        close_gaps: int,
        edge_feather: int,
        matte_shift: int,
        min_foreground_area: int,
        min_hole_area: int,
    ) -> torch.Tensor:
        bg_candidate = self._make_bg_candidate(img, bg_color_tensor, threshold)
        if close_gaps > 0:
            fg_mask = (~bg_candidate).float()
            fg_mask = self._close_gaps(fg_mask, int(close_gaps))
            bg_candidate = ~(fg_mask > 0.5)
            
        bg_mask = self._flood_fill_from_border(bg_candidate)
        
        holes_mask = bg_candidate & (~bg_mask)
        alpha = (~bg_mask).float()
        
        if matte_shift != 0:
            alpha = self._shift_matte(alpha, int(matte_shift))
        if min_hole_area > 0:
            alpha = self._remove_large_holes(alpha, holes_mask, int(min_hole_area))
        if min_foreground_area > 0:
            alpha = self._remove_small_components(alpha, int(min_foreground_area))
        if edge_feather > 0:
            alpha = self._feather(alpha, int(edge_feather))
            
        rgba = torch.cat([img, alpha.unsqueeze(-1)], dim=-1)
        return rgba


class DirectoryCharacterMatteExtractor(CharacterMatteExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
                "background_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "close_gaps": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
                "edge_feather": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
                "matte_shift": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1}),
                "min_foreground_area": ("INT", {"default": 16, "min": 0, "max": 10000, "step": 1}),
                "min_hole_area": ("INT", {"default": 128, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "extract"
    CATEGORY = "image/matting"

    def _list_images(self, directory: str) -> List[pathlib.Path]:
        exts = {".png", ".jpg", ".jpeg"}
        root = pathlib.Path(directory).expanduser()
        if not root.is_dir():
            raise ValueError(f"Directory not found: {root}")
        return [p for p in sorted(root.iterdir()) if p.suffix.lower() in exts and p.is_file()]

    def _load_image(self, path: pathlib.Path) -> torch.Tensor:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    def extract(
        self,
        directory_path: str,
        background_color: str,
        threshold: float,
        close_gaps: int,
        edge_feather: int,
        matte_shift: int,
        min_foreground_area: int,
        min_hole_area: int,
    ):
        paths = self._list_images(directory_path)
        if not paths:
            raise ValueError("No PNG/JPG images found in the directory.")
        
        bg_color_tensor = self._parse_hex_color(background_color)
        
        images = []
        for path in paths:
            img = self._load_image(path)
            rgba = self._process_image(
                img,
                bg_color_tensor,
                threshold,
                close_gaps,
                edge_feather,
                matte_shift,
                min_foreground_area,
                min_hole_area,
            )
            images.append(rgba)
        batch = torch.stack(images, dim=0).clamp(0.0, 1.0)
        return (batch,)


class CharacterMatteExtractor(CharacterMatteExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "background_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "close_gaps": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
                "edge_feather": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
                "matte_shift": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1}),
                "min_foreground_area": ("INT", {"default": 16, "min": 0, "max": 10000, "step": 1}),
                "min_hole_area": ("INT", {"default": 128, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "extract"
    CATEGORY = "image/matting"

    def extract(
        self,
        images: torch.Tensor,
        background_color: str,
        threshold: float,
        close_gaps: int,
        edge_feather: int,
        matte_shift: int,
        min_foreground_area: int,
        min_hole_area: int,
    ):
        bg_color_tensor = self._parse_hex_color(background_color)
        
        results = []
        for img in images:
            rgba = self._process_image(
                img,
                bg_color_tensor,
                threshold,
                close_gaps,
                edge_feather,
                matte_shift,
                min_foreground_area,
                min_hole_area,
            )
            results.append(rgba)
        
        batch = torch.stack(results, dim=0).clamp(0.0, 1.0)
        return (batch,)

NODE_CLASS_MAPPINGS = {
    "DirectoryCharacterMatteExtractor": DirectoryCharacterMatteExtractor,
    "CharacterMatteExtractor": CharacterMatteExtractor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectoryCharacterMatteExtractor": "Directory Character Matte Extractor",
    "CharacterMatteExtractor": "Character Matte Extractor",
}
