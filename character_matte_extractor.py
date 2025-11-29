import math
import pathlib
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DirectoryCharacterMatteExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
                "bg_tolerance": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "value_min": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "saturation_max": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    def _estimate_background(self, img: torch.Tensor, corner: int = 16) -> torch.Tensor:
        h, w, _ = img.shape
        c = max(1, min(corner, h // 2, w // 2))
        patches = [
            img[0:c, 0:c],
            img[0:c, w - c : w],
            img[h - c : h, 0:c],
            img[h - c : h, w - c : w],
        ]
        bg = torch.stack([p.reshape(-1, 3).mean(dim=0) for p in patches], dim=0).mean(dim=0)
        return bg

    def _rgb_to_sv(self, img: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        max_c, _ = img.max(dim=-1)
        min_c, _ = img.min(dim=-1)
        value = max_c
        saturation = (max_c - min_c) / (max_c + 1e-6)
        return saturation, value

    def _make_bg_candidate(
        self,
        img: torch.Tensor,
        bg_color: torch.Tensor,
        bg_tolerance: float,
        value_min: float,
        saturation_max: float,
    ) -> torch.Tensor:
        diff = torch.sqrt(torch.sum((img - bg_color) ** 2, dim=-1))
        saturation, value = self._rgb_to_sv(img)
        is_bg_candidate = (diff <= bg_tolerance) & (value >= value_min) & (saturation <= saturation_max)
        return is_bg_candidate

    def _flood_fill_from_border(self, bg_candidate: torch.Tensor) -> torch.Tensor:
        """
        bg_candidate: bool [H, W] – paper-like candidate pixels.
        Returns a bool mask [H, W] where True means “background connected to image border”.
        """
        h, w = bg_candidate.shape
        visited = torch.zeros_like(bg_candidate, dtype=torch.bool)
        q = deque()

        for x in range(w):
            if bg_candidate[0, x]:
                q.append((0, x))
            if bg_candidate[h - 1, x]:
                q.append((h - 1, x))
        for y in range(h):
            if bg_candidate[y, 0]:
                q.append((y, 0))
            if bg_candidate[y, w - 1]:
                q.append((y, w - 1))

        while q:
            y, x = q.popleft()
            if visited[y, x]:
                continue
            visited[y, x] = True
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w:
                    if bg_candidate[ny, nx] and not visited[ny, nx]:
                        q.append((ny, nx))

        return visited

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

        fg_mask = alpha >= 0.5
        h, w = fg_mask.shape
        visited = torch.zeros_like(fg_mask, dtype=torch.bool)
        alpha_out = alpha.clone()

        for y in range(h):
            for x in range(w):
                if not fg_mask[y, x] or visited[y, x]:
                    continue

                q = deque()
                q.append((y, x))
                coords = []
                touches_border = False

                while q:
                    cy, cx = q.popleft()
                    if visited[cy, cx] or not fg_mask[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    coords.append((cy, cx))
                    if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                        touches_border = True
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if fg_mask[ny, nx] and not visited[ny, nx]:
                                q.append((ny, nx))

                if len(coords) < min_area and not touches_border:
                    for cy, cx in coords:
                        alpha_out[cy, cx] = 0.0

        return alpha_out

    def _remove_large_holes(
        self,
        alpha: torch.Tensor,
        holes_mask: torch.Tensor,
        min_area: int,
    ) -> torch.Tensor:
        if min_area <= 0:
            return alpha

        h, w = holes_mask.shape
        visited = torch.zeros_like(holes_mask, dtype=torch.bool)
        alpha_out = alpha.clone()

        for y in range(h):
            for x in range(w):
                if not holes_mask[y, x] or visited[y, x]:
                    continue

                q = deque()
                q.append((y, x))
                coords = []

                while q:
                    cy, cx = q.popleft()
                    if visited[cy, cx] or not holes_mask[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    coords.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if holes_mask[ny, nx] and not visited[ny, nx]:
                                q.append((ny, nx))

                if len(coords) >= min_area:
                    for cy, cx in coords:
                        alpha_out[cy, cx] = 0.0

        return alpha_out

    def extract(
        self,
        directory_path: str,
        bg_tolerance: float,
        value_min: float,
        saturation_max: float,
        close_gaps: int,
        edge_feather: int,
        matte_shift: int,
        min_foreground_area: int,
        min_hole_area: int,
    ):
        paths = self._list_images(directory_path)
        if not paths:
            raise ValueError("No PNG/JPG images found in the directory.")
        images = []
        for path in paths:
            img = self._load_image(path)
            bg_color = self._estimate_background(img)
            bg_candidate = self._make_bg_candidate(img, bg_color, bg_tolerance, value_min, saturation_max)
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
            images.append(rgba)
        batch = torch.stack(images, dim=0).clamp(0.0, 1.0)
        return (batch,)


NODE_CLASS_MAPPINGS = {"DirectoryCharacterMatteExtractor": DirectoryCharacterMatteExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"DirectoryCharacterMatteExtractor": "Directory Character Matte Extractor"}
