import os
import re
import numpy as np
import torch
from PIL import Image, ImageFilter


class LineArtExtractor:
    """
    Extracts crisp line art from a monochrome input.

    The node takes an IMAGE tensor, thresholds it to isolate the ink,
    optionally denoises with a small median filter, and outputs a PNG-ready
    IMAGE where the lines are black and the background is transparent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "invert_input": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "median_filter": (
                    "INT",
                    {"default": 3, "min": 1, "max": 15, "step": 2},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("line_art",)
    FUNCTION = "extract"
    CATEGORY = "image/processing"

    def extract(self, image, threshold=0.5, invert_input=False, median_filter=3):
        median_filter = _normalize_filter_size(median_filter)

        output_batches = []
        for batch_img in image:
            # Convert torch image [H, W, C] float 0-1 to uint8 array for PIL.
            np_img = (
                np.clip(batch_img.detach().cpu().numpy(), 0.0, 1.0) * 255
            ).astype(np.uint8)
            pil_img = Image.fromarray(np_img)

            output_batches.append(
                _line_art_from_pil(
                    pil_img,
                    threshold=threshold,
                    invert_input=invert_input,
                    median_filter=median_filter,
                )
            )

        stacked = torch.stack(output_batches, dim=0)
        return (stacked,)


class VideoLineArtExtractor:
    """
    Loads a black-and-white video file and emits a batch of RGBA frames with
    transparent backgrounds, suitable for saving as a numbered PNG sequence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "invert_input": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "median_filter": (
                    "INT",
                    {"default": 3, "min": 1, "max": 15, "step": 2},
                ),
                "sample_every": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("line_art_frames",)
    FUNCTION = "extract_video"
    CATEGORY = "image/processing"

    def extract_video(
        self,
        video_path,
        threshold=0.5,
        invert_input=False,
        median_filter=3,
        sample_every=1,
    ):
        median_filter = _normalize_filter_size(median_filter)
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            import cv2
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OpenCV (cv2) is required for VideoLineArtExtractor. "
                "Install with: pip install opencv-python"
            ) from exc

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every != 0:
                idx += 1
                continue

            # OpenCV gives BGR; convert to RGB then to PIL for reuse.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            frames.append(
                _line_art_from_pil(
                    pil_img,
                    threshold=threshold,
                    invert_input=invert_input,
                    median_filter=median_filter,
                )
            )
            idx += 1

        cap.release()

        if not frames:
            raise ValueError("No frames were processed from the video.")

        stacked = torch.stack(frames, dim=0)
        return (stacked,)


class DirectoryLineArtExtractor:
    """
    Reads a directory of sequential monochrome images and emits RGBA frames
    with transparent backgrounds, preserving the on-disk ordering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "invert_input": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "median_filter": (
                    "INT",
                    {"default": 3, "min": 1, "max": 15, "step": 2},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("line_art_frames",)
    FUNCTION = "extract_directory"
    CATEGORY = "image/processing"

    def extract_directory(
        self,
        directory_path,
        threshold=0.5,
        invert_input=False,
        median_filter=3,
    ):
        median_filter = _normalize_filter_size(median_filter)
        if not directory_path or not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [
            os.path.join(directory_path, name)
            for name in os.listdir(directory_path)
            if os.path.splitext(name.lower())[1] in supported
        ]
        files.sort(key=_natural_key)

        if not files:
            raise ValueError("No supported image files were found in the directory.")

        frames = []
        for path in files:
            with Image.open(path) as pil_img:
                pil_rgb = pil_img.convert("RGB")
                frames.append(
                    _line_art_from_pil(
                        pil_rgb,
                        threshold=threshold,
                        invert_input=invert_input,
                        median_filter=median_filter,
                    )
                )

        stacked = torch.stack(frames, dim=0)
        return (stacked,)


def _normalize_filter_size(median_filter: int) -> int:
    """Clamp to at least 1 and ensure it is odd for PIL median filter."""
    median_filter = max(1, int(median_filter))
    if median_filter % 2 == 0:
        median_filter += 1
    return median_filter


def _line_art_from_pil(
    pil_img: Image.Image,
    threshold: float,
    invert_input: bool,
    median_filter: int,
) -> torch.Tensor:
    """Convert a PIL image to RGBA line art tensor with transparent bg."""
    # Grayscale then optional small denoise for steadier lines.
    gray = pil_img.convert("L")
    if median_filter > 1:
        gray = gray.filter(ImageFilter.MedianFilter(size=median_filter))
    if invert_input:
        gray = Image.fromarray(255 - np.array(gray, dtype=np.uint8))

    gray_np = np.array(gray, dtype=np.uint8)
    cutoff = int(threshold * 255)
    alpha_mask = (gray_np >= cutoff).astype(np.uint8) * 255

    # Black ink with transparent background.
    h, w = gray_np.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 3] = alpha_mask

    return torch.from_numpy(rgba / 255.0)


def _natural_key(path: str):
    """Sort helper that preserves numeric order in filenames like 0001.png."""
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


NODE_CLASS_MAPPINGS = {
    "LineArtExtractor": LineArtExtractor,
    "VideoLineArtExtractor": VideoLineArtExtractor,
    "DirectoryLineArtExtractor": DirectoryLineArtExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LineArtExtractor": "Line Art Extractor (PNG)",
    "VideoLineArtExtractor": "Video Line Art Extractor (PNG sequence)",
    "DirectoryLineArtExtractor": "Directory Line Art Extractor (PNG sequence)",
}
