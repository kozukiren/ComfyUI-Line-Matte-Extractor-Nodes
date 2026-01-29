import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


class FlatColorPosterizer:
    """
    入力画像から陰影、ハイライト、線画を除去し、
    指定した色数でベタ塗りした画像を生成するノード。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_colors": ("INT", {"default": 4, "min": 2, "max": 16, "step": 1}),
                "smoothing_radius": ("INT", {"default": 5, "min": 0, "max": 20, "step": 1}),
                "edge_preserve": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "color1_hex": ("STRING", {"default": "", "multiline": False}),
                "color2_hex": ("STRING", {"default": "", "multiline": False}),
                "color3_hex": ("STRING", {"default": "", "multiline": False}),
                "color4_hex": ("STRING", {"default": "", "multiline": False}),
                "color5_hex": ("STRING", {"default": "", "multiline": False}),
                "color6_hex": ("STRING", {"default": "", "multiline": False}),
                "color7_hex": ("STRING", {"default": "", "multiline": False}),
                "color8_hex": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "posterize"
    CATEGORY = "image/postprocessing"

    def _parse_hex_color(self, hex_color: str) -> Optional[np.ndarray]:
        """HEX色文字列をRGB配列に変換"""
        if not hex_color or hex_color.strip() == "":
            return None
            
        hex_color = hex_color.strip().lstrip("#")
        try:
            if len(hex_color) == 6:
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif len(hex_color) == 3:
                r, g, b = tuple(int(hex_color[i]*2, 16) for i in (0, 1, 2))
            else:
                return None
            return np.array([r, g, b], dtype=np.float32) / 255.0
        except ValueError:
            return None

    def _bilateral_filter(
        self, 
        img: np.ndarray, 
        radius: int, 
        sigma_color: float, 
        sigma_space: float
    ) -> np.ndarray:
        """バイラテラルフィルタで平滑化（エッジを保持しながら陰影を軽減）"""
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for this node. Please install it with: pip install opencv-python")
        
        if radius <= 0:
            return img
        
        # バイラテラルフィルタは0-255の範囲で動作
        img_uint8 = (img * 255).astype(np.uint8)
        d = radius * 2 + 1
        
        filtered = cv2.bilateralFilter(img_uint8, d, sigma_color * 100, sigma_space * 100)
        
        return filtered.astype(np.float32) / 255.0

    def _kmeans_quantize(
        self, 
        img: np.ndarray, 
        num_colors: int,
        custom_colors: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """K-meansクラスタリングで色数を削減"""
        if KMeans is None:
            raise ImportError("scikit-learn is required for this node. Please install it with: pip install scikit-learn")
        
        h, w, c = img.shape
        pixels = img.reshape(-1, 3)
        
        # カスタムカラーが指定されている場合
        if custom_colors and len(custom_colors) > 0:
            # カスタムカラーを使用して初期化
            num_custom = min(len(custom_colors), num_colors)
            
            if num_custom == num_colors:
                # すべてカスタムカラーで指定されている場合
                palette = np.array(custom_colors[:num_colors])
            else:
                # カスタムカラー + K-meansで残りを抽出
                kmeans = KMeans(n_clusters=num_colors - num_custom, random_state=42, n_init=10)
                kmeans.fit(pixels)
                auto_colors = kmeans.cluster_centers_
                palette = np.vstack([custom_colors[:num_custom], auto_colors])
        else:
            # 自動でK-meansクラスタリング
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            palette = kmeans.cluster_centers_
        
        # 各ピクセルを最も近いパレット色に割り当て
        distances = np.linalg.norm(pixels[:, np.newaxis] - palette[np.newaxis, :], axis=2)
        labels = np.argmin(distances, axis=1)
        
        quantized = palette[labels]
        return quantized.reshape(h, w, c)

    def _median_cut_quantize(
        self, 
        img: np.ndarray, 
        num_colors: int,
        custom_colors: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """Median Cut法で色数を削減（K-meansの代替）"""
        # シンプルな実装：画像を均等に量子化
        # より高度な実装はPILのquantizeメソッドを使用可能
        h, w, c = img.shape
        pixels = img.reshape(-1, 3)
        
        if custom_colors and len(custom_colors) >= num_colors:
            # カスタムカラーのみを使用
            palette = np.array(custom_colors[:num_colors])
        else:
            # 簡易的なMedian Cut: RGB空間を等分割
            bits = int(np.ceil(np.log2(num_colors)))
            step = 1.0 / (2 ** bits)
            
            quantized_pixels = np.floor(pixels / step) * step + step / 2
            quantized_pixels = np.clip(quantized_pixels, 0, 1)
            
            # ユニークな色を抽出してパレット作成
            unique_colors = np.unique(quantized_pixels.reshape(-1, 3), axis=0)
            
            # K-meansで指定色数に削減
            if len(unique_colors) > num_colors:
                if KMeans is not None:
                    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
                    kmeans.fit(unique_colors)
                    palette = kmeans.cluster_centers_
                else:
                    # KMeansが使えない場合は最初のN色を使用
                    palette = unique_colors[:num_colors]
            else:
                palette = unique_colors
            
            # カスタムカラーを追加
            if custom_colors and len(custom_colors) > 0:
                num_custom = min(len(custom_colors), num_colors)
                palette[:num_custom] = custom_colors[:num_custom]
        
        # 各ピクセルを最も近いパレット色に割り当て
        distances = np.linalg.norm(pixels[:, np.newaxis] - palette[np.newaxis, :], axis=2)
        labels = np.argmin(distances, axis=1)
        
        quantized = palette[labels]
        return quantized.reshape(h, w, c)

    def posterize(
        self,
        images: torch.Tensor,
        num_colors: int,
        smoothing_radius: int,
        edge_preserve: float,
        color1_hex: str = "",
        color2_hex: str = "",
        color3_hex: str = "",
        color4_hex: str = "",
        color5_hex: str = "",
        color6_hex: str = "",
        color7_hex: str = "",
        color8_hex: str = "",
    ):
        """
        ベタ塗りポスタリゼーション処理
        
        Args:
            images: 入力画像テンソル [B, H, W, C]
            num_colors: 削減後の色数
            smoothing_radius: 平滑化の半径
            edge_preserve: エッジ保存の強さ（0.0=強く平滑化、1.0=エッジ保持）
            color1_hex ~ color8_hex: カスタムカラー（HEX形式）
        
        Returns:
            ベタ塗りされた画像テンソル
        """
        # カスタムカラーをパース
        custom_colors = []
        for hex_color in [color1_hex, color2_hex, color3_hex, color4_hex, 
                         color5_hex, color6_hex, color7_hex, color8_hex]:
            color = self._parse_hex_color(hex_color)
            if color is not None:
                custom_colors.append(color)
        
        results = []
        
        for img_tensor in images:
            # Tensorをnumpy配列に変換 [H, W, C]
            img = img_tensor.cpu().numpy()
            
            # バイラテラルフィルタで平滑化（陰影・ハイライトを軽減）
            # edge_preserveが小さいほど強く平滑化
            sigma_color = 0.05 + edge_preserve * 0.2  # 0.05 ~ 0.25
            sigma_space = smoothing_radius * 2.0
            
            if smoothing_radius > 0:
                img_smooth = self._bilateral_filter(img, smoothing_radius, sigma_color, sigma_space)
            else:
                img_smooth = img
            
            # K-meansで色数削減（ベタ塗り）
            img_posterized = self._kmeans_quantize(img_smooth, num_colors, custom_colors)
            
            # Tensorに戻す
            result_tensor = torch.from_numpy(img_posterized).float()
            results.append(result_tensor)
        
        # バッチに結合
        batch = torch.stack(results, dim=0).clamp(0.0, 1.0)
        
        return (batch,)


NODE_CLASS_MAPPINGS = {
    "FlatColorPosterizer": FlatColorPosterizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlatColorPosterizer": "Flat Color Posterizer",
}
