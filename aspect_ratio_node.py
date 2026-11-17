"""
ComfyUI Custom Node: Aspect Ratio Converter
Changes the aspect ratio of an image to 16:9, 1:1 or 9:16.
Supports two modes: crop (center crop) and pad (fill with a solid color).
"""

import torch


class AspectRatioConverter:
    """
    Converts an image to the selected aspect ratio.
    Supports crop (center crop) and pad (letterbox/pillarbox).
    """

    ASPECT_RATIOS = {
        "16:9":  (16, 9),
        "1:1":   (1,  1),
        "9:16":  (9,  16),
    }

    MODES = ["crop", "pad"]

    PAD_COLORS = {
        "black":  [0,   0,   0  ],
        "white":  [255, 255, 255],
        "gray":   [128, 128, 128],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(cls.ASPECT_RATIOS.keys()),),
                "mode": (cls.MODES,),
            },
            "optional": {
                "pad_color": (list(cls.PAD_COLORS.keys()), {"default": "black"}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "convert"
    CATEGORY      = "image/transform"
    OUTPUT_NODE   = False

    def convert(self, image: torch.Tensor, aspect_ratio: str, mode: str, pad_color: str = "black"):
        """
        image: (B, H, W, C) float32 tensor with values in [0, 1]
        Returns a (B, H', W', C) tensor.
        """
        target_w_ratio, target_h_ratio = self.ASPECT_RATIOS[aspect_ratio]

        results = []
        for img in image:                       # img: (H, W, C)
            if mode == "crop":
                out = self._crop(img, target_w_ratio, target_h_ratio)
            else:
                out = self._pad(img, target_w_ratio, target_h_ratio, pad_color)
            results.append(out)

        return (torch.stack(results),)

    def _crop(self, img: torch.Tensor, tw: int, th: int) -> torch.Tensor:
        """Center crop to the target aspect ratio."""
        H, W, C = img.shape
        target_ratio = tw / th

        if W / H > target_ratio:
            # Too wide: trim the width
            new_w = int(H * target_ratio)
            x0 = (W - new_w) // 2
            return img[:, x0:x0 + new_w, :]
        else:
            # Too tall: trim the height
            new_h = int(W / target_ratio)
            y0 = (H - new_h) // 2
            return img[y0:y0 + new_h, :, :]

    def _pad(self, img: torch.Tensor, tw: int, th: int, pad_color: str) -> torch.Tensor:
        """Letterbox/pillarbox padding to the target aspect ratio."""
        H, W, C = img.shape
        target_ratio = tw / th
        color_val = self.PAD_COLORS.get(pad_color, [0, 0, 0])

        if W / H > target_ratio:
            # Too wide: add height (letterbox)
            new_h = int(W / target_ratio)
            canvas = self._make_canvas(new_h, W, C, color_val, img.device)
            y0 = (new_h - H) // 2
            canvas[y0:y0 + H, :, :] = img
        else:
            # Too tall: add width (pillarbox)
            new_w = int(H * target_ratio)
            canvas = self._make_canvas(H, new_w, C, color_val, img.device)
            x0 = (new_w - W) // 2
            canvas[:, x0:x0 + W, :] = img

        return canvas

    @staticmethod
    def _make_canvas(H: int, W: int, C: int, color: list, device) -> torch.Tensor:
        """Creates a solid-color (H, W, C) float32 tensor in [0, 1]."""
        canvas = torch.zeros((H, W, C), dtype=torch.float32, device=device)
        for c_idx, val in enumerate(color[:C]):
            canvas[:, :, c_idx] = val / 255.0
        return canvas


NODE_CLASS_MAPPINGS = {
    "AspectRatioConverter": AspectRatioConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectRatioConverter": "Aspect Ratio Converter",
}
