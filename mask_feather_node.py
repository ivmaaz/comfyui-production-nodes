"""
ComfyUI Custom Node: Mask Feather
Softens the edges of a mask for natural-looking transitions.
Inputs:  MASK + optional IMAGE (RGB or RGBA)
Outputs: MASK (soft) + IMAGE (RGBA with the soft mask as alpha)
"""

import torch
import torch.nn.functional as F


class MaskFeather:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "radius": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "expand": ("INT", {
                    "default": 2,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES  = ("MASK", "IMAGE")
    RETURN_NAMES  = ("mask", "image")
    FUNCTION      = "feather"
    CATEGORY      = "image/segmentation"
    OUTPUT_NODE   = False

    def feather(self, mask: torch.Tensor, radius: int = 6, expand: int = 2, image=None):
        """
        mask:  (B, H, W) float32 [0, 1]
        image: (B, H, W, C) float32 [0, 1], RGB or RGBA (optional)
        If the mask and image batch sizes differ, the shorter batch is repeated.
        """
        m = mask.reshape(-1, 1, *mask.shape[-2:])

        if expand != 0:
            pad = abs(expand)
            if expand > 0:
                m = F.max_pool2d(m, kernel_size=pad * 2 + 1, stride=1, padding=pad)
            else:
                m = -F.max_pool2d(-m, kernel_size=pad * 2 + 1, stride=1, padding=pad)

        # Replicate the border instead of zero padding so masks touching the image edge stay solid there
        kernel = self._gaussian_kernel(radius * 2 + 1, radius / 3.0, mask.device)
        m = F.pad(m, (radius, radius, radius, radius), mode="replicate")
        soft_mask = F.conv2d(m, kernel).squeeze(1).clamp(0, 1)  # (B, H, W)

        if image is not None:
            batch = max(soft_mask.shape[0], image.shape[0])
            soft_mask = soft_mask[[i % soft_mask.shape[0] for i in range(batch)]]
            image = image[[i % image.shape[0] for i in range(batch)]]
            # RGBA input has its alpha replaced, RGB input gets an alpha channel appended
            image_out = torch.cat([image[..., :3], soft_mask.unsqueeze(-1)], dim=-1)
        else:
            # No image connected: return black RGBA with the soft mask as alpha
            image_out = torch.zeros((*soft_mask.shape, 4), dtype=torch.float32, device=mask.device)
            image_out[..., 3] = soft_mask

        return (soft_mask, image_out)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float, device) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32, device=device)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = g.outer(g)
        kernel /= kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "MaskFeather": MaskFeather,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFeather": "Mask Feather",
}
