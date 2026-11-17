from .aspect_ratio_node import AspectRatioConverter
from .mask_feather_node import MaskFeather

NODE_CLASS_MAPPINGS = {
    "AspectRatioConverter": AspectRatioConverter,
    "MaskFeather": MaskFeather,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectRatioConverter": "Aspect Ratio Converter",
    "MaskFeather": "Mask Feather",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
