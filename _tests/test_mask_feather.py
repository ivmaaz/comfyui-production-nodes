"""Test for the Mask Feather node. Usage: python _tests/test_mask_feather.py [image_path]"""
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "output")


def load_package():
    # Load the package through __init__.py, the same way ComfyUI does
    spec = importlib.util.spec_from_file_location("comfyui_production_nodes", os.path.join(PACKAGE_DIR, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_image(path):
    if path is None:
        # Synthetic 1200x800 gradient when no sample image is given
        y, x = np.mgrid[0:800, 0:1200].astype(np.float32)
        return torch.from_numpy(np.stack([x / 1199, y / 799, np.full_like(x, 0.5)], -1))[None]
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None]  # (1, H, W, 3)


def main():
    pkg = load_package()
    node = pkg.NODE_CLASS_MAPPINGS["MaskFeather"]()
    image = load_image(sys.argv[1] if len(sys.argv) > 1 else None)
    _, H, W, _ = image.shape
    print("input", tuple(image.shape))

    # Hard-edged rectangle mask in the middle of the image
    mask = torch.zeros(1, H, W)
    mask[:, H // 4:3 * H // 4, W // 4:3 * W // 4] = 1.0

    failures = 0

    def check(name, cond):
        nonlocal failures
        failures += not cond
        print(("OK   " if cond else "FAIL ") + name)

    soft, rgba = node.feather(mask, radius=10, expand=4, image=image)
    check(f"mask shape {tuple(soft.shape)}", soft.shape == (1, H, W))
    check(f"image shape {tuple(rgba.shape)}", rgba.shape == (1, H, W, 4))
    check("values in [0,1]", soft.min() >= 0 and soft.max() <= 1)
    edge = soft[0, H // 4, W // 2].item()
    check(f"edge is soft ({edge:.3f})", 0.05 < edge < 0.999)
    check("alpha == soft mask", torch.equal(rgba[..., 3], soft))
    check("RGB unchanged", torch.equal(rgba[..., :3], image))

    soft_e, _ = node.feather(mask, radius=1, expand=-5)
    check("negative expand shrinks mask", soft_e.sum() < mask.sum())

    _, rgba_noimg = node.feather(mask, radius=5, expand=0)
    check("no image -> black RGBA", rgba_noimg.shape == (1, H, W, 4) and rgba_noimg[..., :3].abs().sum() == 0)

    rgba_in = torch.cat([image, torch.ones(1, H, W, 1)], dim=-1)
    _, rgba2 = node.feather(mask, radius=5, expand=0, image=rgba_in)
    check("RGBA input keeps 4 channels", rgba2.shape[-1] == 4)

    # Edge darkening fix: a mask touching the image border must stay solid there
    full = torch.ones(1, 64, 64)
    soft_full, _ = node.feather(full, radius=10, expand=0)
    check(f"full-white mask corner stays 1.0 ({soft_full[0, 0, 0].item():.3f})", torch.allclose(soft_full, full))
    half = torch.zeros(1, 64, 64)
    half[:, :, :32] = 1.0
    soft_half, _ = node.feather(half, radius=10, expand=0)
    check(f"mask touching left border stays solid there ({soft_half[0, 0, 0].item():.3f})", soft_half[0, :, 0].min() > 0.999)

    # Batch-size fix: mismatched mask and image batches are broadcast instead of crashing
    s, r = node.feather(mask, radius=5, expand=0, image=image.repeat(3, 1, 1, 1))
    check(f"mask batch 1 + image batch 3 -> {tuple(s.shape)}, {tuple(r.shape)}", s.shape == (3, H, W) and r.shape == (3, H, W, 4))
    s, r = node.feather(mask.repeat(2, 1, 1), radius=5, expand=0, image=image)
    check(f"mask batch 2 + image batch 1 -> {tuple(s.shape)}, {tuple(r.shape)}", s.shape == (2, H, W) and r.shape == (2, H, W, 4))
    s, _ = node.feather(mask[0], radius=5, expand=0)
    check(f"2D mask (H, W) -> {tuple(s.shape)}", s.shape == (1, H, W))

    if torch.cuda.is_available():
        s, r = node.feather(mask.cuda(), radius=10, expand=4, image=image.cuda())
        check(f"cuda run on {s.device}", s.is_cuda and r.is_cuda)

    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray((soft[0].numpy() * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "feather_mask.png"))
    Image.fromarray((rgba[0].numpy() * 255).astype(np.uint8), "RGBA").save(os.path.join(OUT_DIR, "feather_rgba.png"))

    print("PASSED" if failures == 0 else f"FAILED ({failures} checks)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
