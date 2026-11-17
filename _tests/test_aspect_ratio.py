"""Test for the Aspect Ratio Converter node. Usage: python _tests/test_aspect_ratio.py [image_path]"""
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


def save_image(t, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    arr = (t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT_DIR, name))


def main():
    pkg = load_package()
    cls = pkg.NODE_CLASS_MAPPINGS["AspectRatioConverter"]
    node = cls()
    image = load_image(sys.argv[1] if len(sys.argv) > 1 else None)
    print("input", tuple(image.shape))

    failures = 0
    cases = [(image, "sample")]
    # Synthetic sizes, including ones that already match a target ratio exactly
    for h, w in [(1080, 1920), (1920, 1080), (512, 512), (777, 1001)]:
        cases.append((torch.rand(2, h, w, 3), f"{w}x{h}"))
    cases.append((torch.rand(1, 600, 400, 4), "400x600 RGBA"))

    for img, label in cases:
        for ratio in cls.ASPECT_RATIOS:
            for mode in cls.MODES:
                (out,) = node.convert(img, ratio, mode, "gray")
                tw, th = cls.ASPECT_RATIOS[ratio]
                _, H, W, C = out.shape
                err = abs(W / H - tw / th)
                ok = out.shape[0] == img.shape[0] and C == img.shape[3] and err < 0.01
                if mode == "crop":
                    ok = ok and H <= img.shape[1] and W <= img.shape[2]
                else:
                    ok = ok and H >= img.shape[1] and W >= img.shape[2]
                failures += not ok
                print(f"{'OK  ' if ok else 'FAIL'} {label:12s} {ratio:5s} {mode:4s} -> {W}x{H}")
                if label == "sample":
                    save_image(out, f"aspect_{ratio.replace(':', 'x')}_{mode}.png")

    if torch.cuda.is_available():
        (out,) = node.convert(image.cuda(), "1:1", "pad", "white")
        print("OK   cuda pad ->", tuple(out.shape), out.device)

    print("PASSED" if failures == 0 else f"FAILED ({failures} cases)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
