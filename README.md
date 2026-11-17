# ComfyUI Production Nodes

Two small, dependency-free ComfyUI nodes for everyday production work.

## Nodes

| Node | Category | What it does |
|---|---|---|
| **Aspect Ratio Converter** | `image/transform` | Center-crops or pads (black, white or gray) an image batch to 16:9, 1:1 or 9:16, so one master image can be turned into landscape, square and story/reel formats for different social platforms in a single workflow. |
| **Mask Feather** | `image/segmentation` | Grows or shrinks a mask and blurs its edges, returning the soft mask plus an RGBA cut-out, so subjects composite onto new backgrounds without hard, jagged outlines. |

### Aspect Ratio Converter

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | Image batch to convert |
| `aspect_ratio` | `16:9` / `1:1` / `9:16` | Target aspect ratio |
| `mode` | `crop` / `pad` | `crop` cuts from the center, `pad` adds letterbox/pillarbox bars |
| `pad_color` | `black` / `white` / `gray` | Bar color for `pad` mode (optional) |

Output: `image` at the new aspect ratio. RGBA images keep their alpha channel; padded areas are transparent.

### Mask Feather

| Input | Type | Description |
|---|---|---|
| `mask` | MASK | Mask to soften |
| `radius` | INT 1–50 | Blur radius in pixels |
| `expand` | INT -20–20 | Grow (positive) or shrink (negative) the mask before blurring |
| `image` | IMAGE | Optional RGB or RGBA image to apply the soft mask to |

Outputs: `mask` (soft mask) and `image` (RGBA with the soft mask as alpha; black RGBA if no image is connected). If the mask and image batch sizes differ, the shorter batch is repeated.

## Installation

Clone into your ComfyUI `custom_nodes` folder and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/comfyui-production-nodes.git
```

No extra dependencies are needed; the nodes only use PyTorch, which ComfyUI already includes.

## Usage example

Social formats from one image:

```
Load Image ─┬─> Aspect Ratio Converter (16:9, crop) ──> Save Image
            ├─> Aspect Ratio Converter (1:1,  crop) ──> Save Image
            └─> Aspect Ratio Converter (9:16, pad, black) ──> Save Image
```

Soft cut-out for compositing:

```
Load Image ──> (any segmentation / background removal node) ──MASK──> Mask Feather (radius 6, expand 2) ──IMAGE──> Save Image
     └───────────────────────────────────────────────────────IMAGE──┘
```

Feed Mask Feather's `mask` output into a compositing node (for example the core **ImageCompositeMasked**) to place the subject on a new background.

## Tests

```bash
python _tests/test_aspect_ratio.py [image_path]
python _tests/test_mask_feather.py [image_path]
```

Without an image path the tests use a generated gradient. Output images are written to `_tests/output/`. With the Windows portable build, run them with `python_embeded\python.exe`.

## License

[MIT](LICENSE)
