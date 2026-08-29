[README.md](https://github.com/user-attachments/files/31589080/README.md)
# GS-in-GS: Lossless High-Capacity Invisible Watermarking for 3D Gaussian Splatting

Official implementation of **GS-in-GS**, a novel framework that embeds an entire 3D Gaussian Splatting (3DGS) model (the *watermark scene*) into another 3DGS model (the *carrier scene*) in a lossless and invisible manner.

## Core Idea

The key insight is that **low-contribution Gaussians** in a trained 3DGS model contribute negligibly to rendering quality. GS-in-GS identifies these low-contribution Gaussians via opacity-based contribution analysis, then **replaces their properties** (color, rotation, scale) with the corresponding properties of the watermark 3DGS model. The result is a single watermarked model that renders identically to the original carrier scene while embedding a complete, recoverable watermark scene.

### Why "GS-in-GS"?

- **Lossless**: The watermark model can be recovered with **100% accuracy** (exact property values).
- **High-Capacity**: An entire 3DGS model (tens of thousands of Gaussians) can be embedded.
- **Invisible**: The watermarked model is visually indistinguishable from the original carrier scene.

## Pipeline Overview

```
1. Contribution Analysis  → 2. Gaussian Replacement  → 3. Model Merging  → 4. Watermark Extraction
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `new_contribution_analysis2.py` | Compute opacity^α contribution for each Gaussian; identify the lowest-contribution Gaussians |
| 2 | `replace.py` / `replace_*.py` | Select target Gaussians and replace their properties with watermark model properties |
| 3 | `embed.py` / `embed_*.py` | Merge modified low-contribution Gaussians with high-contribution Gaussians |
| 4 | `extract.py` | Reconstruct the watermark model from the watermarked carrier model |
| Eval | `duibi.py` | Compare extracted watermark with original to measure accuracy |
| Eval | `quality_test.py` | Evaluate visual quality (SSIM, PSNR, LPIPS) |
| Eval | `Robustness_testing.py` | Test robustness under attacks (noise, rotation, translation, crop) |

## Project Structure

```
GS-in-GS/
├── watermarking/                  # Core watermarking algorithms
│   ├── new_contribution_analysis2.py   # Contribution analysis (final version)
│   ├── replace.py / replace_*.py       # Gaussian replacement (core + ablation)
│   ├── embed.py / embed_*.py           # Embedding merge (core + ablation)
│   ├── extract.py                      # Watermark extraction
│   ├── duibi.py                        # Model comparison / accuracy
│   ├── quality_test.py                 # Quality evaluation (SSIM/PSNR/LPIPS)
│   ├── Robustness_testing.py           # Robustness testing
│   ├── custom_render.py                # Custom rendering
│   ├── modify_ggbond_properties.py     # Create zero-property watermark models
│   └── Remove_redundancies.py          # Remove unselected low-contribution Gaussians
├── scene/                         # 3DGS scene module
├── utils/                         # 3DGS utility functions
├── arguments/                     # Argument configuration
├── gaussian_renderer/             # 3DGS renderer
├── diff-gaussian-rasterization/   # CUDA rasterizer
├── simple_knn/                    # KNN module for densification
├── .gitignore
└── README.md
```

### Ablation Study Variants

The `embed_*.py` and `replace_*.py` variants support ablation experiments with different property combinations:

| Suffix | Properties Replaced |
|--------|-------------------|
| `_c` | Color only |
| `_cr` | Color + Rotation |
| `_cs` | Color + Scale |
| `_csr` | Color + Scale + Rotation |
| `_cα` | Color + Opacity |
| `_cαr` | Color + Opacity + Rotation |
| `_cαs` | Color + Opacity + Scale |
| `_cαsr` | Color + Opacity + Scale + Rotation (all) |
| `_α` | Opacity only |
| `_αr` | Opacity + Rotation |
| `_αs` | Opacity + Scale |
| `_αsr` | Opacity + Scale + Rotation |
| `_s` | Scale only |
| `_sr` | Scale + Rotation |

## Environment Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.6+
- Required packages: `plyfile`, `numpy`, `matplotlib`, `scikit-image`, `lpips`, `open3d`

## Usage

### 1. Contribution Analysis

Identify low-contribution Gaussians in the carrier scene:

```bash
python watermarking/new_contribution_analysis2.py
```

### 2. Gaussian Replacement

Replace properties of low-contribution Gaussians with watermark model properties:

```bash
# Basic replacement (rotation only)
python watermarking/replace.py

# Full property replacement (color + opacity + scale + rotation)
python watermarking/replace_cαsr.py
```

### 3. Embedding & Merging

Merge the modified Gaussians to create the watermarked model:

```bash
# Basic embedding (rotation only)
python watermarking/embed.py

# Full property embedding
python watermarking/embed_cαsr.py
```

### 4. Watermark Extraction

Extract the watermark model from the watermarked carrier:

```bash
python watermarking/extract.py
```

### 5. Evaluation

**Accuracy evaluation:**
```bash
python watermarking/duibi.py
```

**Quality evaluation (SSIM/PSNR/LPIPS):**
```bash
python watermarking/quality_test.py
```

**Robustness evaluation:**
```bash
python watermarking/Robustness_testing.py
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2025gsings,
  title={GS-in-GS: Lossless High-Capacity Invisible Watermarking for 3D Gaussian Splatting},
  author={Wang, Hao},
  journal={Journal of Electronic Testing},
  year={2025}
}
```

## License

This project is released for research purposes. See [LICENSE.md](diff-gaussian-rasterization/LICENSE.md) for the 3DGS rasterizer license.
