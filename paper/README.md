# CVPR 2026 Paper: Image Semantic Segmentation on MSRC V2 Dataset

## Overview

This directory contains the LaTeX source files for the research paper titled:

**"Image Semantic Segmentation on MSRC V2 Dataset: A Comparative Study from Traditional Methods to Deep Learning"**

## Paper Structure

The paper presents a comprehensive comparative study of image semantic segmentation methods:

### 1. Traditional Methods
- Superpixel-based segmentation using SLIC algorithm
- Feature extraction (16-dimensional: color, texture, spatial features)
- Classification with Random Forest and Gaussian Mixture Models
- Markov Random Field (MRF) optimization for spatial smoothing

### 2. Deep Learning Methods
- U-Net: Encoder-decoder architecture with skip connections
- DeepLabV3: ResNet-50 backbone with Atrous Spatial Pyramid Pooling (ASPP)

### 3. Key Results

| Method | Pixel Accuracy | mIoU |
|--------|---------------|------|
| RF + MRF | 75.48% | 49.84% |
| GMM + MRF | 71.89% | 43.21% |
| U-Net | 86.53% | 61.47% |
| DeepLabV3 | **89.71%** | **67.62%** |

## Files

- `main.tex` - Main LaTeX document
- `cvpr2026_style.sty` - CVPR 2026 style package
- `README.md` - This file

## Compilation

To compile the paper:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

Or use a LaTeX editor such as Overleaf, TeXShop, or VSCode with LaTeX Workshop.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{author2026imageseg,
  title={Image Semantic Segmentation on MSRC V2 Dataset: A Comparative Study from Traditional Methods to Deep Learning},
  author={Author Name},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Related Materials

- The experiments are documented in `final.ipynb` in the parent directory
- Training logs are saved in JSON format for reproducibility
