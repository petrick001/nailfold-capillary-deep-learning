# Nailfold Capillary Deep Learning

A simplified biomedical deep learning implementation inspired by:

> Bharathi et al. — *A Deep Learning System for Quantitative Assessment of Microvascular Abnormalities in Nailfold Capillary Images*.

This project implements a student-level reproduction of a biomedical image analysis workflow for nailfold capillaroscopy using deep learning segmentation, quantitative measurement extraction, and density estimation.

---

# Overview

Nailfold capillaroscopy is a non-invasive imaging technique used to analyze small blood vessels near the fingernail. Abnormal capillary morphology may indicate diseases such as systemic sclerosis (SSc) and Raynaud’s phenomenon.

This project focuses on:
- vessel segmentation using U-Net,
- quantitative feature extraction,
- capillary density estimation,
- biomedical image processing workflows.

The implementation was developed as part of the Master 1 VIBOT program at Université Bourgogne Europe.

---

# Features

- U-Net biomedical image segmentation
- Binary vessel mask prediction
- Overlay generation
- Capillary density estimation
- Morphological measurement extraction
- Dice score and IoU evaluation
- CSV result export
- Reproducible biomedical AI workflow

---

# Dataset

The project uses a professor-provided dataset containing:

## Segmentation Dataset
- Native nailfold images
- Corresponding binary masks (ground truth)

## Density Dataset
- Native nailfold images
- 1 mm scale images
- Capillary-count annotations

---

# Methodology

The simplified workflow is:

```text
Input image
    ↓
Preprocessing
    ↓
U-Net Segmentation
    ↓
Binary Mask Prediction
    ↓
Post-processing
    ↓
Measurement Extraction
    ↓
Density Evaluation
```

The implementation includes:
- supervised image-mask learning,
- segmentation prediction,
- connected-component analysis,
- width estimation,
- density/count evaluation.

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- pandas
- matplotlib
- scikit-learn

---

# Deep Learning Architecture

The implementation uses a simplified U-Net architecture consisting of:

- Encoder
- Bottleneck
- Decoder
- Skip Connections

The network learns vessel structures from grayscale nailfold images resized to:

```text
256 × 256 pixels
```

---

# Evaluation Metrics

The project evaluates segmentation using:

- Dice Score
- Intersection over Union (IoU)

Density estimation is evaluated using:

- Mean Absolute Count Error (MAE)

---

# Experimental Results

The implementation successfully:
- trained a U-Net model,
- generated segmentation masks,
- created overlays,
- extracted quantitative measurements,
- evaluated capillary density.

Although segmentation performance remained limited due to dataset size and simplified architecture, the project demonstrates a complete biomedical AI implementation workflow.

---

# Project Structure

```text
nailfold-capillary-deep-learning/
│
├── nailfold_student_implementation.py
├── README.md
├── requirements_nailfold.txt
├── report.pdf
│
├── nailfold_work/
│   ├── prepared/
│   │   ├── segmentation/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── density_ground_truth.csv
│   │
│   ├── outputs/
│   │   ├── segmentation_test/
│   │   │   ├── overlays/
│   │   │   └── predicted_masks/
│   │   │
│   │   ├── density/
│   │   │   └── overlays/
│   │   │
│   │   └── unet_best.pth
│   │
│   └── raw/
│       ├── Density/
│       └── Segmentation/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/petrick001/nailfold-capillary-deep-learning.git
cd nailfold-capillary-deep-learning
```

Install dependencies:

```bash
pip install -r requirements_nailfold.txt
```

---

# Running the Project

Run the complete workflow:

```bash
python nailfold_student_implementation.py --mode all
```

This will:
- prepare the dataset,
- train the model,
- generate predictions,
- evaluate density,
- export results.

---

# Report

The IEEE-style implementation report is included in the repository and explains:
- dataset preparation,
- U-Net architecture,
- training methodology,
- experimental setup,
- quantitative evaluation,
- limitations and future improvements.

---

# Future Improvements

Possible future extensions include:
- GPU training,
- stronger data augmentation,
- ResNet34 refinement stage,
- logistic regression classification,
- MRI tumour segmentation adaptation,
- clinical-scale datasets.

---

# Author

## Theodore Petrick Reimmer
Master 1 VIBOT  
Université Bourgogne Europe  
Le Creusot, France

Email: petreimmer@gmail.com

---

# References

1. Bharathi et al., *A deep learning system for quantitative assessment of microvascular abnormalities in nailfold capillary images*, Rheumatology, 2023.

2. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI, 2015.

3. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.

---

# Academic Note

This repository contains a simplified educational implementation inspired by the referenced article. It is intended for academic learning and research demonstration purposes only and is not designed for clinical diagnosis or medical decision-making.