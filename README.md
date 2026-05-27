# Spike Imaging Velocimetry

Official repository for **Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Streams**, accepted by **AAAI 2026**.

This repository provides the official resources for **SIV** and the **PSSD** dataset. SIV explores spike cameras for high-speed fluid velocity measurement and introduces a dedicated learning framework for dense motion estimation from spike streams.

- **Paper:** *Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Streams*
- **Conference:** AAAI 2026
- **Dataset:** PSSD: Particle Scenes with Spike and Displacement
- **Task:** Spike-based particle image velocimetry / dense fluid motion estimation

---

## News

- **2026-01:** SIV is accepted by **AAAI 2026**.
- **2026-01:** The official SIV repository and PSSD dataset are publicly released.

---

## Overview

Particle Image Velocimetry (PIV) is a widely used non-invasive technique for estimating fluid velocity fields from particle image sequences. However, conventional frame-based imaging can be limited in high-speed, high-dynamic-range, and particle-sparse fluid scenes.

SIV investigates the use of **spike cameras** for PIV. Spike cameras provide ultra-high temporal resolution and high dynamic range, making them suitable for capturing fast and complex fluid motion. Given spike streams captured at two time moments, SIV estimates dense displacement / velocity fields of particles in the fluid.

The framework is designed for challenging fluid scenarios, including:

- high-speed fluid motion,
- sparse tracer particle observations,
- unstructured turbulent patterns,
- small-scale vortices and shear layers,
- high-dynamic-range imaging conditions.

---

## Highlights

- **Spike-based PIV framework**  
  We formulate dense fluid velocity estimation from spike streams and study the potential of spike cameras for high-speed PIV.

- **SIV network**  
  We propose a dedicated architecture for dense motion estimation of fluids using spike streams.

- **Detail-Preserving Hierarchical Transform (DPHT)**  
  A spike representation module that preserves particle details during hierarchical feature extraction.

- **Graph Encoder (GE)**  
  A graph-based context encoder that models adaptive feature interactions in turbulent and unstructured flow fields.

- **Multi-scale Velocity Refinement (MSVR)**  
  A refinement module that improves full-resolution velocity reconstruction, especially for fine-scale vortical structures.

- **PSSD dataset**  
  We release a spike-based PIV dataset with spike streams, image pairs, and dense ground-truth displacement fields.

---

## Method

SIV estimates dense velocity fields from spike streams through three major components:

1. **DPHT** extracts detail-preserving spike representations with a hierarchical 3D-2D feature transformation.
2. **GE** projects features into a graph space and performs adaptive context aggregation for turbulent flow structures.
3. **MSVR** refines multi-scale velocity predictions and reconstructs full-resolution flow fields.

This design aims to better handle the characteristics of fluid motion and spike streams, rather than directly applying general-purpose optical flow networks to PIV data.

---

## PSSD Dataset

We release **PSSD: Particle Scenes with Spike and Displacement**, a benchmark dataset for spike-based particle image velocimetry.

PSSD contains:

- synthetic spike streams,
- corresponding image pairs,
- dense ground-truth displacement fields.

The dataset covers three representative fluid motion estimation scenarios:

1. **Steady Turbulence**
2. **High-speed Flow**
3. **High Dynamic Range Scenes**

It also includes four flow types:

- Channel
- Isotropic
- MHD
- Mixing

Each sample provides spike-based observations and dense displacement labels, supporting supervised training and evaluation of spike-based PIV and optical-flow-style methods.

### Dataset Download

The PSSD dataset can be downloaded from Quark Cloud:

```text
Link: https://pan.quark.cn/s/a79259cd081a
Extraction code: xNgh
```

After downloading, please organize the dataset according to the instructions in the code/configuration files.

---

## Repository Structure

The repository is organized for training and evaluating SIV on the PSSD dataset. The main components include model definitions, dataset loading utilities, configuration files, and training / evaluation scripts.

A typical structure is:

```text
SIV/
├── configs/          # Configuration files
├── datasets/         # PSSD dataset loaders
├── models/           # SIV network and modules
├── scripts/          # Training and evaluation scripts
├── assets/           # Figures and visual examples
└── README.md
```

The exact structure may be updated as the repository is maintained.

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/YunzZhang/SIV.git
cd SIV
```

Prepare the environment and install the required dependencies according to the provided configuration files.

Then download PSSD and set the dataset path in the corresponding config file before training or evaluation.

Example commands:

```bash
# Training
python train.py --config configs/siv_pssd.yaml

# Evaluation
python test.py --config configs/siv_pssd.yaml --checkpoint checkpoints/siv_pssd.pth
```

Please adjust the command names and paths according to the released code structure.

---

## Results

SIV achieves strong performance on the PSSD benchmark and outperforms representative image-based PIV methods, spike-based optical flow methods, and adapted optical flow baselines.

The model is particularly effective in challenging fluid regions, including high-speed flows, shear layers, and small-scale vortical structures.

Detailed quantitative results, visual comparisons, and ablation studies are provided in the paper.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zhang2026spike,
  title     = {Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Streams},
  author    = {Zhang, Yunzhong and Zhou, You and Su, Changqing and Cheng, Zhen and Yu, Zhaofei and Xiong, Bo and Huang, Tiejun and Cao, Xun},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

---

## Acknowledgements

This work was supported by the Postgraduate Research \& Practice Innovation Program of Jiangsu Province, the National Natural Science Foundation of China, the Jiangsu Association for Science and Technology Young Elite Scientists Sponsorship Program, and the Beijing Natural Science Foundation.

---

## Contact

For questions about the code, dataset, or paper, please contact:

```text
Yunzhong Zhang
Email: ltq@smail.nju.edu.cn
GitHub: https://github.com/YunzZhang
```
