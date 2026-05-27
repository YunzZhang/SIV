# Spike Imaging Velocimetry

[![arXiv](https://img.shields.io/badge/arXiv-2504.18864-b31b1b.svg)](https://arxiv.org/abs/2504.18864)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/37133)

Official repository for **Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Streams**, accepted by **AAAI 2026**.

This repository provides the official implementation of **SIV** and the **PSSD** dataset for spike-based particle image velocimetry. SIV explores spike cameras for high-speed fluid velocity measurement and introduces a dedicated deep learning framework for dense motion estimation from spike streams.

- **Paper:** *Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Streams*
- **arXiv:** [https://arxiv.org/abs/2504.18864](https://arxiv.org/abs/2504.18864)
- **AAAI Proceedings:** [https://ojs.aaai.org/index.php/AAAI/article/view/37133](https://ojs.aaai.org/index.php/AAAI/article/view/37133)
- **Conference:** AAAI 2026
- **Dataset:** PSSD: Particle Scenes with Spike and Displacement
- **Task:** Spike-based particle image velocimetry / dense fluid motion estimation

---

## News

- **2026-01:** SIV is accepted by **AAAI 2026**.
- **2026-01:** The official SIV code and PSSD dataset are publicly released.

---

## Overview

Particle Image Velocimetry (PIV) is a widely used non-invasive technique for estimating fluid velocity fields from particle image sequences. However, conventional frame-based imaging can be limited in high-speed, high-dynamic-range, and particle-sparse fluid scenes.

SIV investigates the use of **spike cameras** for PIV. Spike cameras provide ultra-high temporal resolution and high dynamic range, making them suitable for capturing fast and complex fluid motion. Given spike streams captured at two time moments, SIV estimates dense displacement / velocity fields of particles in the fluid.

The framework is designed for challenging fluid scenarios, including high-speed fluid motion, sparse tracer particle observations, unstructured turbulent patterns, small-scale vortices and shear layers, and high-dynamic-range imaging conditions.

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

After downloading, please set the dataset path in the corresponding configuration file under `configs/`.

---

## Repository Structure

```text
SIV/
├── configs/
│   ├── Problem1.yml
│   ├── Problem2.yml
│   ├── Problem3.yml
│   └── yml_parser.py
├── datasets/
│   ├── dat_loader.py
│   └── ds_utils.py
├── model/
│   ├── attention.py
│   ├── corr.py
│   ├── extractor.py
│   ├── get_model.py
│   ├── path_match.py
│   ├── rep.py
│   ├── siv3.py
│   ├── sr_refine.py
│   └── utils.py
├── logger.py
├── main.py
└── README.md
```

---

## Environment

The code is implemented with PyTorch. A typical environment can be prepared as follows:

```bash
conda create -n siv python=3.8 -y
conda activate siv

pip install torch torchvision torchaudio
pip install opencv-python numpy tqdm tensorboardX easydict pyyaml h5py
```

Please adjust the PyTorch installation command according to your CUDA version.

---

## Training

The main entry is `main.py`. Training is the default mode. Do **not** add `--eval` during training.

The argument `--dt` supports two settings:

- `--dt 21`
- `--dt 11`

### Train on Problem 1, dt = 21

```bash
python main.py \
  --configs ./configs/Problem1.yml \
  --dt 21 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --model_iters 8 \
  --save_name siv_problem1_dt21
```

### Train on Problem 1, dt = 11

```bash
python main.py \
  --configs ./configs/Problem1.yml \
  --dt 11 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --model_iters 8 \
  --save_name siv_problem1_dt11
```

### Train on Problem 2

```bash
python main.py \
  --configs ./configs/Problem2.yml \
  --dt 21 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --model_iters 8 \
  --save_name siv_problem2_dt21
```

### Train on Problem 3

```bash
python main.py \
  --configs ./configs/Problem3.yml \
  --dt 21 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --model_iters 8 \
  --save_name siv_problem3_dt21
```

Short-form arguments are also supported:

```bash
python main.py \
  -c ./configs/Problem1.yml \
  -dt 21 \
  -bs 4 \
  -lr 1e-4 \
  -mit 8 \
  -sn siv_problem1_dt21
```

Training logs, TensorBoard files, visualizations, and checkpoints will be saved under `outputs/`, `vis/`, and `eval_vis/`.

A typical checkpoint path is:

```text
outputs/Problem1/dt=21/<date>/<run_name>/<model_name>_epochXXX.pth
```

---

## Evaluation

Evaluation requires two arguments:

- `--eval` or `-e`
- `--pretrained` or `-prt`, the path to a trained checkpoint

### Evaluate Problem 1, dt = 21

```bash
python main.py \
  --configs ./configs/Problem1.yml \
  --dt 21 \
  --eval \
  --pretrained ./path/to/checkpoint.pth \
  --model_iters 8 \
  --save_name eval_problem1_dt21
```

### Evaluate Problem 1, dt = 11

```bash
python main.py \
  --configs ./configs/Problem1.yml \
  --dt 11 \
  --eval \
  --pretrained ./path/to/checkpoint.pth \
  --model_iters 8 \
  --save_name eval_problem1_dt11
```

Short-form evaluation command:

```bash
python main.py \
  -c ./configs/Problem1.yml \
  -dt 21 \
  -e \
  -prt ./path/to/checkpoint.pth \
  -mit 8 \
  -sn eval_problem1_dt21
```

Evaluation results will be printed in the log file and visualized flow maps will be saved under:

```text
eval_vis/Problem<id>/dt=<dt>/
```

---

## Important Arguments

| Argument | Short | Default | Description |
| --- | --- | --- | --- |
| `--configs` | `-c` | `./configs/Problem1.yml` | Path to the configuration file. |
| `--dt` | `-dt` | `21` | Time interval. Only `21` and `11` are supported. |
| `--batch_size` | `-bs` | `4` | Training batch size. |
| `--learning_rate` | `-lr` | `1e-4` | Initial learning rate. |
| `--num_workers` | `-j` | `12` | Number of dataloader workers. |
| `--model_iters` | `-mit` | `8` | Number of iterative updates in the model. |
| `--eval` | `-e` | `False` | Enable evaluation mode. |
| `--pretrained` | `-prt` | `None` | Path to pretrained checkpoint. |
| `--save_name` | `-sn` | `None` | Suffix of the output folder. |
| `--no_warm` | `-nw` | `False` | Disable learning-rate warm-up. |
| `--warm_iters` | `-wi` | `3000` | Number of warm-up iterations. |
| `--valid_freq` | `-vf` | `10` | Validation frequency during training. |
| `--valid_vis_freq` | `-vvf` | `40` | Visualization frequency during evaluation. |
| `--mixed_precision` |  | `False` | Enable mixed precision. |

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
