# Spike Imaging Velocimetry

Official implementation of **Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Streams**, accepted by **AAAI 2026**.

This repository provides the code, pretrained models, and the **Particle Scenes with Spike and Displacement (PSSD)** dataset for spike-based particle image velocimetry.

> **SIV** explores the use of spike cameras for high-speed fluid velocity measurement and provides a dedicated deep learning framework for dense motion estimation from spike streams.

---

## Highlights

- **Spike-based PIV framework.**  
  We investigate spike cameras as high-temporal-resolution and high-dynamic-range sensors for particle image velocimetry.

- **SIV network.**  
  We propose a specialized architecture for fluid velocity estimation from spike streams.

- **Three task-specific modules.**
  - **DPHT**: Detail-Preserving Hierarchical Transform for spike stream representation.
  - **GE**: Graph Encoder for adaptive contextual aggregation in turbulent flows.
  - **MSVR**: Multi-scale Velocity Refinement for recovering fine-scale vortical structures.

- **PSSD dataset.**  
  We introduce **Particle Scenes with Spike and Displacement**, a spike-based PIV dataset with labeled displacement fields.

- **Strong performance.**  
  SIV achieves state-of-the-art performance across multiple fluid motion estimation settings.

---

## News

- **[2026]** Code and PSSD dataset are publicly released.
- **[2026]** SIV is accepted by AAAI 2026.

---

## Method Overview

SIV estimates dense fluid velocity fields from spike streams captured by spike cameras.

Given a spike stream pair, SIV first extracts detail-preserving spike representations through **DPHT**. Then, the **Graph Encoder** captures fluid-specific contextual information by modeling feature interactions in a graph space. Finally, the **Multi-scale Velocity Refinement** module reconstructs the full-resolution velocity field and improves the recovery of small-scale structures such as shear layers and vortices.

The overall pipeline is designed for the challenges of fluid motion estimation, including high-speed motion, sparse particle observations, unstructured turbulent patterns, and high-dynamic-range imaging conditions.

---

## PSSD Dataset

We release **PSSD: Particle Scenes with Spike and Displacement**, a benchmark dataset for spike-based particle image velocimetry.

PSSD contains synthetic spike streams, corresponding image pairs, and ground-truth displacement fields. It covers multiple representative fluid dynamics scenarios:

1. **Steady Turbulence**
2. **High-speed Flow**
3. **High Dynamic Range Scenes**

The dataset includes four flow types:

- Channel
- Isotropic
- MHD
- Mixing

Each sample contains spike streams and corresponding dense displacement labels, enabling supervised training and evaluation of spike-based PIV and optical-flow-style methods.

Dataset download:

```text
Link: https://pan.quark.cn/s/a79259cd081a
Extraction code: xNgh
