# MUGSQA (Multi-Uncertainty-Based Gaussian Splatting Quality Assessment) Dataset

<a href="https://arxiv.org/abs/2511.06830"><img src="https://img.shields.io/badge/Paper-b5212f.svg?logo=arxiv" alt="arXiv" style="vertical-align: middle;"></a>
<a href="https://huggingface.co/datasets/Solivition/MUGSQA"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-d96902.svg" alt="huggingface" style="vertical-align: middle;"></a>
👈Find our dataset here

<p align="left">
    <img src="media\MUGSQA_pipeline.png" alt="MUGSQA" style="width:100%; max-width:1200px;">
</p>

Official repository for the <strong>ICASSP 2026</strong> Paper <strong>MUGSQA: Novel Multi-Uncertainty-Based Gaussian Splatting Quality Assessment Method, Dataset, and Benchmarks</strong>. 

## 📊Benchmark

The benchmark code and instructions will be released in this repository soon.


## 📃Dataset Summary

**MUGSQA** is a large-scale dataset designed for **Gaussian Splatting Quality Assessment (GSQA)**. It is constructed by introducing multiple uncertainties during the reconstruction process and collecting large-scale subjective quality scores.

The dataset contains **2,414 reconstructed Gaussian models**, each paired with rendered videos and Mean Opinion Scores (MOS). It supports research on:

* Gaussian Splatting quality assessment
* reconstruction robustness evaluation
* rendering-based and rendering-free quality metrics

The dataset simulates several uncertainties that commonly occur during reconstruction, including:

* input view resolution
* number of input views
* view-to-object distance
* initialization of the point cloud
* method of GS reconstruction

These factors create diverse reconstruction distortions that are useful for benchmarking reconstruction methods and quality metrics. 


## 📁Dataset Structure

The dataset repository contains the following files:

```
reference/
main.tar.gz
additional.tar.gz
mos_main.xlsx
mos_additional.xlsx
```

### 1. Reference Videos

```
reference/
```

This folder contains **ground-truth reference videos** rendered from the original source objects. Due to copyright restrictions, the original 3D mesh models are not included. Only the rendered reference videos are provided. These videos are used in the subjective quality assessment experiments as reference stimuli.

### 2. Main Set
```
main.tar.gz
```

After extraction:

```
main/
├── sample_folder_1/
├── sample_folder_2/
...
└── sample_folder_1970/
```

The **main set contains 1,970 reconstructed Gaussian objects**. Each sample folder represents one distorted reconstruction generated under specific uncertainty settings.

#### 2.1. Naming Convention

Each sample folder follows the format:

```
modelname_resolution_views_distance_method_pointcloud
```

Example:

```
12th-c-ce-water-moon-guanyin_480res_9views_5distance_lgs_rndpc
```

Where:
| Field      | Description               |
| ---------- | ------------------------- |
| modelname  | name of the source object |
| resolution | input view resolution     |
| views      | number of input views     |
| distance   | view-to-object distance   |
| method     | reconstruction method     |
| pointcloud | initial point cloud type  |

These parameters correspond to the reconstruction uncertainty settings used during dataset generation.

#### 2.2. Files inside each sample folder

Each distorted sample folder contains two files with the same name:

```
sample_name.mp4
sample_name.ply
```

| File   | Description                                         |
| ------ | --------------------------------------------------- |
| `.mp4` | rendered video of the reconstructed Gaussian object |
| `.ply` | reconstructed 3D Gaussian model                     |

### 3. Additional Set

```
additional.tar.gz
```

After extraction:

```
additional/
├── sample_folder_1/
...
└── sample_folder_444/
```

The **additional set contains 444 reconstructed samples**. Unlike the main set, the additional set includes reconstructions generated using multiple Gaussian Splatting methods:

* 3DGS
* Mip-Splatting
* Scaffold-GS
* EAGLES
* Octree-GS

All other settings are consistent with the main set.

### 4. MOS Annotations

The subjective quality scores are stored in:

```
mos_main.xlsx
mos_additional.xlsx
```

Each entry corresponds to one distorted sample and its **Mean Opinion Score (MOS)**. MOS values represent perceptual quality collected through a large-scale subjective study. Higher MOS indicates better perceived quality, and the MOS range is 0 to 5.


## 🧰Usage Example

Example workflow:

1. Extract the dataset archives:

```
tar -xzf main.tar.gz
tar -xzf additional.tar.gz
```

2. Load reconstructed models:

```
sample_folder/
├── sample_name.mp4
└── sample_name.ply
```

3. Use the `.ply` files for rendering-free quality assessment or use the `.mp4` files for rendering-based quality assessment.

4. Match the sample name with the corresponding MOS score in the Excel files.


## 🔗Citation
```latex
@inproceedings{chen2026mugsqa,
  title={{MUGSQA: Novel Multi-Uncertainty-Based Gaussian Splatting Quality Assessment Method, Dataset, and Benchmarks}}, 
  author={Tianang Chen and Jian Jin and Shilv Cai and Zhuangzi Li and Weisi Lin},
  booktitle={Proc. ICASSP},
  year={2026}
}
```
