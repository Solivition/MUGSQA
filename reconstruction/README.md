# `reconstruction/` — Gaussian-Splatting method patches

This directory contains **patch overlays** for upstream Gaussian-Splatting repositories. Each method subfolder only keeps the files that differ from upstream. Apply a patch by cloning the upstream repo, copying the matching subfolder over it, and copying [`train_all.py`](train_all.py) alongside.

The full workflow (smoke test, environment setup, sweep command) is documented in the [main README's Dataset Construction → Stage 2 section](../README.md#dataset-construction).

## Methods

| `--method` tag | Patch folder | Upstream repo | `--train_script` |
|---|---|---|---|
| `3dgs`          | [gaussian-splatting/](gaussian-splatting/) | https://github.com/graphdeco-inria/gaussian-splatting | `train.py` |
| `mip-splatting` | [mip-splatting/](mip-splatting/)           | https://github.com/autonomousvision/mip-splatting     | `train.py` |
| `lgs`           | [lightgaussian/](lightgaussian/)           | https://github.com/VITA-Group/LightGaussian           | `train_densify_prune.py` |
| `scaffold-gs`   | [scaffold-gs/](scaffold-gs/)               | https://github.com/city-super/Scaffold-GS             | `train.py` |
| `octree-gs`     | [octree-gs/](octree-gs/)                   | https://github.com/city-super/Octree-GS               | `train.py` |
| `efficient`     | [efficientgaussian/](efficientgaussian/)   | https://github.com/Sharath-girish/efficientgaussian   | `train_eval.py` |

## License notice

Files under each method subfolder are **derivative works of their respective upstream repositories** and inherit each upstream's license. Many files carry headers such as `# Copyright (C) 2023, Inria ... under the terms of the LICENSE.md file.` — the `LICENSE.md` they refer to is the one shipped by the upstream repo. When you apply a patch by copying our files over a fresh clone of the upstream repo, that upstream `LICENSE.md` is what governs the resulting tree.

Most of these upstreams (3DGS, Mip-Splatting, Scaffold-GS, Octree-GS, EfficientGaussian) ship a **non-commercial research-use** license. Please review each upstream's `LICENSE.md` before redistributing or using the patched code in a commercial setting.

Only [`train_all.py`](train_all.py) in this directory is a new file authored by the MUGSQA project; it is released under the repository's [MIT License](../LICENSE) along with the rest of the MUGSQA toolkit code.
