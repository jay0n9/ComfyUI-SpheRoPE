# ComfyUI-SpheRoPE

Native ComfyUI nodes for seamless 360-degree equirectangular panorama generation with native FLUX.1 and Wan video models, adapted from [SpheRoPE](https://github.com/orhir/SpheRoPE).

The native path is designed to fit an existing ComfyUI graph: it patches a regular FLUX.1 or Wan `MODEL`, packages ERP conditioning into ordinary positive/negative `CONDITIONING`, works with the standard `KSampler`, and returns a wrapped `VAE` that can be connected to the standard `VAE Decode` node.

> Non-commercial use only. This repository follows the upstream [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Installation

Clone this repository into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jay0n9/ComfyUI-SpheRoPE.git
```

No additional Python packages are required for the native nodes beyond
ComfyUI's existing `torch`, `numpy`, and `Pillow` stack. The included
`requirements.txt` is intentionally package-free so ComfyUI Manager does not
replace the environment's CUDA-enabled PyTorch build.

## Standard KSampler workflow

1. Load a FLUX.1 or native Wan model, CLIP, and VAE as usual.
2. Connect `MODEL` and `VAE` to **SpheRoPE Pipeline Patch**.
3. Connect the patched `spherope_model` to a standard **KSampler**.
4. Use **SpheRoPE ERP Conditioning** instead of separate positive/negative CLIP text encode nodes.
5. Connect its positive and negative outputs to the standard **KSampler**.
6. Decode the sampled latent with a standard **VAE Decode**, using `circular_vae` from **SpheRoPE Pipeline Patch**.
7. Generate at a 2:1 equirectangular resolution such as 1024×512 or 2048×1024.

The ERP geometry prompt is intentionally internal and fixed:

> Single unified continuous environment, monolithic scene composition, solitary spatial layout, flawlessly stitched 360 panorama, true equirectangular projection, accurate spherical geometry, continuous horizontal wrap, zero parallax error.

`erp_gamma` controls the added ERP branch. A practical starting value is `4.0`; set `erp_enabled` to false to disable that branch without changing graph wiring. Regular KSampler CFG remains available as usual.

An importable example is included at [`examples/spherope_native_flux1_modular.json`](examples/spherope_native_flux1_modular.json).

## Public nodes

| Node | Purpose |
| --- | --- |
| SpheRoPE Pipeline Patch | Combines the native Wan/FLUX.1 spherical-frequency-coordinate model patch and circular VAE wrapper. |
| SpheRoPE ERP Conditioning | Encodes positive/negative prompts and embeds the fixed ERP branch plus `erp_gamma` metadata. |

Only these two integrated nodes are registered in ComfyUI. Legacy and internal helper implementations remain unregistered to keep the node menu and workflow surface minimal.

The native SFC model patch supports FLUX.1 (`axes_dim == [16, 56, 56]`) and native ComfyUI Wan models with three-axis RoPE. For Wan VACE, connect `ModelSampling -> SpheRoPE Pipeline Patch -> Mobius Model Patch -> KSampler`.

## Attribution and license

This project is an independent ComfyUI integration/port based on the ideas and implementation of:

- [orhir/SpheRoPE](https://github.com/orhir/SpheRoPE)
- *SpheRoPE: Zero-Shot Optimization-Free 360 Panorama Generation with Spherical RoPE*, Hirschorn et al., [arXiv:2606.32033](https://arxiv.org/abs/2606.32033)

Changes in this repository include native ComfyUI model hooks, standard-KSampler ERP conditioning, a combined model/VAE patch node, a circular VAE proxy, and ComfyUI workflow integration. This repository is not affiliated with or endorsed by the upstream authors.

The code is distributed under **Creative Commons Attribution-NonCommercial 4.0 International** to preserve the upstream terms. Model weights and other third-party components may have separate licenses; you are responsible for complying with them.

### Wan CausVid quality-safe starting values

Use `sfc_strength = 0.25` and `erp_gamma = 0.05` with CausVid CFG 1.0. Gamma values around 0.5 or above can dominate four-step Wan denoising and collapse the image into horizontal bands. Increase gamma only in small increments after an A/B test.
