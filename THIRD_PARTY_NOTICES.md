# Third-Party Notices

## SpheRoPE

This repository contains an independent ComfyUI adaptation of concepts and implementation from:

- Project: **SpheRoPE**
- Upstream repository: https://github.com/orhir/SpheRoPE
- Paper: *SpheRoPE: Zero-Shot Optimization-Free 360 Panorama Generation with Spherical RoPE*, Hirschorn et al.
- Paper URL: https://arxiv.org/abs/2606.32033
- Upstream license: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- License URL: https://creativecommons.org/licenses/by-nc/4.0/

Modifications and additions in this repository include:

- a native ComfyUI FLUX.1 spherical-frequency-coordinate model patch;
- ERP conditioning metadata and a standard-KSampler post-CFG hook;
- a combined model/VAE pipeline patch node;
- a horizontally circular VAE proxy for standard VAE Decode;
- compatibility nodes and example ComfyUI workflow wiring.

The original work has been modified. This repository is not affiliated with, sponsored by, or endorsed by the SpheRoPE authors.

The complete CC BY-NC 4.0 license text is reproduced in `LICENSE`. Model weights, ComfyUI, FLUX, and other dependencies are not relicensed by this repository and remain subject to their respective licenses.