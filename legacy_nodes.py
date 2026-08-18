"""Legacy isolated upstream runner; intentionally not registered by the package."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SPHEROPE_PYTHON = Path(os.environ.get("SPHEROPE_PYTHON", sys.executable))
SPHEROPE_SCRIPT = Path(
    os.environ.get(
        "SPHEROPE_SCRIPT",
        Path.home() / "SpheRoPE" / "generate_panorama.py",
    )
)
AVAILABLE_GPUS = [
    item.strip()
    for item in os.environ.get("SPHEROPE_GPU_IDS", "0").split(",")
    if item.strip()
]


class SpheRoPEGeneratePanorama:
    """Run SpheRoPE in its isolated Conda environment and return a ComfyUI image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A panoramic mountain landscape at golden hour"}),
                "model": (["flux1", "flux2"], {"default": "flux1"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "erp_gamma": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "offload": (["sequential", "model"], {"default": "sequential"}),
                "gpu_id": (AVAILABLE_GPUS, {"default": AVAILABLE_GPUS[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("panorama",)
    FUNCTION = "generate"
    CATEGORY = "SpheRoPE"
    DESCRIPTION = "Generate a seamless 2:1 equirectangular panorama with SpheRoPE."

    def generate(self, prompt, model, width, height, steps, guidance_scale, erp_gamma, seed, offload, gpu_id):
        if width != height * 2:
            raise ValueError(f"SpheRoPE requires a 2:1 ERP image; got {width}x{height}.")
        if not SPHEROPE_PYTHON.is_file() or not SPHEROPE_SCRIPT.is_file():
            raise RuntimeError(
                "SpheRoPE subprocess runtime is not configured. Set "
                "SPHEROPE_PYTHON and SPHEROPE_SCRIPT before starting ComfyUI."
            )

        fd, output_name = tempfile.mkstemp(prefix="spherope_", suffix=".png")
        os.close(fd)
        output_path = Path(output_name)
        command = [
            str(SPHEROPE_PYTHON),
            str(SPHEROPE_SCRIPT),
            "--model", model,
            "--prompt", prompt,
            "--output", str(output_path),
            "--width", str(width),
            "--height", str(height),
            "--num-inference-steps", str(steps),
            "--guidance-scale", str(guidance_scale),
            "--erp-gamma", str(erp_gamma),
            "--seed", str(seed),
            "--offload", offload,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        try:
            result = subprocess.run(
                command,
                cwd=str(SPHEROPE_SCRIPT.parent),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"SpheRoPE failed (exit {result.returncode}):\n{result.stdout[-12000:]}")
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"SpheRoPE produced no image. Output:\n{result.stdout[-12000:]}")

            with Image.open(output_path) as image:
                rgb = image.convert("RGB")
                array = np.asarray(rgb, dtype=np.float32) / 255.0
            return (torch.from_numpy(array).unsqueeze(0),)
        finally:
            output_path.unlink(missing_ok=True)


NODE_CLASS_MAPPINGS = {
    "SpheRoPEGeneratePanorama": SpheRoPEGeneratePanorama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpheRoPEGeneratePanorama": "SpheRoPE Generate Panorama",
}
