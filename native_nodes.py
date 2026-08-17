# Native ComfyUI port of SpheRoPE (CC BY-NC 4.0).
import math
import torch
import torch.nn as nn

ERP_PROMPT = (
    "Single unified continuous environment, monolithic scene composition, "
    "solitary spatial layout, flawlessly stitched 360 panorama, true "
    "equirectangular projection, accurate spherical geometry, continuous "
    "horizontal wrap, zero parallax error."
)

def _sfc_matrices(h, w, device, dtype):
    base = 10000.0
    rows = torch.arange(h, dtype=torch.float64)
    cols = torch.arange(w, dtype=torch.float64)
    rg, cg = torch.meshgrid(rows, cols, indexing="ij")
    z = rg.reshape(-1)
    col = cg.reshape(-1)
    theta = rows / (h - 1) * math.pi - math.pi / 2
    phi = cols / w * 2 * math.pi - math.pi
    pg, tg = torch.meshgrid(phi, theta, indexing="xy")
    radius = w / (2 * math.pi)
    x = (torch.cos(tg) * torch.cos(pg) * radius).reshape(-1)
    y = (torch.cos(tg) * torch.sin(pg) * radius).reshape(-1)

    tf = 1 / (base ** (2 * torch.arange(8, dtype=torch.float64) / 16))
    hf = 1 / (base ** (2 * torch.arange(28, dtype=torch.float64) / 56))
    wf = 1 / (base ** (2 * torch.arange(28, dtype=torch.float64) / 56))
    ta = torch.ones(h * w, 1, dtype=torch.float64) * tf.unsqueeze(0)
    ha = z.unsqueeze(1) * hf.unsqueeze(0)

    fundamental = 2 * math.pi / w
    kval = wf / fundamental
    rounded = torch.round(kval)
    error = torch.abs(kval - rounded) / (kval + 1e-8)
    invalid = torch.where(~((kval >= 1) & (error <= 0.06)))[0]
    split = invalid[0].item() if len(invalid) else len(wf)
    linear = col.unsqueeze(1) * (rounded * fundamental).unsqueeze(0)
    spherical = torch.empty((h * w, 28), dtype=torch.float64)
    spherical[:, 0::2] = x.unsqueeze(1) * wf[0::2].unsqueeze(0)
    spherical[:, 1::2] = y.unsqueeze(1) * wf[1::2].unsqueeze(0)
    wa = torch.where((torch.arange(28) < split).unsqueeze(0), linear, spherical)

    angles = torch.cat([ta, ha, wa], dim=-1)
    cos = torch.cos(angles).to(device=device, dtype=dtype)
    sin = torch.sin(angles).to(device=device, dtype=dtype)
    return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(h * w, 64, 2, 2)

class _SFCEmbedND(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.cache = {}

    def forward(self, ids):
        pe = self.original(ids)
        if ids.ndim != 3 or ids.shape[-1] < 3:
            return pe
        h = int(ids[..., 1].max().item()) + 1
        w = int(ids[..., 2].max().item()) + 1
        count = h * w
        if h < 2 or w < 2 or count > ids.shape[1]:
            return pe
        key = (h, w, pe.device.type, pe.device.index, pe.dtype)
        sfc = self.cache.get(key)
        if sfc is None:
            sfc = _sfc_matrices(h, w, pe.device, pe.dtype)
            self.cache[key] = sfc
        out = pe.clone()
        out[:, :, -count:] = sfc[None, None].expand(ids.shape[0], -1, -1, -1, -1, -1)
        return out

def _spherope_erp_post_cfg(args):
    positive = args.get("cond")
    if not positive:
        return args["denoised"]

    settings = next(
        (item for item in positive if "spherope_erp_geometric" in item),
        None,
    )
    if settings is None:
        return args["denoised"]

    gamma = float(settings.get("spherope_erp_gamma", 0.0))
    if gamma == 0.0:
        return args["denoised"]

    geometric = settings.get("spherope_erp_processed")
    if geometric is None:
        import comfy.sampler_helpers
        import comfy.samplers

        geometric_conds = {
            "positive": comfy.sampler_helpers.convert_cond(
                settings["spherope_erp_geometric"]
            )
        }
        comfy.samplers.process_conds(
            args["model"],
            args["input"],
            geometric_conds,
            args["input"].device,
        )
        geometric = geometric_conds["positive"]
        settings["spherope_erp_processed"] = geometric

    import comfy.samplers

    (geometric_denoised,) = comfy.samplers.calc_cond_batch(
        args["model"],
        [geometric],
        args["input"],
        args["sigma"],
        args["model_options"],
    )
    # Standard KSampler already produced u + cfg*(c-u). Add gamma*(g-c).
    return args["denoised"] + gamma * (
        geometric_denoised - args["cond_denoised"]
    )


class SpheRoPESFCModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("sfc_model",)
    FUNCTION = "patch"
    CATEGORY = "SpheRoPE/native"

    def patch(self, model):
        out = model.clone()
        original = out.get_model_object("diffusion_model.pe_embedder")
        if not hasattr(original, "axes_dim") or list(original.axes_dim) != [16, 56, 56]:
            raise ValueError("SpheRoPE SFC Model Patch currently supports FLUX.1 only.")
        out.add_object_patch("diffusion_model.pe_embedder", _SFCEmbedND(original))
        out.set_model_sampler_post_cfg_function(_spherope_erp_post_cfg)
        return (out,)

class SpheRoPEEncodeERP:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "default": "A panoramic mountain landscape at golden hour"}),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
        }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "erp_geometric", "negative")
    FUNCTION = "encode"
    CATEGORY = "SpheRoPE/native"

    def encode(self, clip, prompt, negative_prompt):
        from nodes import CLIPTextEncode
        encoder = CLIPTextEncode()
        positive = encoder.encode(clip, prompt)[0]
        geometric = encoder.encode(clip, prompt + ". " + ERP_PROMPT)[0]
        negative = encoder.encode(clip, negative_prompt)[0]
        return positive, geometric, negative

class SpheRoPEERPConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "positive_prompt": ("STRING", {"multiline": True, "default": "A panoramic mountain landscape at golden hour"}),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "erp_gamma": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "erp_enabled": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "SpheRoPE/native"
    DESCRIPTION = "Encode prompts and package the anchored ERP branch and CFG settings into positive conditioning."

    def encode(self, clip, positive_prompt, negative_prompt, erp_gamma, erp_enabled):
        from nodes import CLIPTextEncode
        encoder = CLIPTextEncode()
        positive = encoder.encode(clip, positive_prompt)[0]
        negative = encoder.encode(clip, negative_prompt)[0]
        geometric = encoder.encode(
            clip, positive_prompt.rstrip(" .") + ". " + ERP_PROMPT
        )[0]

        tagged = []
        for index, item in enumerate(positive):
            conditioning, options = item
            options = options.copy()
            if index == 0:
                options["spherope_erp_geometric"] = geometric

                options["spherope_erp_gamma"] = float(erp_gamma) if erp_enabled else 0.0
                options["spherope_erp_enabled"] = bool(erp_enabled)
            tagged.append([conditioning, options])
        return tagged, negative


def _pop_spherope_settings(conditioning):
    if not conditioning:
        raise ValueError("SpheRoPE ERP Conditioning received an empty positive conditioning.")
    settings = conditioning[0][1]
    geometric = settings.get("spherope_erp_geometric")
    if geometric is None:
        raise ValueError(
            "Positive conditioning has no SpheRoPE ERP metadata. "
            "Connect the output of SpheRoPE ERP Conditioning."
        )
    guidance_scale = float(settings.get("spherope_guidance_scale", 3.5))
    erp_gamma = float(settings.get("spherope_erp_gamma", 0.0))
    clean = []
    for tensor, options in conditioning:
        clean_options = {
            key: value for key, value in options.items()
            if not key.startswith("spherope_")
        }
        clean.append([tensor, clean_options])
    return clean, geometric, guidance_scale, erp_gamma


class SpheRoPEERPGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        }}
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "build"
    CATEGORY = "SpheRoPE/native"
    DESCRIPTION = "Build 3-way ERP CFG from settings packaged by SpheRoPE ERP Conditioning."

    def build(self, model, positive, negative):
        from comfy_extras.nodes_custom_sampler import Guider_DualCFG
        positive, geometric, guidance_scale, erp_gamma = _pop_spherope_settings(positive)
        guider = Guider_DualCFG(model)
        # u + guidance_scale*(c-u) + erp_gamma*(g-c)
        guider.set_conds(geometric, positive, negative)
        guider.set_cfg(erp_gamma, guidance_scale, nested=False)
        return (guider,)


class SpheRoPESemanticCFGGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "erp_geometric": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            "erp_gamma": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
        }}
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "build"
    CATEGORY = "SpheRoPE/native"

    def build(self, model, positive, erp_geometric, negative, guidance_scale, erp_gamma):
        from comfy_extras.nodes_custom_sampler import Guider_DualCFG
        guider = Guider_DualCFG(model)
        # u + guidance_scale*(c-u) + erp_gamma*(g-c)
        guider.set_conds(erp_geometric, positive, negative)
        guider.set_cfg(erp_gamma, guidance_scale, nested=False)
        return (guider,)

class _CircularVAEProxy:
    def __init__(self, vae, pad_latent_columns):
        self._spherope_vae = vae
        self._spherope_pad = int(pad_latent_columns)

    def __getattr__(self, name):
        return getattr(self._spherope_vae, name)

    def _pad_latent(self, latent):
        pad = min(self._spherope_pad, latent.shape[-1] // 2 - 1)
        if pad < 1:
            raise ValueError("Latent width is too small for circular VAE decoding.")
        padded = torch.cat([latent[..., -pad:], latent, latent[..., :pad]], dim=-1)
        return padded, pad

    def _crop_image(self, images, pad):
        pixel_pad = pad * int(self._spherope_vae.spacial_compression_decode())
        return images[..., pixel_pad:-pixel_pad, :]

    def decode(self, latent, *args, **kwargs):
        padded, pad = self._pad_latent(latent)
        images = self._spherope_vae.decode(padded, *args, **kwargs)
        return self._crop_image(images, pad)

    def decode_tiled(self, latent, *args, **kwargs):
        padded, pad = self._pad_latent(latent)
        images = self._spherope_vae.decode_tiled(padded, *args, **kwargs)
        return self._crop_image(images, pad)


class SpheRoPECircularVAEPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "pad_latent_columns": ("INT", {"default": 18, "min": 1, "max": 64}),
        }}
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("circular_vae",)
    FUNCTION = "patch"
    CATEGORY = "SpheRoPE/native"
    DESCRIPTION = "Wrap a VAE with horizontal latent padding/cropping so standard VAE Decode nodes remain seam-safe."

    def patch(self, vae, pad_latent_columns):
        return (_CircularVAEProxy(vae, pad_latent_columns),)


class SpheRoPEPipelinePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "pad_latent_columns": ("INT", {"default": 18, "min": 1, "max": 64}),
        }}
    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("spherope_model", "circular_vae")
    FUNCTION = "patch"
    CATEGORY = "SpheRoPE/native"
    DESCRIPTION = "Apply the FLUX.1 SFC/ERP sampler hooks and wrap the VAE for seam-safe standard decoding."

    def patch(self, model, vae, pad_latent_columns):
        spherope_model = SpheRoPESFCModelPatch().patch(model)[0]
        circular_vae = _CircularVAEProxy(vae, pad_latent_columns)
        return spherope_model, circular_vae


class SpheRoPECircularVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "vae": ("VAE",),
            "pad_latent_columns": ("INT", {"default": 18, "min": 1, "max": 64}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "SpheRoPE/native"

    def decode(self, samples, vae, pad_latent_columns):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        pad = min(pad_latent_columns, latent.shape[-1] // 2 - 1)
        if pad < 1:
            raise ValueError("Latent width is too small for circular VAE decoding.")
        padded = torch.cat([latent[..., -pad:], latent, latent[..., :pad]], dim=-1)
        images = vae.decode(padded)
        pixel_pad = pad * int(vae.spacial_compression_decode())
        images = images[:, :, pixel_pad:-pixel_pad, :]
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)

NODE_CLASS_MAPPINGS = {
    "SpheRoPESFCModelPatch": SpheRoPESFCModelPatch,
    "SpheRoPEEncodeERP": SpheRoPEEncodeERP,
    "SpheRoPEERPConditioning": SpheRoPEERPConditioning,
    "SpheRoPEERPGuider": SpheRoPEERPGuider,
    "SpheRoPESemanticCFGGuider": SpheRoPESemanticCFGGuider,
    "SpheRoPECircularVAEPatch": SpheRoPECircularVAEPatch,
    "SpheRoPEPipelinePatch": SpheRoPEPipelinePatch,
    "SpheRoPECircularVAEDecode": SpheRoPECircularVAEDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpheRoPESFCModelPatch": "SpheRoPE SFC Model Patch",
    "SpheRoPEEncodeERP": "SpheRoPE Encode ERP Prompts",
    "SpheRoPEERPConditioning": "SpheRoPE ERP Conditioning",
    "SpheRoPEERPGuider": "SpheRoPE ERP CFG Guider",
    "SpheRoPESemanticCFGGuider": "SpheRoPE Semantic CFG Guider",
    "SpheRoPECircularVAEPatch": "SpheRoPE Circular VAE Patch",
    "SpheRoPEPipelinePatch": "SpheRoPE Pipeline Patch",
    "SpheRoPECircularVAEDecode": "SpheRoPE Circular VAE Decode",
}
