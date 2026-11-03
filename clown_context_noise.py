"""Geiru Context Noise: ethanfel's (Lumifel's) context noise, as a node in the Geiru Context Suite.

The idea is ethanfel's. On the Banodoco Discord (Aug 2026) he described recalculating "how much of the sampler's current
noise to apply to the context every sampling steps, while fading that treatment to zero near the end of the context so
the join stays clean". His own nodes are in https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop (Drift-Control).
All credit to him for it; this node just does the same thing on Geiru Context's held frames, so it can sit next to
Geiru Keystone and Late Lock.

Instead of handing the held frames to H3 fully clean at every step, give them some of the sampler's current noise, more
at the start of the held context and fading to none at its end, so the frames right next to the new ones stay clean.

H3 already supports this through partial masks: a mask value m puts a held row at m x the current noise level (the
model is told so too), and the sampler mixes it with the clean latent by the same amount. So this node just sets the
held latents' mask values on a ramp. After sampling, the real held latents are put back, so the join lines up.

Wire it after Geiru Context (and before Geiru Keystone, which keeps latent 0 clean and locked), and plug the model
through so the real latents can be put back. Geiru Context's degradation_protection switch turns it on and off.
"""
import torch

from .nodes import _split, _carry_masks, _pack, NestedTensor
from .clown_keyframe import _add_restore

CATEGORY = "Geiru Context Suite"
WHO = "Geiru Context Noise"


def _ramp(latent, strength, skip_first, shape):
    samples = latent["samples"]
    packed = NestedTensor is not None and isinstance(samples, NestedTensor)
    parts = _split(samples)
    vi = max(range(len(parts)), key=lambda i: parts[i].ndim)
    video = parts[vi]
    if video.ndim != 5:
        raise ValueError(f"{WHO}: expected a 5-D video latent [B, C, T, H, W], got {tuple(video.shape)}.")
    masks = _carry_masks(latent, list(parts), packed, vi)
    vm = masks[vi]
    if vm.shape != video.shape:
        return latent, {}
    T = int(video.shape[2])
    held = [t for t in range(T) if float(vm[:, :, t:t + 1].max()) < 0.5]
    if skip_first and held and held[0] == 0:
        held = held[1:]
    if not held:
        return latent, {}
    n = len(held)
    vm = vm.clone()
    originals = dict(latent.get("clown_keyframe_originals") or {})
    values = {}
    for i, t in enumerate(held):
        pos = i / max(1, n - 1)                       # 0 at the start of the held context, 1 at its end (the join)
        w = (1.0 - pos) if shape == "linear" else (1.0 - pos) ** 2
        m = float(strength) * w
        if m <= 0.0:
            continue
        vm[:, :, t:t + 1] = m
        values[t] = round(m, 3)
        originals.setdefault(t, video[:, :, t:t + 1].clone())
    masks[vi] = vm
    out = _pack(latent, parts, masks, packed, WHO)
    out["clown_keyframe_originals"] = originals
    return out, values


class H3ClownContextNoise:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT", "MODEL")
    RETURN_NAMES = ("av_latent", "model")
    FUNCTION = "noise"
    DESCRIPTION = ("Gives the held frames some of the sampler's current noise at every step instead of handing them over "
                   "fully clean: most at the start of the held context, fading to none at the join. The real held "
                   "latents are put back after sampling. Put it after Geiru Context, before Geiru Keystone, and plug the "
                   "model through.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "From Geiru Context."}),
                "enabled": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "How much of the current noise the first held latent gets (1 = the "
                                                  "same noise level as the new frames). Fades to 0 at the join."}),
                "fade": (["linear", "ease"], {"default": "linear"}),
                "skip_first": ("BOOLEAN", {"default": True,
                                           "tooltip": "Leave latent 0 alone (for Geiru Keystone's clean frame 0)."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Plug the model through so the real held latents are put back."}),
            },
        }

    def noise(self, latent, enabled=True, strength=1.0, fade="linear", skip_first=True, model=None):
        if not bool(enabled) or float(strength) <= 0:
            print(f"[{WHO}] off - latent passed through")
            return (latent, model)
        if isinstance(latent, dict) and latent.get("geiru_protection", True) is False:
            print(f"[{WHO}] degradation protection is off on Geiru Context - latent passed through")
            return (latent, model)
        out, values = _ramp(latent, strength, bool(skip_first), fade)
        if not values:
            print(f"[{WHO}] nothing held (first shot?) - latent passed through")
            return (latent, model)
        print(f"[{WHO}] held latents get part of the current noise: {values} (fading to clean at the join)"
              + ("" if model is not None else " - plug the model through to put the real ones back"))
        return (out, _add_restore(model, out, WHO))


NODE_CLASS_MAPPINGS = {"H3ClownContextNoise": H3ClownContextNoise}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownContextNoise": "Geiru Context Noise"}
