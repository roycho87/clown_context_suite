"""Clown Keystone and Clown Late Lock.

Clown Keystone puts a latent from a clean clip (normally clip 1) into every shot, locked:
- keystone = clip 1's first latent (frame 0, a single frame on its own).
- capstone = clip 1's last single-frame latent (the first latent of its last 17-frame block, e.g. latent 55 of a
  57-latent clip = frame 187).
Placement options: keystone front (default), keystone front + back, capstone front + back, keystone front + capstone back.
front = this shot's latent 0, back = this shot's last single-frame slot.

H3 latents come in 17-frame blocks of (1, 4, 4, 4, 4) frames, so only that one single-frame latent is replaced and the
block alignment doesn't change.

Clown Late Lock leaves the grafted slot(s) free for the first few steps, then locks them for the rest, so the shot
decides its own motion and pose before the graft pulls on it.

Stitching: the grafted frames aren't part of the continuation, so cut the last 17 frames of each shot and the first 22
of the next (Seamless Video Combiner: Use Beginning pair on, Beginning Overlap 39, Beginning Join 22).

Wire it after Clown Context (and Clown Anchoring), right before the sampler.
"""
import torch

from .nodes import _split, _carry_masks, _pack, _has_samples, _frames_of, NestedTensor


def _frames_before(n_latents):
    return _frames_of(n_latents)

CATEGORY = "Clown Context Suite"
PLACEMENTS = ["keystone front", "keystone front + back", "capstone front + back", "keystone front + capstone back"]


class H3ClownKeyframe:
    """Clown Keystone: clip 1's keystone / capstone grafted onto every shot."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "graft"
    DESCRIPTION = ("Copies the first latent (frame 0, a single frame) of clip 1 into the start of this shot and "
                   "locks it, so every shot starts from the original clean frame. Put it after Clown Context, right before "
                   "the sampler. Feed clip 1 from a Load Latent fixed at clip 1 (outside the Chain group). When you stitch, cut "
                   "the last 17 frames of each shot and the first 22 of the next. With shot_number plugged in, clip 1 isn't "
                   "loaded on the first shot.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The shot's AV latent (after Clown Context / Clown Anchoring)."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "on - graft the keystone / capstone", "label_off": "off - pass through"}),
                "keyframe_mask": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Noise mask on the grafted keystone / capstone. 0 = kept exactly. 0.2-0.3 = kept roughly, "
                                                       "the sampler may adjust it. 1 = ignored (it's redrawn from noise)."}),
                "first_shot": ("INT", {"default": 1, "min": -9999, "max": 99999,
                                       "tooltip": "source_clip: the clip the keystone / capstone come from (set the Load Latent "
                                                  "feeding clip 1 to the same clip). On that shot and earlier nothing is grafted and "
                                                  "the latent passes through. Only used when shot_number is plugged in."}),
                "placement": (PLACEMENTS, {"default": PLACEMENTS[0],
                                           "tooltip": "keystone = clip 1's first latent (frame 0). capstone = its last "
                                                      "single-frame latent (first latent of its last 17-frame block). front = this "
                                                      "shot's latent 0, back = this shot's last single-frame slot."}),
            },
            "optional": {
                "keyframe": ("LATENT", {"lazy": True,
                                        "tooltip": "Clip 1 (a Load Latent fixed at clip 1): where the keystone (its first latent) and "
                                                   "capstone (its last single-frame latent) come from."}),
                "shot_number": ("INT", {"forceInput": True, "tooltip": "From Clown Shot Counter. Skips the first shot."}),
            },
        }

    def check_lazy_status(self, enabled=True, first_shot=1, shot_number=None, keyframe=None, **kw):
        if not bool(enabled):
            return []
        if shot_number is not None and int(shot_number) <= int(first_shot):
            return []
        return ["keyframe"] if keyframe is None else []

    def graft(self, latent, enabled=True, keyframe_mask=0.0, first_shot=1, keyframe=None, shot_number=None,
              placement="keystone front"):
        who = "Clown Keystone"
        if not bool(enabled):
            print(f"[{who}] off - latent passed through")
            return (latent,)
        if shot_number is not None and int(shot_number) <= int(first_shot):
            print(f"[{who}] shot {shot_number} is the first shot - clip 1 not loaded, latent passed through")
            return (latent,)
        if not _has_samples(keyframe):
            print(f"[{who}] no clip 1 latent - latent passed through")
            return (latent,)
        samples = latent["samples"]
        packed = NestedTensor is not None and isinstance(samples, NestedTensor)
        parts = _split(samples)
        vi = max(range(len(parts)), key=lambda i: parts[i].ndim)
        video = parts[vi]
        if video.ndim != 5:
            raise ValueError(f"{who}: expected a 5-D video latent [B, C, T, H, W], got {tuple(video.shape)}.")
        ks = keyframe["samples"]
        if isinstance(ks, (list, tuple)):
            ks = [p for p in ks if p is not None]
        kparts = _split(ks)
        src = max(kparts, key=lambda p: p.ndim)
        if src.ndim != 5 or src.shape[1] != video.shape[1] or src.shape[3:] != video.shape[3:]:
            raise ValueError(f"{who}: clip 1 latent {tuple(src.shape)} doesn't match this shot {tuple(video.shape)} "
                             "(same model and resolution needed).")
        video = video.clone()
        key_src = 0                                          # keystone: clip 1's first latent
        cap_src = 5 * ((int(src.shape[2]) - 1) // 5)         # capstone: its last single-frame latent
        back = 5 * ((int(video.shape[2]) - 1) // 5)          # this shot's last single-frame slot
        pick = {"keystone front": (key_src, None),
                "keystone front + back": (key_src, key_src),
                "capstone front + back": (cap_src, cap_src),
                "keystone front + capstone back": (key_src, cap_src)}.get(placement, (key_src, None))
        front_src, back_src = pick
        if back_src is not None and back <= 0:
            back_src = None
        slots = [(0, front_src)] + ([(back, back_src)] if back_src is not None else [])
        for dst, s_i in slots:
            video[:, :, dst:dst + 1] = src[:, :, s_i:s_i + 1].to(device=video.device, dtype=video.dtype)
        parts[vi] = video
        masks = _carry_masks(latent, parts, packed, vi)
        vmask = masks[vi]
        vmask = torch.ones_like(video) if vmask.shape != video.shape else vmask.clone()
        for dst, _ in slots:
            vmask[:, :, dst:dst + 1] = float(keyframe_mask)
        masks[vi] = vmask
        name = lambda i: "keystone" if i == key_src else "capstone"
        note = ", ".join(f"{name(s_i)} (clip 1 latent {s_i}) -> latent {dst} = frame {_frames_before(dst)}" for dst, s_i in slots)
        out = _pack(latent, parts, masks, packed, who)
        out["clown_keyframe_slots"] = [int(dst) for dst, _ in slots]   # tells Clown Late Lock what was grafted
        print(f"[{who}] shot {shot_number}: {placement}: {note} (mask {keyframe_mask})")
        return (out,)


LOCK_SLOTS = ["capstone slot (back)", "keystone slot (front)", "keystone + capstone slots"]
_OLD_SLOTS = {"back": LOCK_SLOTS[0], "front": LOCK_SLOTS[1], "front + back": LOCK_SLOTS[2]}


class H3ClownLateLock:
    """Keeps Clown Keystone's grafted slot(s) FREE for the first swap_step steps, then locks them for the rest."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "patch"
    DESCRIPTION = ("Put it between Clown Keystone and the sampler: latent from Clown Keystone in, latent out to the sampler; "
                   "model in, model out to the guider. For the first swap_step steps the chosen slot(s) generate freely "
                   "(so motion and pose are decided with nothing pulling); from then on they're locked to what Clown "
                   "Keystone put there (re-noised to each step's level by the sampler, so it blends in).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "latent": ("LATENT", {"tooltip": "From Clown Keystone (passed through to the sampler)."}),
            "swap_step": ("INT", {"default": 4, "min": 0, "max": 200,
                                  "tooltip": "Steps the slot stays free before it's locked. 4 of 8 = free for steps 1-4, locked for 5-8. "
                                             "0 = locked from the start (same as plain Clown Keystone)."}),
            "slots": (LOCK_SLOTS, {"default": LOCK_SLOTS[0],
                                   "tooltip": "capstone slot = the shot's last single-frame slot (back). keystone slot = latent 0 "
                                              "(front). Only slots Clown Keystone actually filled are touched."}),
        }}

    def patch(self, model, latent, swap_step=4, slots=LOCK_SLOTS[0]):
        slots = _OLD_SLOTS.get(slots, slots)
        who = "Clown Late Lock"
        parts = _split(latent["samples"])
        video = max(parts, key=lambda p: p.ndim)
        if video.ndim != 5:
            raise ValueError(f"{who}: expected a 5-D video latent, got {tuple(video.shape)}.")
        C, T, H, W = (int(x) for x in video.shape[1:])
        back = 5 * ((T - 1) // 5)
        want = {LOCK_SLOTS[0]: [back], LOCK_SLOTS[1]: [0], LOCK_SLOTS[2]: [0, back]}[slots]
        grafted = latent.get("clown_keyframe_slots") or []
        idx = [t for t in want if t in grafted]
        if not idx:
            print(f"[{who}] nothing grafted in the {slots} slot(s) on this shot (e.g. the first shot) - model passed through")
            return (model, latent)
        n_video = C * T * H * W
        step = int(swap_step)
        m = model.clone()
        prev = m.model_options.get("denoise_mask_function")

        def fn(sigma, mask, extra_options):
            if prev is not None:
                mask = prev(sigma, mask, extra_options)
            sigmas = extra_options.get("sigmas")
            i = 0
            if sigmas is not None:
                s = float(torch.as_tensor(sigma).flatten()[0])
                i = int(torch.argmin((sigmas.float().cpu() - s).abs()))
            val = 0.0 if i >= step else 1.0
            mask = mask.contiguous().clone()                      # own, dense memory (the mask can be an expanded view)
            if mask.ndim == 3 and mask.shape[-1] >= n_video:      # packed AV mask [B, 1, N]
                vm = mask[:, :, :n_video].reshape(mask.shape[0], C, T, H, W).clone()
                for t in idx:
                    vm[:, :, t] = val
                mask = torch.cat([vm.reshape(mask.shape[0], 1, n_video), mask[:, :, n_video:]], dim=-1)
            elif mask.ndim == 5:                                  # video-only mask
                for t in idx:
                    mask[:, :, t] = val
            return mask

        m.model_options["denoise_mask_function"] = fn
        if latent.get("noise_mask") is None:
            print(f"[{who}] warning: the latent has no noise mask, so nothing can be locked")
        print(f"[{who}] {slots} slot(s) {idx}: free for steps 1-{step}, locked from step {step + 1}")
        return (m, latent)


NODE_CLASS_MAPPINGS = {"H3ClownKeyframe": H3ClownKeyframe, "H3ClownLateLock": H3ClownLateLock}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownKeyframe": "Clown Keystone", "H3ClownLateLock": "Clown Late Lock"}
