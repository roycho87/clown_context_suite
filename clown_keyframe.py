"""Clown Keystone and Clown Late Lock.

Clown Keystone uses latents from a clean clip (normally clip 1) in every shot:
- keystone = clip 1's first latent (frame 0, a single frame on its own). It's locked into this shot's latent 0.
- capstone = clip 1's last single-frame latent (the first latent of its last 17-frame block, e.g. latent 55 of a
  57-latent clip = frame 187).
Placement options: keystone front (default), keystone front + back, capstone front + back, keystone front + capstone back.

Front: written into the shot's latent 0, locked. With restore on, the chain's real latent 0 is put back into the
output after sampling, so the saved latent and the video keep the real frame.
Back: not written into the shot. It's tacked onto the back of the video for each model call (locked) and cut off
again, so nothing in the shot is replaced. At the back it lands in a 4-frame latent slot.
Restore and the back both need the model: plug it through Keystone, or use Clown Late Lock.

Clown Late Lock picks the step: the capstone is tacked on from swap_step, and the keystone slot is held empty (left as
noise the model doesn't change) until swap_step, then locked in. So the shot decides its own motion and pose first.

Wire it after Clown Context (and Clown Anchoring), right before the sampler.
"""
import torch

import comfy.patcher_extension
import comfy.utils

from .nodes import _split, _carry_masks, _pack, _has_samples, _frames_of, NestedTensor


def _restore_wrapper(originals, who):
    """OUTER_SAMPLE wrapper: after sampling, put the latents Keystone replaced back into the output."""
    def wrapper(executor, *args, **kwargs):
        out = executor(*args, **kwargs)
        try:
            shapes = kwargs.get("latent_shapes", None)
            multi = shapes is not None and len(shapes) > 1
            parts = [p.clone() for p in comfy.utils.unpack_latents(out, shapes)] if multi else [out.clone()]
            v = parts[0]
            done = []
            for slot, t in originals.items():
                if v.ndim == 5 and slot < v.shape[2] and t.shape[1] == v.shape[1] and t.shape[3:] == v.shape[3:]:
                    v[:, :, slot:slot + 1] = t.to(device=v.device, dtype=v.dtype)
                    done.append(slot)
            parts[0] = v
            out = comfy.utils.pack_latents(parts)[0] if multi else parts[0]
            if done:
                print(f"[{who}] put the original latent(s) {done} back into the output")
        except Exception as e:  # never lose a finished render over this
            print(f"[{who}] couldn't restore the original latent ({e}); output left as sampled")
        return out
    return wrapper


def _tack_wrapper(lat, start, prev, who):
    """model_function_wrapper: from step `start` on, tack `lat` (one video latent, locked) onto the back of the video
    for each model call and cut it off the output again, so the sampler never sees it."""
    warned = {"shape": False}

    def wrapper(apply_model, args):
        x, t, c = args["input"], args["timestep"], args["c"]

        def run(xx, cc):
            if prev is None:
                return apply_model(xx, t, **cc)
            return prev(apply_model, {"input": xx, "timestep": t, "c": cc, "cond_or_uncond": args.get("cond_or_uncond")})

        to = c.get("transformer_options", {}) or {}
        sigmas = to.get("sample_sigmas", None)
        step = 0
        if sigmas is not None:
            step = int(torch.argmin((sigmas.float().cpu() - float(t.flatten()[0])).abs()))
        shapes = c.get("latent_shapes", None)
        if step < start or shapes is None or len(shapes) < 2:
            return run(x, c)
        win = to.get("context_window", None)
        if win is not None and win.index_list[-1] != win.total_frames - 1:
            return run(x, c)                       # context windows: only the window holding the end
        parts = comfy.utils.unpack_latents(x, shapes)
        v = parts[0]
        if v.ndim != 5 or lat.shape[1] != v.shape[1] or lat.shape[3:] != v.shape[3:]:
            if not warned["shape"]:
                print(f"[{who}] back latent {tuple(lat.shape)} doesn't match the shot {tuple(v.shape)} - skipped")
                warned["shape"] = True
            return run(x, c)
        n = v.shape[2]
        d = lat.to(device=v.device, dtype=v.dtype).expand(v.shape[0], -1, -1, -1, -1)
        x2, shapes2 = comfy.utils.pack_latents([torch.cat([v, d], dim=2)] + list(parts[1:]))
        c2 = dict(c)
        c2["latent_shapes"] = shapes2
        dm = c.get("denoise_mask", None)
        if dm is None:
            dm = torch.ones((v.shape[0], 1, n) + tuple(v.shape[3:]), device=v.device, dtype=torch.float32)
        c2["denoise_mask"] = torch.cat([dm, torch.zeros_like(dm[:, :, :1])], dim=2)   # the extra latent is locked
        out = comfy.utils.unpack_latents(run(x2, c2), shapes2)
        out = [out[0][:, :, :n]] + list(out[1:])
        return comfy.utils.pack_latents(out)[0]

    wrapper._clown_back = True
    wrapper._prev = prev
    return wrapper


def _set_tack(model, lat, start, who):
    """Clone the model and tack `lat` onto the back from `start`, replacing an earlier Keystone/Late Lock tack-on."""
    m = model.clone()
    prev = m.model_options.get("model_function_wrapper", None)
    if getattr(prev, "_clown_back", False):
        prev = prev._prev
    m.set_model_unet_function_wrapper(_tack_wrapper(lat, int(start), prev, who))
    return m


def _add_restore(model, latent, who):
    originals = latent.get("clown_keyframe_originals") if isinstance(latent, dict) else None
    if model is None or not originals:
        return model
    m = model.clone()
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "clown_keystone_restore",
                           _restore_wrapper(originals, who))
    return m


def _frames_before(n_latents):
    return _frames_of(n_latents)

CATEGORY = "Clown Context Suite"
PLACEMENTS = ["keystone front", "keystone front + back", "capstone front + back", "keystone front + capstone back"]


class H3ClownKeyframe:
    """Clown Keystone: clip 1's keystone / capstone grafted onto every shot."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT", "MODEL")
    RETURN_NAMES = ("av_latent", "model")
    FUNCTION = "graft"
    DESCRIPTION = ("Copies the first latent (frame 0, a single frame) of clip 1 into the start of this shot and "
                   "locks it, so every shot starts from the original clean frame. Put it after Clown Context, right before "
                   "the sampler. Feed clip 1 from a Load Latent fixed at clip 1 (outside the Chain group). Trim the first "
                   "frame of each shot when you stitch. With shot_number plugged in, clip 1 isn't loaded on the first shot.")

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
                "model": ("MODEL", {"tooltip": "Optional: plug the model through here (model out -> guider / Late Lock) so "
                                               "the chain's real latent 0 is put back into the output after sampling."}),
                "restore": ("BOOLEAN", {"default": True, "label_on": "on - put the real latent 0 back after sampling",
                                        "label_off": "off - keep the keystone in the output",
                                        "tooltip": "Needs the model plugged through Keystone or Late Lock. Only latent 0 "
                                                   "is restored, and only when Clown Context was holding it."}),
            },
        }

    def check_lazy_status(self, enabled=True, first_shot=1, shot_number=None, keyframe=None, **kw):
        if not bool(enabled):
            return []
        if shot_number is not None and int(shot_number) <= int(first_shot):
            return []
        return ["keyframe"] if keyframe is None else []

    def graft(self, latent, enabled=True, keyframe_mask=0.0, first_shot=1, keyframe=None, shot_number=None,
              placement="keystone front", model=None, restore=True):
        who = "Clown Keystone"
        if not bool(enabled):
            print(f"[{who}] off - latent passed through")
            return (latent, model)
        if shot_number is not None and int(shot_number) <= int(first_shot):
            print(f"[{who}] shot {shot_number} is the first shot - clip 1 not loaded, latent passed through")
            return (latent, model)
        if not _has_samples(keyframe):
            print(f"[{who}] no clip 1 latent - latent passed through")
            return (latent, model)
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
        old_masks = _carry_masks(latent, list(parts), packed, vi)
        old_vmask = old_masks[vi]
        original_0 = video[:, :, 0:1].clone()   # the chain's own latent 0 (Clown Context's held frame)
        held_0 = old_vmask.shape == video.shape and float(old_vmask[:, :, 0:1].max()) < 0.5
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
        slots = [(0, front_src)]
        back_lat = src[:, :, back_src:back_src + 1].clone() if back_src is not None else None
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
        if back_lat is not None:
            note += f", {name(back_src)} (clip 1 latent {back_src}) tacked onto the back"
        out = _pack(latent, parts, masks, packed, who)
        out["clown_keyframe_slots"] = [int(dst) for dst, _ in slots]   # tells Clown Late Lock what was grafted
        if back_lat is not None:
            out["clown_keyframe_back"] = back_lat                       # tacked on by the model patch, never written in
        if bool(restore) and held_0:
            out["clown_keyframe_originals"] = {0: original_0}             # put back after sampling
        print(f"[{who}] shot {shot_number}: {placement}: {note} (mask {keyframe_mask})")
        if bool(restore):
            if not held_0:
                print(f"[{who}] latent 0 wasn't held by Clown Context, nothing to restore")
            elif model is None:
                print(f"[{who}] restore is on: plug the model through Keystone or Late Lock to put latent 0 back")
        m = model
        if back_lat is not None:
            if model is None:
                print(f"[{who}] the back needs the model plugged through Keystone or Clown Late Lock")
            else:
                m = _set_tack(m, back_lat, 0, who)
        return (out, _add_restore(m, out, who) if bool(restore) else m)


LOCK_SLOTS = ["capstone slot (back)", "keystone slot (front)", "keystone + capstone slots"]
_OLD_SLOTS = {"back": LOCK_SLOTS[0], "front": LOCK_SLOTS[1], "front + back": LOCK_SLOTS[2]}


class H3ClownLateLock:
    """Front: holds Keystone's keystone slot empty for the first swap_step steps, then locks it in.
    Back: tacks Keystone's back latent onto the back of the video from swap_step on (cut off again after each call)."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "patch"
    DESCRIPTION = ("Put it between Clown Keystone and the sampler: latent from Clown Keystone in, latent out to the sampler; "
                   "model in, model out to the guider. Capstone (back): for the first swap_step steps nothing is there, then "
                   "Keystone's back latent is tacked onto the back of the video for each model call (locked) and cut off "
                   "again, so nothing in the shot is replaced. Keystone (front): for the first swap_step steps the slot is "
                   "held empty (left as noise the model doesn't change), then locked to the keystone.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "latent": ("LATENT", {"tooltip": "From Clown Keystone (passed through to the sampler)."}),
            "swap_step": ("INT", {"default": 4, "min": 0, "max": 200,
                                  "tooltip": "Steps before the graft comes in. 4 of 8 = steps 1-4 without, 5-8 with. "
                                             "0 = from the start."}),
            "slots": (LOCK_SLOTS, {"default": LOCK_SLOTS[0],
                                   "tooltip": "capstone slot = Keystone's back latent, tacked onto the back. keystone slot = "
                                              "latent 0 (front). Only what Clown Keystone actually set up is touched."}),
        }}

    def patch(self, model, latent, swap_step=4, slots=LOCK_SLOTS[0]):
        slots = _OLD_SLOTS.get(slots, slots)
        who = "Clown Late Lock"
        parts = _split(latent["samples"])
        video = max(parts, key=lambda p: p.ndim)
        if video.ndim != 5:
            raise ValueError(f"{who}: expected a 5-D video latent, got {tuple(video.shape)}.")
        C, T, H, W = (int(x) for x in video.shape[1:])
        want_front = slots in (LOCK_SLOTS[1], LOCK_SLOTS[2])
        want_back = slots in (LOCK_SLOTS[0], LOCK_SLOTS[2])
        grafted = latent.get("clown_keyframe_slots") or []
        idx = [0] if want_front and 0 in grafted else []
        back_lat = latent.get("clown_keyframe_back", None) if want_back else None
        step = int(swap_step)
        if not idx and back_lat is None:
            print(f"[{who}] nothing set up for the {slots} on this shot (e.g. the first shot) - model passed through")
            return (_add_restore(model, latent, who), latent)
        if back_lat is not None:
            model = _set_tack(model, back_lat, step, who)
            print(f"[{who}] capstone tacked onto the back from step {step + 1}")
        if not idx:
            return (_add_restore(model, latent, who), latent)
        n_video = C * T * H * W
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
        m = _add_restore(m, latent, who)
        if latent.get("noise_mask") is None:
            print(f"[{who}] warning: the latent has no noise mask, so nothing can be locked")
        print(f"[{who}] keystone slot held empty for steps 1-{step}, locked from step {step + 1}")
        return (m, latent)


NODE_CLASS_MAPPINGS = {"H3ClownKeyframe": H3ClownKeyframe, "H3ClownLateLock": H3ClownLateLock}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownKeyframe": "Clown Keystone", "H3ClownLateLock": "Clown Late Lock"}
