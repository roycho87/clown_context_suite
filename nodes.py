"""
Clown Context Suite for MiniMax H3
----------------------------------
Chained continuation for MiniMax H3 in ComfyUI.

    Clown Context           previous shot's TAIL -> this shot's HEAD (video and audio)
    Clown Anchoring         a source's HEAD      -> this shot's TAIL (video and audio)
    Clown Shot Counter      shot number: classic counter, or counted from saved latents
    Clown Force Audio       puts a real sound track into the latent so the sampler keeps it

(Clown Prophecy and Clown Carryover moved to roycho87/Clown-Prophecy-Experimental-.)

The anchor nodes copy real latents into the starting latent at MATCHING BLOCK
POSITIONS (FRAME_PER_TOKEN = (1, 4, 4, 4, 4) from frame 0; 17-frame blocks) and set
the noise mask to 0 on them, so the sampler keeps them exactly.

Audio runs at 40 Hz against 24 fps video, so F frames = round(F * 40/24) audio steps.
"""

import math
import os
import re

import torch

try:
    from comfy.nested_tensor import NestedTensor
except Exception:  # very old ComfyUI
    NestedTensor = None


CATEGORY = "Clown Context Suite"

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
BLOCK = 17
FPS = 24.0
AUDIO_HZ = 40.0
FRAME_RESCALE = AUDIO_HZ / FPS          # 5/3 audio steps per video frame


# --------------------------------------------------------------------------- helpers

def _need_nested(who):
    if NestedTensor is None:
        raise RuntimeError(f"{who}: this ComfyUI has no comfy.nested_tensor; update ComfyUI.")


def _as_tensor(part):
    if isinstance(part, dict) and "samples" in part:
        return part["samples"]
    return part


def _split(value):
    """NestedTensor / list / tensor -> list of tensors, video (5-D) first."""
    if NestedTensor is not None and isinstance(value, NestedTensor):
        parts = list(value.unbind())
    elif isinstance(value, (list, tuple)):
        parts = [_as_tensor(p) for p in value]
    else:
        parts = [value]
    return parts


def _has_samples(lat):
    if lat is None or not isinstance(lat, dict):
        return False
    s = lat.get("samples", None)
    if s is None:
        return False
    if isinstance(s, (list, tuple)) and (len(s) == 0 or any(p is None for p in s)):
        return False
    return True


def _frames_of(n_latents):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(n_latents))


def _latents_in(frames):
    """Latent count for `frames` frames starting at a block boundary (frames must end on a latent edge)."""
    full, r = divmod(frames, BLOCK)
    return 5 * full + (0 if r == 0 else 1 + (r - 1) // 4)


def _aligned_overlap(overlap, clip_frames):
    """Largest overlap <= requested whose start (clip_frames - overlap) is a block boundary."""
    f = int(overlap) - ((int(overlap) - clip_frames) % BLOCK)
    return max(0, min(f, clip_frames))


def _drop_blocks(drop_frames):
    """drop_frames -> whole 17-frame blocks to drop. Old workflows saved a True/False switch here."""
    if isinstance(drop_frames, bool):
        return 1 if drop_frames else 0
    return max(0, int(drop_frames)) // BLOCK


def _video_first(parts):
    return sorted(parts, key=lambda t: t.ndim, reverse=True)


def _audio_index(parts):
    """Index of the audio stream in a split AV latent, or None.

    The video stream is 5-D [B, C, T, H, W]; H3's audio stream is 4-D
    [B, C, 2, T] (or 3-D unbatched). Anything else is left alone.
    """
    vi = max(range(len(parts)), key=lambda i: parts[i].ndim)
    for i, p in enumerate(parts):
        if i != vi and p.ndim in (3, 4):
            return i
    return None


def _audio_steps(frames):
    """Video frames -> 40 Hz audio latent steps, rounded the way H3 allocates the grid."""
    return int(round(float(frames) * FRAME_RESCALE))


def _audio_len(a):
    return int(a.shape[-1])


def _grid_note(who, frames, label):
    """Warn when a window does not land exactly on the 40 Hz grid."""
    if int(frames) % 3 != 0:
        want = float(frames) * FRAME_RESCALE
        print(f"[{who}] {label} {frames} frames wants {want:.2f} audio steps, "
              f"using {_audio_steps(frames)} (multiples of 3 land exactly on the 40 Hz grid)")


def _overhang_note(who, video, audio, label):
    """H3 rounds the audio grid to the nearest step; a clip can over- or undershoot by 1/3 of a step."""
    frames = _frames_of(int(video.shape[2]))
    over = _audio_len(audio) - FRAME_RESCALE * frames
    if not (-0.5 < over < 0.5):
        print(f"[{who}] {label} audio grid looks wrong: {_audio_len(audio)} steps for {frames} frames. "
              "Holding audio anyway, but check the join.")
    return over


def _feather(mask, start, length, ticks, from_end):
    """Half-cosine release over `ticks` steps at one edge of a held audio run.

    Off by default. A hard audio mask edge can click; this lets the model ease
    back in instead. `from_end` feathers the last ticks of the run, otherwise
    the first.
    """
    ticks = max(0, min(int(ticks), length))
    if ticks <= 0:
        return
    for k in range(ticks):
        # 0 at the held side, 1 at the free side
        t = (k + 1) / float(ticks + 1)
        v = 0.5 - 0.5 * math.cos(math.pi * t)
        idx = start + length - 1 - k if from_end else start + k
        mask[..., idx] = v


def _carry_masks(latent, parts, packed, vi):
    """Existing noise mask if it matches, otherwise fresh all-ones masks."""
    old = latent.get("noise_mask", None)
    masks = _split(old) if old is not None else None
    if masks is not None and len(masks) != len(parts):
        masks = None
    if masks is None:
        masks = [torch.ones_like(p) for p in parts]
        if old is not None and not packed:
            masks[vi] = old
    return masks


def _pack(latent, parts, masks, packed, who):
    out = {kk: v for kk, v in latent.items() if kk not in ("samples", "noise_mask")}
    if packed or len(parts) > 1:
        _need_nested(who)
        out["samples"] = NestedTensor(tuple(parts))
        out["noise_mask"] = NestedTensor(tuple(masks))
    else:
        out["samples"] = parts[0]
        out["noise_mask"] = masks[0]
    return out


AUDIO_INPUTS = {
    "audio_frames": ("INT", {"default": 0, "min": 0, "max": 4096,
                             "tooltip": "Frames of audio to hold. 0 = the same window as the video. "
                                        "The window is END-ALIGNED with the pinned video, so this only "
                                        "changes how far back the sound reaches. Multiples of 3 land "
                                        "exactly on the 40 Hz audio grid."}),
    "audio_feather_ticks": ("INT", {"default": 0, "min": 0, "max": 64,
                                    "tooltip": "Half-cosine release over this many audio steps at the free "
                                               "end of the held audio, so the model eases back in instead of "
                                               "cutting. 0 = hard edge. 8 steps is 0.2 s."}),
}


# --------------------------------------------------------------------------- repack


# --------------------------------------------------------------------------- context (head)

def _drop_fl(drop_last, drop_first, F_all, who):
    """Whole 17-frame blocks to drop from an F_all-frame window, returned as (earlier, later).

    drop_first  = frames dropped from the BEGINNING  (the earliest frames of the window)
    drop_last = frames dropped from the END (the latest frames of the window)
    """
    early, late = _drop_blocks(drop_first), _drop_blocks(drop_last)
    for name, v, k in (("drop_last", drop_last, late), ("drop_first", drop_first, early)):
        if not isinstance(v, bool) and int(v) != BLOCK * k:
            print(f"[{who}] {name} {v} is not a multiple of 17 - using {BLOCK * k}")
    return early, late


DROP_TIP = ("in whole 17-frame blocks (0, 17, 34 ...). Every held frame stays in the same "
            "block position it was made in.")


class H3ClownContextExperimental:
    """Previous shot's tail held at the start of this shot, video and audio."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "anchor"
    DESCRIPTION = (
        "Holds the previous shot's last frames at the start of this shot, in matching block "
        "positions, masked so the sampler keeps them. Audio is held too. drop_last / drop_first "
        "trim the held window from either end."
    )

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "latent": ("LATENT", {"tooltip": "Starting AV latent that goes into the sampler."}),
            "head_frames": ("INT", {"default": 39, "min": 5, "max": 124, "step": 17, "round": 1,
                                    "tooltip": "Overlap in frames (use the audio overlap, e.g. 39). This shot's frames "
                                               "0..head_frames-1 line up with the previous shot's last head_frames frames."}),
            "drop_last": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 17,
                                   "tooltip": "Frames dropped from the END (the latest frames of the overlap) and generated freely, "
                                              + DROP_TIP + " 39 with 17 -> previous 204-225 held in this shot's 0-21; 22-38 free."}),
            "drop_first": ("INT", {"default": 17, "min": 0, "max": 4096, "step": 17,
                                  "tooltip": "Frames dropped from the BEGINNING (the earliest frames of the overlap) and generated freely, "
                                             + DROP_TIP + " 39 with 17 -> previous 221-242 held in this shot's 17-38; 0-16 free."}),
            "hold_audio": ("BOOLEAN", {"default": True,
                                       "tooltip": "Hold the previous shot's audio under the held video. Off = video only, audio is regenerated."}),
            "first_shot": ("INT", {"default": 1, "min": -9999, "max": 99999,
                                   "tooltip": "Number of the first shot of your run. On that shot (or earlier) the previous latent is ignored."}),
        }
        req.update(AUDIO_INPUTS)
        return {
            "required": req,
            "optional": {
                "shot_number": ("INT", {"forceInput": True,
                                        "tooltip": "Connect your Shot Number. If left empty, the first-shot check is skipped."}),
                "previous_shot": ("LATENT", {"lazy": True,
                                             "tooltip": "Previous shot's saved latent (list or packed). If empty, the latent passes through. "
                                                        "Not loaded at all on the first shot."}),
            },
        }

    def check_lazy_status(self, first_shot=1, shot_number=None, previous_shot=None, **kw):
        # first shot: don't run the Load Latent node at all (MultiRef's Load errors on an empty folder at clip_index 0)
        if shot_number is not None and int(shot_number) <= int(first_shot):
            return []
        return ["previous_shot"] if previous_shot is None else []

    def anchor(self, latent, head_frames, drop_last=0, drop_first=17, hold_audio=True, first_shot=1,
               audio_frames=0, audio_feather_ticks=0, shot_number=None, previous_shot=None):
        who = "Clown Context Experimental"
        if shot_number is not None and int(shot_number) <= int(first_shot):
            print(f"[{who}] shot {shot_number} is the first shot (first_shot = {first_shot}) - previous latent ignored, latent passed through unchanged")
            return (latent,)
        if not _has_samples(previous_shot):
            print(f"[{who}] no previous shot latent - latent passed through unchanged")
            return (latent,)
        samples = latent["samples"]
        packed = NestedTensor is not None and isinstance(samples, NestedTensor)
        parts = _split(samples)
        vi = max(range(len(parts)), key=lambda i: parts[i].ndim)
        video = parts[vi]
        if video.ndim != 5:
            raise ValueError(f"{who}: expected a 5-D video latent [B, C, T, H, W], got {tuple(video.shape)}.")
        src_parts = _split(previous_shot["samples"])
        src = _video_first(src_parts)[0]
        if src.ndim != 5 or src.shape[1] != video.shape[1] or src.shape[3:] != video.shape[3:]:
            raise ValueError(
                f"{who}: source latent {tuple(src.shape)} doesn't match this shot {tuple(video.shape)} "
                "(same model and resolution needed)."
            )
        T, Ts = video.shape[2], src.shape[2]
        Lt, Ls = _frames_of(T), _frames_of(Ts)
        F_all = min(_aligned_overlap(head_frames, Ls), Lt, Ls)
        if F_all != int(head_frames):
            print(f"[{who}] overlap {head_frames} doesn't start on a block boundary - using {F_all}")
        kf, kb = _drop_fl(drop_last, drop_first, F_all, who)
        F = F_all - BLOCK * (kf + kb)          # held frames
        n = _latents_in(F_all) - 5 * (kf + kb)  # held latents
        if F <= 0 or n <= 0:
            print(f"[{who}] nothing to hold (overlap {F_all}, drop_last {drop_last}, drop_first {drop_first}) - latent passed through")
            return (latent,)
        hs = BLOCK * kf                          # first held frame of this shot
        sf0 = Ls - F_all + hs                    # matching frame of the previous shot
        d0 = 5 * kf
        s0 = 5 * ((Ls - F_all) // BLOCK) + 5 * kf
        video = video.clone()
        video[:, :, d0:d0 + n] = src[:, :, s0:s0 + n].to(device=video.device, dtype=video.dtype)
        parts[vi] = video

        masks = _carry_masks(latent, parts, packed, vi)
        vmask = masks[vi]
        vmask = torch.ones_like(video) if vmask.shape != video.shape else vmask.clone()
        vmask[:, :, d0:d0 + n] = 0.0
        masks[vi] = vmask

        # ---- audio: under the held frames, end-aligned with them ----
        held_audio = 0
        ai = _audio_index(parts)
        si = _audio_index(src_parts)
        if hold_audio and ai is not None and si is not None:
            ta, sa = parts[ai], src_parts[si]
            if ta.ndim == 3:
                ta = ta.unsqueeze(0)
            if sa.ndim == 3:
                sa = sa.unsqueeze(0)
            if ta.shape[1:-1] != sa.shape[1:-1]:
                print(f"[{who}] audio latent shapes differ ({tuple(sa.shape)} vs {tuple(ta.shape)}) - audio not held")
            else:
                want = min(int(audio_frames), F) if int(audio_frames) > 0 else F
                _grid_note(who, want, "audio window")
                _overhang_note(who, src, sa, "previous shot")
                he = hs + F                                    # one past the last held frame
                a1 = min(_audio_steps(he), _audio_len(ta))
                at = min(_audio_steps(he) - _audio_steps(he - want), a1)
                a0 = a1 - at
                src1 = min(_audio_steps(sf0 + F), _audio_len(sa))
                at = min(at, src1)
                a0 = a1 - at
                if at > 0:
                    ta = ta.clone()
                    ta[..., a0:a1] = sa[..., src1 - at:src1].to(device=ta.device, dtype=ta.dtype)
                    parts[ai] = ta
                    amask = masks[ai]
                    amask = torch.ones_like(ta) if amask.shape != ta.shape else amask.clone()
                    amask[..., a0:a1] = 0.0
                    _feather(amask, a0, at, audio_feather_ticks, from_end=True)
                    masks[ai] = amask
                    held_audio = at
        elif hold_audio:
            print(f"[{who}] no audio stream in one of the latents - video only")

        out = _pack(latent, parts, masks, packed, who)
        extra = (f", audio {held_audio} steps ({held_audio / AUDIO_HZ:.2f}s)"
                 + (f" with a {audio_feather_ticks}-step release" if audio_feather_ticks else "")
                 if held_audio else ", no audio held")
        print(f"[{who}] held latents {d0}..{d0 + n - 1} = frames {hs}..{hs + F - 1} "
              f"<- previous shot's frames {sf0}..{sf0 + F - 1} "
              f"(drop_last {BLOCK * kb}, drop_first {BLOCK * kf}){extra}")
        return (out,)


class H3BlockTailAnchorExperimental:
    """Next shot's head held at the end of this shot, video and audio."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "anchor"
    DESCRIPTION = (
        "Holds the source's first frames at the end of this shot, in matching block positions, "
        "masked so the sampler keeps them. Audio is held too. drop_last / drop_first trim the "
        "held window from either end. With shot_number plugged in, next_shot isn't loaded at all on the "
        "first shot (so a Load Latent of a clip that doesn't exist yet never runs)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "latent": ("LATENT", {"tooltip": "Starting AV latent that goes into the sampler."}),
            "tail_frames": ("INT", {"default": 39, "min": 5, "max": 124, "step": 17, "round": 1,
                                    "tooltip": "Window in frames: this shot's last tail_frames frames line up with the "
                                               "source's frames 0..tail_frames-1."}),
            "drop_last": ("INT", {"default": 17, "min": 0, "max": 4096, "step": 17,
                                   "tooltip": "Frames dropped from the END (last frames) (the end of this shot) and generated freely, "
                                              + DROP_TIP + " 39 with 17 -> source 0-21 held in this shot's 204-225; 226-242 free."}),
            "drop_first": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 17,
                                  "tooltip": "Frames dropped from the BEGINNING (the earliest frames of the window) and generated freely, "
                                             + DROP_TIP + " 39 with 17 -> source 17-38 held in this shot's 221-242; 204-220 free."}),
            "hold_audio": ("BOOLEAN", {"default": True,
                                       "tooltip": "Hold the source's audio over the pinned video. Off = video only, audio is regenerated."}),
        }
        req.update(AUDIO_INPUTS)
        req["first_shot"] = ("INT", {"default": 1, "min": -9999, "max": 99999,
                                     "tooltip": "Number of the first shot of your run. On that shot (or earlier) next_shot isn't "
                                                "loaded and the latent passes through. Only used when shot_number is plugged in."})
        return {
            "required": req,
            "optional": {
                "next_shot": ("LATENT", {"lazy": True,
                                         "tooltip": "Source latent (next shot / refined tail), list or packed. If empty, the latent passes through. "
                                                    "Not loaded at all on the first shot when shot_number is plugged in."}),
                "shot_number": ("INT", {"forceInput": True,
                                        "tooltip": "Optional. Connect your Shot Number to skip the first shot. If left empty, "
                                                   "Anchoring works as before."}),
            },
        }

    def check_lazy_status(self, first_shot=1, shot_number=None, next_shot=None, **kw):
        # first shot: don't run whatever feeds next_shot (e.g. a Load Latent of clip 1 that isn't made yet)
        if shot_number is not None and int(shot_number) <= int(first_shot):
            return []
        return ["next_shot"] if next_shot is None else []

    def anchor(self, latent, tail_frames, drop_last=17, drop_first=0, hold_audio=True,
               audio_frames=0, audio_feather_ticks=0, next_shot=None, first_shot=1, shot_number=None):
        who = "Clown Anchoring Experimental"
        if shot_number is not None and int(shot_number) <= int(first_shot):
            print(f"[{who}] shot {shot_number} is the first shot (first_shot = {first_shot}) - next_shot not loaded, latent passed through unchanged")
            return (latent,)
        if not _has_samples(next_shot):
            print(f"[{who}] no next shot latent - latent passed through unchanged")
            return (latent,)
        samples = latent["samples"]
        packed = NestedTensor is not None and isinstance(samples, NestedTensor)
        parts = _split(samples)
        vi = max(range(len(parts)), key=lambda i: parts[i].ndim)
        video = parts[vi]
        if video.ndim != 5:
            raise ValueError(f"{who}: expected a 5-D video latent [B, C, T, H, W], got {tuple(video.shape)}.")
        src_parts = _split(next_shot["samples"])
        src = _video_first(src_parts)[0]
        if src.ndim != 5 or src.shape[1] != video.shape[1] or src.shape[3:] != video.shape[3:]:
            raise ValueError(
                f"{who}: source latent {tuple(src.shape)} doesn't match this shot {tuple(video.shape)} "
                "(same model and resolution needed)."
            )
        T, Ts = video.shape[2], src.shape[2]
        Lt, Ls = _frames_of(T), _frames_of(Ts)
        F_all = min(_aligned_overlap(tail_frames, Lt), Lt, Ls)
        if F_all != int(tail_frames):
            print(f"[{who}] overlap {tail_frames} doesn't start on a block boundary - using {F_all}")
        kf, kb = _drop_fl(drop_last, drop_first, F_all, who)
        F = F_all - BLOCK * (kf + kb)
        n = _latents_in(F_all) - 5 * (kf + kb)
        if F <= 0 or n <= 0:
            print(f"[{who}] nothing to hold (overlap {F_all}, drop_last {drop_last}, drop_first {drop_first}) - latent passed through")
            return (latent,)
        ws = Lt - F_all                          # this shot's first window frame
        hs = ws + BLOCK * kf                     # first held frame of this shot
        sf0 = BLOCK * kf                         # matching source frame
        d0 = T - _latents_in(F_all) + 5 * kf
        s0 = 5 * kf
        video = video.clone()
        video[:, :, d0:d0 + n] = src[:, :, s0:s0 + n].to(device=video.device, dtype=video.dtype)
        parts[vi] = video

        masks = _carry_masks(latent, parts, packed, vi)
        vmask = masks[vi]
        vmask = torch.ones_like(video) if vmask.shape != video.shape else vmask.clone()
        vmask[:, :, d0:d0 + n] = 0.0
        masks[vi] = vmask

        # ---- audio: under the pinned frames, end-aligned with them ----
        held_audio = 0
        ai = _audio_index(parts)
        si = _audio_index(src_parts)
        if hold_audio and ai is not None and si is not None:
            ta, sa = parts[ai], src_parts[si]
            if ta.ndim == 3:
                ta = ta.unsqueeze(0)
            if sa.ndim == 3:
                sa = sa.unsqueeze(0)
            if ta.shape[1:-1] != sa.shape[1:-1]:
                print(f"[{who}] audio latent shapes differ ({tuple(sa.shape)} vs {tuple(ta.shape)}) - audio not held")
            else:
                want = min(int(audio_frames), F) if int(audio_frames) > 0 else F
                _grid_note(who, want, "audio window")
                _overhang_note(who, src, sa, "next shot")
                # this shot's audio grid is counted back from its end, the source's from its start
                off = _audio_len(ta) - _audio_steps(Lt)
                he = hs + F
                a1 = min(_audio_steps(he) + off, _audio_len(ta))
                at = _audio_steps(he) - _audio_steps(he - want)
                src1 = min(_audio_steps(sf0 + F), _audio_len(sa))
                at = max(0, min(at, a1, src1))
                a0 = a1 - at
                if at > 0:
                    ta = ta.clone()
                    ta[..., a0:a1] = sa[..., src1 - at:src1].to(device=ta.device, dtype=ta.dtype)
                    parts[ai] = ta
                    amask = masks[ai]
                    amask = torch.ones_like(ta) if amask.shape != ta.shape else amask.clone()
                    amask[..., a0:a1] = 0.0
                    _feather(amask, a0, at, audio_feather_ticks, from_end=False)
                    masks[ai] = amask
                    held_audio = at
        elif hold_audio:
            print(f"[{who}] no audio stream in one of the latents - video only")

        out = _pack(latent, parts, masks, packed, who)
        extra = (f", audio {held_audio} steps ({held_audio / AUDIO_HZ:.2f}s)"
                 + (f" with a {audio_feather_ticks}-step release" if audio_feather_ticks else "")
                 if held_audio else ", no audio held")
        print(f"[{who}] held latents {d0}..{d0 + n - 1} = frames {hs}..{hs + F - 1} "
              f"<- source frames {sf0}..{sf0 + F - 1} "
              f"(drop_last {BLOCK * kb}, drop_first {BLOCK * kf}){extra}")
        return (out,)


# --------------------------------------------------------------------------- anchoring (tail)



# --------------------------------------------------------------------------- tail refresh (latent only)

def _video_audio(latent, who):
    samples = latent["samples"]
    packed = NestedTensor is not None and isinstance(samples, NestedTensor)
    parts = _split(samples)
    vi = max(range(len(parts)), key=lambda i: parts[i].ndim)
    if parts[vi].ndim != 5:
        raise ValueError(f"{who}: expected a 5-D video latent [B, C, T, H, W], got {tuple(parts[vi].shape)}.")
    return parts, packed, vi, _audio_index(parts)


# --------------------------------------------------------------------------- reference lock


# --------------------------------------------------------------------------- color match to a chosen clip


# --------------------------------------------------------------------------- Clown Prophecy Sampler
























class H3ClownShotCounter:
    """Shot counter for the Clown Prophecy setup."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("shot_number", "previous_shot", "continuation")
    FUNCTION = "count"
    DESCRIPTION = ("classic: shot_number is the box, and it changes by itself: fixed = never, increment = +1 each run, "
                   "increment after accept = +1 after an accepted Prophecy run (one at a time only), "
                   "follow motion context = copies Motion Context Save Latent's clip_index each run (read only). "
                   "count saved latents: the shot is worked out when the run starts = highest saved clip in latent_prefix + 1. "
                   "A rejected Prophecy ending saves nothing, so a batch rerolls the same shot until one is accepted, then moves on. "
                   "Each new chain needs its own empty folder. "
                   "previous_shot = shot - 1, continuation = true from shot 2 on.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_number": ("INT", {"default": 1, "min": 1, "max": 99999}),
                "shot_counter": (["increment after accept", "increment", "decrement", "fixed", "follow motion context"],
                                 {"default": "increment after accept",
                                  "tooltip": "classic mode only. increment / decrement: +1 / -1 each time you hit run. increment after accept: +1 after an accepted Prophecy run. fixed: never. "
                                             "follow motion context: copies Motion Context Save Latent's clip_index each run (the Chain node's count); never changes it."}),
                "use_classic_counter": ("BOOLEAN", {"default": True, "label_on": "classic (the shot_number box)",
                                                    "label_off": "count saved latents",
                                                    "tooltip": "Off = shot = highest saved latent in latent_prefix + 1, worked out at run time. Works with batches."}),
                "latent_prefix": ("STRING", {"default": "h3_prophecy_test/clip",
                                             "tooltip": "count mode: the Save Latent filename_prefix (files <prefix>_00001.safetensors ...)."}),
            }
        }

    @staticmethod
    def _saved_shot(latent_prefix):
        import re, folder_paths
        out_dir = os.path.abspath(folder_paths.get_output_directory())
        full = os.path.abspath(os.path.join(out_dir, str(latent_prefix).strip()))
        if not full.startswith(out_dir):
            return 0
        folder, name = os.path.dirname(full), os.path.basename(full)
        if not os.path.isdir(folder):
            return 0
        pat = re.compile(re.escape(name) + r"_(\d{5})\.safetensors$")
        nums = [int(m.group(1)) for f in os.listdir(folder) for m in [pat.match(f)] if m]
        return max(nums) if nums else 0

    @classmethod
    def IS_CHANGED(cls, shot_number, shot_counter, use_classic_counter=True, latent_prefix=""):
        if use_classic_counter:
            return int(shot_number)
        return cls._saved_shot(latent_prefix) + 1

    def count(self, shot_number, shot_counter, use_classic_counter=True, latent_prefix=""):
        if use_classic_counter:
            shot = int(shot_number)
        else:
            last = self._saved_shot(latent_prefix)
            shot = last + 1
            print(f"[Clown Shot Counter] count saved latents: last saved clip {last} -> shot {shot}")
        return {"ui": {"clown_shot": [shot], "clown_classic": [bool(use_classic_counter)]},
                "result": (shot, max(0, shot - 1), shot > 1)}


class H3ClownForceAudio:
    """Puts a real sound track into an H3 AV latent so the sampler keeps it."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "force"
    DESCRIPTION = ("Encodes the audio with the H3 audio VAE and puts it in the latent's audio part, cut or padded "
                   "with silence to the shot length. audio_mask 0 = the sampler keeps it exactly (lip-sync to it), "
                   "1 = regenerate. Plug the result into Clown Prophecy Sampler's latent: Prophecy then uses this sound "
                   "and passes it on to the main sampler, also when it's bypassed or skipped.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "Reference To Video's LATENT output."}),
            "audio_vae": ("VAE", {"tooltip": "H3 audio VAE."}),
            "audio": ("AUDIO",),
            "audio_mask": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    def force(self, latent, audio_vae, audio, audio_mask):
        from comfy_extras.nodes_minimax_h3 import _encode_ref_audio
        who = "Clown Force Audio"
        parts, packed, vi, ai = _video_audio(latent, who)
        if ai is None or ai == vi:
            raise ValueError(f"{who}: the latent has no audio part (use an H3 AV latent).")
        a = parts[ai]
        t = int(a.shape[-1])
        z, _ = _encode_ref_audio(audio_vae, audio)
        z = z.to(device=a.device, dtype=a.dtype)
        if z.ndim == 3 and a.ndim == 4:
            z = z.unsqueeze(0)
        n = int(z.shape[-1])
        if n >= t:
            z = z[..., :t]
        else:
            z = torch.nn.functional.pad(z, (0, t - n))
        if z.shape != a.shape:
            raise ValueError(f"{who}: encoded audio {tuple(z.shape)} doesn't fit the latent's audio {tuple(a.shape)}.")
        parts = list(parts)
        parts[ai] = z
        masks = _carry_masks(latent, parts, packed, vi)
        masks[ai] = torch.full_like(z, float(audio_mask))
        print(f"[{who}] audio {n} steps ({n / AUDIO_HZ:.2f}s) into a {t}-step shot"
              + (f", padded with {t - n} steps of silence" if n < t else (f", cut by {n - t} steps" if n > t else ""))
              + f"; mask {audio_mask}")
        return (_pack(latent, parts, masks, True, who),)













# ---- live folder check for the Clown Shot Counter display
try:
    from server import PromptServer
    from aiohttp import web as _web

    @PromptServer.instance.routes.get("/clown/saved_shot")
    async def _clown_saved_shot(request):
        prefix = request.rel_url.query.get("prefix", "")
        try:
            last = H3ClownShotCounter._saved_shot(prefix)
            return _web.json_response({"last": int(last), "next": int(last) + 1})
        except Exception as e:
            return _web.json_response({"error": str(e)}, status=400)
except Exception as _e:
    print(f"[clown nodes] saved_shot route not registered: {_e}")




NODE_CLASS_MAPPINGS = {
    "H3ClownContextExperimental": H3ClownContextExperimental,
    "H3BlockTailAnchorExperimental": H3BlockTailAnchorExperimental,
    "H3ClownShotCounter": H3ClownShotCounter,
    "H3ClownForceAudio": H3ClownForceAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ClownContextExperimental": "Clown Context",
    "H3BlockTailAnchorExperimental": "Clown Anchoring",
    "H3ClownShotCounter": "Clown Shot Counter",
    "H3ClownForceAudio": "Clown Force Audio",
}
