"""Clowntext Hybrid Windows.

One long MiniMax H3 render in overlapping windows, done the HybridWindows way (JalenBrunson's ComfyUI-HybridWindows,
from the Rolling Diffusion paper), on top of kijai's MiniMax H3 context windows in ComfyUI (PR #17425,
comfy/context_windows.py, GPL-3). What this node adds is the keystone and capstone.

1. Window 1 runs the first switch_step steps. A snapshot is saved.
2. Window 1 carries on to the last step, only to get its finished frame 0 (the keystone) and its last single-frame
   latent (the capstone). That finish is thrown away and window 1 goes back to the snapshot.
3. Windows 2, 3... run the first switch_step steps in order. The overlap with the windows before is locked to those
   windows' clean guess at switch_step (finished_handoff: their finished frames). The keystone is locked into each
   window's first slot, and the capstone is tacked onto its back from capstone_step (and cut off again).
4. The rest of the steps run as kijai's blended windows over the whole video.

sound_first: the sound is rendered first on its own (blended windows on every step, picture at 64x64), then locked in,
so the main render only makes the picture.

The sampler itself is unchanged: it's re-run in short passes on the full latent, with everything outside the window
being worked on frozen, so memory use stays at one window.
"""
import copy
import logging

import torch

import comfy.conds
import comfy.context_windows as cw
import comfy.nested_tensor
import comfy.model_sampling
import comfy.patcher_extension
import comfy.utils

from .nodes import _split

CATEGORY = "Clown Context Suite"
WHO = "Clowntext Hybrid Windows"
HYBRID = "hybrid"


def _video_latent_t(frames):
    """H3 pixel frames (17n + 5) -> video latents (5n + 2)."""
    return ((int(frames) - 5) // 17) * 5 + 2


def _ix(dim, ids):
    return tuple([slice(None)] * dim + [list(ids)])


def _supported(base_model):
    return (hasattr(cw, "WindowingState") and hasattr(base_model, "map_context_window_to_modalities")
            and hasattr(base_model, "_window_payload") and hasattr(base_model, "_denoise_mask_values"))

class ClownContextHandler(cw.IndexListContextHandler):
    """kijai's handler, plus a mode that runs only one window (with its own keep mask) for the window-by-window passes."""

    def __init__(self, detail_swap=True, detail_latent=None, sound_first=False, **kwargs):
        super().__init__(**kwargs)
        self.schedule = HYBRID
        self.sound_first = bool(sound_first)
        self.sound_pass = False              # True while the 64x64 sound render runs (plain kijai windows)
        self.back_capstone = False
        self.capstone_step = 4
        self.capstone = None                 # window 1's capstone while a later window runs, else None
        self.detail_swap = bool(detail_swap)
        self.detail_latent = detail_latent   # optional [B, C, 1, H, W] (real latent space)
        self.switch_step = 6
        self.finished_handoff = False
        self.only_window = None              # None = normal blended windows, int = run only this window
        self.window_subset = None
        self.mask_conds = None               # model mask conds for the single-window pass

    def _with_masks(self, cond_list):
        if cond_list is None or self.mask_conds is None:
            return cond_list
        out = []
        for c in cond_list:
            c2 = c.copy()
            mc = dict(c2.get("model_conds", {}))
            for name in ("denoise_mask", "audio_denoise_mask"):
                if name in self.mask_conds:
                    mc[name] = comfy.conds.CONDRegular(self.mask_conds[name])
                else:
                    mc.pop(name, None)
            c2["model_conds"] = mc
            out.append(c2)
        return out

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        if self.only_window is None:
            return super().execute(calc_cond_batch, model, conds, x_in, timestep, model_options)
        self._model = model
        self.set_step(timestep, model_options)
        # the model reads its keep mask from the conds (per-token timestep), so it must match this pass's mask
        conds = [self._with_masks(c) for c in conds]
        ws = self._build_window_state(x_in, conds, model)
        windows = self.get_context_windows(model, ws.latents[0], model_options)
        wi = min(int(self.only_window), len(windows) - 1)
        res = self.evaluate_context_windows(calc_cond_batch, model, x_in, conds, timestep, [(wi, windows[wi])],
                                            model_options, window_state=ws, total_windows=len(windows))[0]
        out = []
        for ci in range(len(conds)):
            finalized = []
            for mi in range(len(ws.latents)):
                full = ws.latents[mi].clone()            # outside the window: denoised = x (frozen, masked anyway)
                mw = res.window.get_window_for_modality(mi)
                full[_ix(mw.dim, mw.index_list)] = res.sub_conds_out[ci][mi].to(full.dtype)
                if ws.guide_latents[mi] is not None:
                    full = torch.cat([full, ws.guide_latents[mi]], dim=self.dim)
                finalized.append(full)
            if ws.is_multimodal and len(finalized) > 1:
                out.append(comfy.utils.pack_latents(finalized)[0])
            else:
                out.append(finalized[0])
        return out


def _pass_args(extra_args, n):
    """Each restarted sampler pass gets its own seed. Samplers that add noise every step (er_sde, euler_ancestral...)
    seed their noise from extra_args["seed"], so without this every pass would add the exact same noise again and
    it would pile up on the later windows."""
    seed = extra_args.get("seed", None)
    if seed is None or n == 0:
        return extra_args
    args = dict(extra_args)
    args["seed"] = (int(seed) + 7919 * n) % (2 ** 63)
    return args

def _clown_sampler_sample(executor, guider, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None,
                          disable_pbar=False):
    mo = extra_args["model_options"]
    h = mo.get("context_handler", None)
    if not isinstance(h, ClownContextHandler):
        return executor(guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    model = guider.model_patcher.model
    conds = [guider.conds.get("positive", guider.conds.get("negative", []))]
    if h.freenoise:
        noise = h._apply_freenoise(noise, conds, extra_args["seed"], model)
    shapes = h._get_latent_shapes(conds) or [latent_image.shape]
    multi = len(shapes) > 1
    dims = model.context_window_modality_dims(shapes, h.dim) if multi else [h.dim]

    def unpack(t):
        return [p.clone() for p in comfy.utils.unpack_latents(t, shapes)] if multi else [t.clone()]

    def pack(parts):
        return comfy.utils.pack_latents(parts)[0] if multi else parts[0]

    total = shapes[0][h.dim]
    windows = cw.create_windows_static_standard(total, h, mo)
    steps = len(sigmas) - 1
    h.only_window, h.mask_conds = None, None
    if h.sound_pass or len(windows) < 2 or steps < 1:
        return executor(guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    return _hybrid(executor, guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar,
                   h, model, shapes, multi, dims, unpack, pack, windows)


def _hybrid(executor, guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar,
            h, model, shapes, multi, dims, unpack, pack, windows):
    """Clowntext Hybrid Windows (like JalenBrunson's HybridWindows / Rolling Diffusion, plus keystone and capstone).
    1. Window 1 runs steps 1..switch_step. Snapshot.
    2. Window 1 carries on to the end from the snapshot, only to get its finished frame 0 (keystone) and its last
       single-frame latent (capstone). That finish is thrown away and window 1 goes back to the snapshot.
    3. Windows 2, 3... run steps 1..switch_step in order. Their overlap with the windows before is pinned (locked) to
       those windows' clean guess from their last early step; the keystone goes in the window's first slot, the
       capstone is tacked onto the back from capstone_step. Each position keeps the state of the first window that made it.
    4. The rest of the steps: kijai's blended windows over everything.
    finished_handoff: every window (not just window 1) is run to the last step, from a snapshot saved at switch_step,
    and the next window's overlap is pinned to those finished frames instead of a clean guess. The finished frames are
    only for handing over: every window goes back to its switch_step snapshot before the blended finish."""
    ms = model.model_sampling
    if not isinstance(ms, comfy.model_sampling.CONST):
        raise RuntimeError(f"{WHO}: only flow models (like MiniMax H3) are supported.")
    steps = len(sigmas) - 1
    k = max(1, min(int(h.switch_step), steps))
    nscale = float(getattr(ms, "noise_scale", 1.0))
    base_parts = unpack(denoise_mask if denoise_mask is not None else torch.ones_like(latent_image))
    clean_parts = unpack(latent_image)
    st = unpack(ms.noise_scaling(sigmas[0], noise, latent_image))
    pins = [p.clone() for p in clean_parts]             # clean guess for each position, from the window that made it
    done = [set() for _ in shapes]
    per_window = [model.map_context_window_to_modalities(w, shapes, h.dim) if multi else [w] for w in windows]
    detail = None
    if h.detail_latent is not None:
        d = model.latent_format.process_in(h.detail_latent.float())
        v = st[0]
        if d.shape[1] == v.shape[1] and d.shape[3:] == v.shape[3:]:
            detail = d.to(device=v.device, dtype=v.dtype).expand(v.shape[0], -1, -1, -1, -1)
    handoff = bool(getattr(h, "finished_handoff", False)) and k < steps
    need_key = (h.detail_swap and detail is None) or h.back_capstone
    if handoff:
        extra = len(windows) * (steps - k)
    else:
        extra = (steps - k) if (k < steps and need_key) else 0
    total_all = len(windows) * k + extra + (steps - k)
    counter = {"n": 0, "pass": 0}
    guess = {}

    def progress():
        offset = counter["n"]

        def cb(step, x0, x, total):
            guess["x0"] = x0
            if callback is not None:
                callback(offset + step, x0, x, total_all)
        return cb

    def one_window(sig, wi, state, slot=None, slot_src=None):
        """Run window wi on sig from `state`. Pinned positions (done) are locked to their clean guess."""
        act = per_window[wi]
        li = [p.clone() for p in state]
        m = [torch.zeros_like(p) for p in base_parts]
        for mi in range(len(shapes)):
            new = [i for i in act[mi] if i not in done[mi]]
            pinned = [i for i in act[mi] if i in done[mi]]
            if new:
                sl = _ix(dims[mi], new)
                li[mi][sl] = clean_parts[mi][sl]
                m[mi][sl] = base_parts[mi][sl]
            if pinned:
                sl = _ix(dims[mi], pinned)
                li[mi][sl] = pins[mi][sl]
        if slot is not None:
            li[0][:, :, slot:slot + 1] = slot_src.to(li[0].dtype)
            m[0][:, :, slot:slot + 1] = 0.0
        li_p, m_p = pack(li), pack(m)
        s0 = float(sig[0])
        noise_w = (pack(state) - (1.0 - s0) * li_p) / (s0 * nscale)
        h.only_window, h.window_subset = wi, None
        h.mask_conds = model._denoise_mask_values(m_p, shapes) if multi else {}
        guess.pop("x0", None)
        try:
            out = executor(guider, sig, _pass_args(extra_args, counter["pass"]), progress(), noise_w, li_p, m_p,
                           disable_pbar)
        finally:
            h.only_window, h.window_subset, h.mask_conds = None, None, None
        counter["n"] += len(sig) - 1
        counter["pass"] += 1
        x0 = unpack(guess["x0"]) if "x0" in guess else None
        return unpack(out * (1.0 - float(sig[-1]))), x0

    def keep(wi, out, x0):
        for mi in range(len(shapes)):
            new = [i for i in per_window[wi][mi] if i not in done[mi]]
            if new:
                sl = _ix(dims[mi], new)
                st[mi][sl] = out[mi][sl].to(st[mi].dtype)
                if x0 is not None:
                    pins[mi][sl] = x0[mi][sl].to(pins[mi].dtype)
                done[mi].update(new)

    key_src = cap_src = None
    try:
        for wi, idxs in enumerate(windows):
            h.capstone, slot, note = None, None, ""
            if wi > 0:
                if h.detail_swap and key_src is not None and idxs[0] in done[0]:
                    slot = idxs[0]
                    note += f", keystone in latent {slot}"
                if h.back_capstone and cap_src is not None:
                    h.capstone = cap_src
                    note += f", capstone on the back from step {h.capstone_step + 1}"
            n_pin = sum(1 for i in idxs if i in done[0])
            logging.info(f"[{WHO}] hybrid: window {wi + 1} steps 1-{k}"
                         + (f", {n_pin} overlap latents pinned to the windows before"
                            + (" (finished frames)" if handoff else "") if n_pin else "") + note)
            out, x0 = one_window(sigmas[0:k + 1], wi, st, slot, key_src)
            if handoff:
                # run this window on to the end from its step-k snapshot, same pins / keystone / capstone; the
                # finished frames are what the next window's overlap gets pinned to
                logging.info(f"[{WHO}] hybrid: window {wi + 1} finished on steps {k + 1}-{steps} for the handoff, "
                             f"then rewound to step {k}")
                snap = [p.clone() for p in st]
                for mi in range(len(shapes)):
                    sl = _ix(dims[mi], per_window[wi][mi])
                    snap[mi][sl] = out[mi][sl].to(snap[mi].dtype)
                fin, _ = one_window(sigmas[k:], wi, snap, slot, key_src)
                if wi == 0:
                    k0 = idxs[0]
                    key_src = detail if detail is not None else fin[0][:, :, k0:k0 + 1].clone()
                    ci = idxs[5 * ((len(idxs) - 1) // 5)]
                    cap_src = fin[0][:, :, ci:ci + 1].clone()
                keep(wi, out, fin)
                continue
            if wi == 0:
                if k < steps and need_key:
                    # carry window 1 on to the end, only for its finished keystone / capstone, then go back
                    logging.info(f"[{WHO}] hybrid: window 1 finished on steps {k + 1}-{steps} for the keystone / "
                                 f"capstone, then rewound to step {k}")
                    snap = [p.clone() for p in st]
                    for mi in range(len(shapes)):
                        sl = _ix(dims[mi], per_window[0][mi])
                        snap[mi][sl] = out[mi][sl].to(snap[mi].dtype)
                    fin, _ = one_window(sigmas[k:], 0, snap)
                else:
                    fin = out
                k0 = idxs[0]
                key_src = detail if detail is not None else fin[0][:, :, k0:k0 + 1].clone()
                ci = idxs[5 * ((len(idxs) - 1) // 5)]
                cap_src = fin[0][:, :, ci:ci + 1].clone()
            keep(wi, out, x0)
        h.capstone = None
        if k >= steps:
            return pack(st) / max(1.0 - float(sigmas[-1]), 1e-8)
        logging.info(f"[{WHO}] hybrid: steps {k + 1}-{steps} blended over all {len(windows)} windows")
        li_p, m_p = pack(clean_parts), pack(base_parts)
        sk = float(sigmas[k])
        noise_w = (pack(st) - (1.0 - sk) * li_p) / (sk * nscale)
        return executor(guider, sigmas[k:], _pass_args(extra_args, counter["pass"]), progress(), noise_w, li_p, m_p,
                        disable_pbar)
    finally:
        h.capstone = None
        h.only_window, h.window_subset, h.mask_conds = None, None, None


def _make_capstone_wrapper(h, prev):
    """model_function_wrapper: while a later window finishes, tack window 1's capstone onto the back of the window
    (locked) from capstone_step on, and cut it off the output again."""
    def wrapper(apply_model, args):
        x, t, c = args["input"], args["timestep"], args["c"]

        def run(xx, cc):
            if prev is None:
                return apply_model(xx, t, **cc)
            return prev(apply_model, {"input": xx, "timestep": t, "c": cc, "cond_or_uncond": args.get("cond_or_uncond")})

        cap = h.capstone
        shapes = c.get("latent_shapes", None)
        if cap is None or shapes is None or len(shapes) < 2:
            return run(x, c)
        to = c.get("transformer_options", {}) or {}
        sigmas = to.get("sample_sigmas", None)
        if sigmas is not None:
            step = int(torch.argmin((sigmas.float().cpu() - float(t.flatten()[0])).abs()))
            if step < h.capstone_step:
                return run(x, c)
        parts = comfy.utils.unpack_latents(x, shapes)
        v = parts[0]
        if v.ndim != 5 or cap.shape[1] != v.shape[1] or cap.shape[3:] != v.shape[3:]:
            return run(x, c)
        n = v.shape[2]
        d = cap.to(device=v.device, dtype=v.dtype).expand(v.shape[0], -1, -1, -1, -1)
        x2, shapes2 = comfy.utils.pack_latents([torch.cat([v, d], dim=2)] + list(parts[1:]))
        c2 = dict(c)
        c2["latent_shapes"] = shapes2
        dm = c.get("denoise_mask", None)
        if dm is None:
            dm = torch.ones((v.shape[0], 1, n) + tuple(v.shape[3:]), device=v.device, dtype=torch.float32)
        k = int(cap.shape[2])
        c2["denoise_mask"] = torch.cat([dm, torch.zeros_like(dm[:, :, :1]).expand(-1, -1, k, -1, -1)], dim=2)   # locked
        out = comfy.utils.unpack_latents(run(x2, c2), shapes2)
        return comfy.utils.pack_latents([out[0][:, :, :n]] + list(out[1:]))[0]
    return wrapper


SOUND_SIZE = 4   # latent pixels: 64 / 16


def _clown_outer_sample(executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None,
                        disable_pbar=False, seed=None, latent_shapes=None):
    guider = executor.class_obj
    h = guider.model_options.get("context_handler", None)
    if (not isinstance(h, ClownContextHandler) or not h.sound_first or h.sound_pass
            or latent_shapes is None or len(latent_shapes) < 2):
        return executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed,
                        latent_shapes=latent_shapes)
    nv, na = comfy.utils.unpack_latents(noise, latent_shapes)[:2]
    lv, la = comfy.utils.unpack_latents(latent_image, latent_shapes)[:2]
    if denoise_mask is not None:
        mv, ma = comfy.utils.unpack_latents(denoise_mask, latent_shapes)[:2]
    else:
        mv, ma = torch.ones_like(lv), torch.ones_like(la)

    # ---- sound render: picture at 64x64, every step kijai's way, held sound kept ----
    s = SOUND_SIZE
    small_shape = tuple(lv.shape[:3]) + (s, s)
    small_noise = torch.zeros(small_shape, dtype=nv.dtype, device=nv.device)
    hh, ww = min(s, nv.shape[3]), min(s, nv.shape[4])
    small_noise[:, :, :, :hh, :ww] = nv[:, :, :, :hh, :ww]
    small_lat = torch.zeros(small_shape, dtype=lv.dtype, device=lv.device)
    small_mask = torch.ones(small_shape, dtype=mv.dtype, device=mv.device)
    NT = comfy.nested_tensor.NestedTensor
    steps = len(sigmas) - 1
    pbar = comfy.utils.ProgressBar(steps)

    def cb(step, x0, x, total):          # progress bar only, no preview picture
        pbar.update_absolute(step + 1, total, None)

    g2 = copy.copy(guider)
    logging.info(f"[{WHO}] rendering the sound first (picture at 64x64)")
    h.sound_pass = True
    try:
        out = g2.sample(NT([small_noise, na.clone()]), NT([small_lat, la.clone()]), sampler, sigmas,
                        denoise_mask=NT([small_mask, ma.clone()]), callback=cb, disable_pbar=disable_pbar, seed=seed)
    finally:
        h.sound_pass = False
    sound = out.unbind()[1]

    # ---- main render: the finished sound locked in, only the picture is made ----
    li2 = comfy.utils.pack_latents([lv, sound.to(device=la.device, dtype=la.dtype)])[0]
    dev = denoise_mask.device if denoise_mask is not None else guider.model_patcher.load_device
    m2 = comfy.utils.pack_latents([mv.to(dev).float(), torch.zeros_like(ma).to(dev).float()])[0]
    logging.info(f"[{WHO}] sound done, locked in for the picture")
    return executor(noise, li2, sampler, sigmas, m2, callback, disable_pbar, seed, latent_shapes=latent_shapes)


class H3ClownHybridWindows:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = ("MiniMax H3 hybrid windows (like JalenBrunson's HybridWindows): each window runs the first "
                   "switch_step steps in order, its overlap pinned to the windows before it, then the rest of the steps "
                   "are blended windows. Window 1 is carried on to the end once for a keystone (its frame 0) and a "
                   "capstone (its last single-frame latent), then rewound. Needs a ComfyUI with MiniMax H3 context windows (PR #17425).")

    @classmethod
    def INPUT_TYPES(cls):
        fuse = cw.ContextFuseMethods.LIST_STATIC
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": ("INT", {"default": 124, "min": 22, "max": 16384, "step": 17,
                                           "tooltip": "Window length in frames (17n + 5)."}),
                "context_overlap": ("INT", {"default": 39, "min": 5, "max": 16384, "step": 17,
                                            "tooltip": "Overlap in frames (17n + 5)."}),
                "switch_step": ("INT", {"default": 6, "min": 1, "max": 200,
                                        "tooltip": "Steps each window runs on its own, in order, before everything is "
                                                   "blended. 6 of 8 = steps 1-6 window by window, 7-8 blended."}),
                "keystone_swap": ("BOOLEAN", {"default": True,
                                              "tooltip": "Window 1's finished frame 0 (or detail_latent's) is locked into "
                                                         "the first slot of every later window during its own steps."}),
                "back_capstone": ("BOOLEAN", {"default": True,
                                              "tooltip": "Window 1's finished last single-frame latent is tacked onto the "
                                                         "back of every later window (locked) from capstone_step, and cut "
                                                         "off again."}),
                "capstone_step": ("INT", {"default": 4, "min": 0, "max": 200,
                                          "tooltip": "Steps before the capstone is tacked on. 4 with switch_step 6 = "
                                                     "on for steps 5-6."}),
                "fuse_method": (fuse, {"default": cw.ContextFuseMethods.PYRAMID,
                                       "tooltip": "How the blended steps average the windows."}),
                "freenoise": ("BOOLEAN", {"default": True}),
                "split_conds_to_windows": ("BOOLEAN", {"default": False,
                                                       "tooltip": "Give each window its own prompt (combine prompts "
                                                                  "with Conditioning Combine)."}),
            },
            "optional": {
                "sound_first": ("BOOLEAN", {"default": True,
                                            "tooltip": "Render the sound first on its own (blended windows, picture at "
                                                       "64x64), then lock it in and render the picture."}),
                "detail_latent": ("LATENT", {"tooltip": "Optional: use this latent's first latent as the keystone "
                                                        "instead of window 1's finished frame 0."}),
                "finished_handoff": ("BOOLEAN", {"default": False,
                                                 "tooltip": "On: every window is run all the way to the last step, and "
                                                            "the next window's overlap is locked to those finished frames "
                                                            "(like context motion). Each window is saved at switch_step "
                                                            "and goes back to that save before the blended finish. Costs "
                                                            "the last steps once more per window. Off: the overlap is "
                                                            "locked to the window before's clean guess at switch_step."}),
            },
        }

    def patch(self, model, context_length=124, context_overlap=39, switch_step=6, keystone_swap=True,
              back_capstone=True, capstone_step=4, fuse_method=cw.ContextFuseMethods.PYRAMID, freenoise=True,
              split_conds_to_windows=False, sound_first=True, detail_latent=None, finished_handoff=False):
        if not _supported(model.model):
            raise RuntimeError(f"{WHO} needs a newer ComfyUI with MiniMax H3 context windows (PR #17425). Update ComfyUI.")
        length = _video_latent_t(context_length)
        overlap = min(_video_latent_t(context_overlap), length - 5)
        detail = None
        if detail_latent is not None and isinstance(detail_latent, dict) and detail_latent.get("samples") is not None:
            s = detail_latent["samples"]
            if isinstance(s, (list, tuple)):
                s = [p for p in s if p is not None]
            parts = _split(s)
            src = max(parts, key=lambda p: p.ndim)
            if src.ndim == 5:
                detail = src[:, :, 0:1].clone()
        handler = ClownContextHandler(
            detail_swap=keystone_swap, detail_latent=detail, sound_first=sound_first,
            context_schedule=cw.get_matching_context_schedule(cw.ContextSchedules.STATIC_STANDARD),
            fuse_method=cw.get_matching_fuse_method(fuse_method),
            context_length=length, context_overlap=overlap, context_stride=1, closed_loop=False, dim=2,
            freenoise=freenoise, cond_retain_index_list="", split_conds_to_windows=split_conds_to_windows,
            latent_retain_index_list="", causal_window_fix=False)
        handler.back_capstone = bool(back_capstone)
        handler.capstone_step = int(capstone_step)
        handler.switch_step = int(switch_step)
        handler.finished_handoff = bool(finished_handoff)
        m = model.clone()
        m.model_options["context_handler"] = handler
        if handler.back_capstone:
            m.set_model_unet_function_wrapper(_make_capstone_wrapper(handler, m.model_options.get("model_function_wrapper")))
        cw.create_prepare_sampling_wrapper(m)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, "ClownHybridWindows_sampler_sample",
                               _clown_sampler_sample)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "ClownHybridWindows_outer_sample",
                               _clown_outer_sample)
        print(f"[{WHO}] window {length} latents, overlap {overlap}: steps 1-{switch_step} window by window, then blended"
              + (", finished handoff" if finished_handoff else "") + (", keystone" if keystone_swap else "")
              + (f", capstone from step {int(capstone_step) + 1}" if back_capstone else "")
              + (", sound first" if sound_first else ""))
        return (m,)


NODE_CLASS_MAPPINGS = {"H3ClownHybridWindows": H3ClownHybridWindows}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownHybridWindows": "Clowntext Hybrid Windows"}
