"""Geiru Hybrid Windows (was Clowntext Hybrid Windows).

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

context_noise (off by default): ethanfel's (Lumifel's) context noise on each window's pinned overlap during its own
steps: part of the current noise, fading to none at the join, instead of locking the overlap fully clean.

sound_first (off by default): the sound is rendered first on its own (blended windows on every step, picture at 64x64), then locked in,
so the main render only makes the picture.

The sampler itself is unchanged: it's re-run in short passes on the full latent, with everything outside the window
being worked on frozen, so memory use stays at one window.
"""
import copy
import logging
import os

import torch

import comfy.conds
import comfy.context_windows as cw
import comfy.nested_tensor
import comfy.model_sampling
import comfy.patcher_extension
import comfy.utils

from .nodes import _split

CATEGORY = "Geiru Context Suite"
WHO = "Geiru Hybrid Windows"
PROGRESS_DIR = "geiru_hybrid_progress"     # in ComfyUI's output folder
PROGRESS_FILE = "crash_protection.pt"


def _free_vram():
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _progress_path():
    try:
        import folder_paths
        base = folder_paths.get_output_directory()
    except Exception:
        base = "output"
    d = os.path.join(base, PROGRESS_DIR)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, PROGRESS_FILE)


def _cond_sig(conds):
    """A fingerprint of the prompts (the conditioning tensors), so a changed prompt is never picked up as the same run."""
    sig = []
    for c in (conds[0] if conds else []) or []:
        mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
        for name in sorted(mc):
            t = getattr(mc[name], "cond", None)
            if torch.is_tensor(t):
                f = t.detach().float()
                flat = f.flatten()
                step = max(1, flat.numel() // 64)
                sig.append([name, list(map(int, f.shape)), round(float(f.sum()), 3), round(float(f.abs().sum()), 3),
                            [round(float(v), 4) for v in flat[::step][:64].cpu()]])
    return sig


def _run_id(extra_args, shapes, sigmas, windows, k, h):
    """What a saved run must match to be picked up again: seed, prompts, latent size, schedule, windows and settings."""
    return {"seed": int(extra_args.get("seed", 0) or 0), "conds": getattr(h, "cond_sig", None),
            "shapes": [list(map(int, sh)) for sh in shapes],
            "sigmas": [round(float(x), 5) for x in sigmas], "windows": [list(map(int, w)) for w in windows],
            "k": int(k), "keystone": bool(h.detail_swap), "capstone": bool(h.back_capstone),
            "handoff": bool(getattr(h, "finished_handoff", False)), "noise": float(h.context_noise),
            "fade": str(h.context_noise_fade),
            **({"keystone_step": int(h.keystone_step)} if int(getattr(h, "keystone_step", 0)) > 0 else {})}


def _cpu(t):
    return None if t is None else t.detach().to("cpu")


def _save_progress(run_id, **data):
    path = _progress_path()
    tmp = path + ".tmp"
    torch.save({"run_id": run_id, **data}, tmp)
    os.replace(tmp, path)
    return path


def _load_progress(run_id):
    """The saved progress of this exact run if it crashed partway, else None."""
    path = _progress_path()
    if not os.path.isfile(path):
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)   # tensors, lists and numbers only; never runs code from the file
    except Exception as e:
        logging.warning(f"[{WHO}] crash protection: couldn't read the saved progress ({e}) - starting from the beginning")
        return None
    if data.get("run_id") != run_id:
        logging.info(f"[{WHO}] crash protection: the saved progress is from a different run - starting from the beginning")
        return None
    return data


def _clear_progress():
    path = _progress_path()
    for f in (path, path + ".tmp"):
        try:
            if os.path.isfile(f):
                os.remove(f)
        except Exception as e:
            logging.warning(f"[{WHO}] crash protection: couldn't remove {f}: {e}")
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
        self.keystone_step = 0               # steps before the keystone is locked in (0 = from the first step)
        self.capstone = None                 # window 1's capstone while a later window runs, else None
        self.detail_swap = bool(detail_swap)
        self.detail_latent = detail_latent   # optional [B, C, 1, H, W] (real latent space)
        self.switch_step = 6
        self.finished_handoff = False
        self.crash_protection = True         # save after every window; pick a crashed run back up; delete on finish
        self.cond_sig = None                 # fingerprint of the prompts, set per sampling run
        self.only_window = None              # None = normal blended windows, int = run only this window
        self.window_subset = None
        self.mask_conds = None               # model mask conds for the single-window pass
        self.context_noise = 0.0             # ethanfel's context noise on the pinned overlap (0 = off)
        self.context_noise_fade = "linear"   # linear / ease / ethanfel

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


FADES = ["linear", "ease", "ethanfel"]


def _fade_weight(j, n, fade):
    """Weight of the j-th of n pinned overlap latents (0 = start of the overlap, n-1 = next to the join)."""
    if n <= 1:
        return 0.0
    if fade == "ethanfel":                           # ethanfel's drift_control: full, then .75/.5/.25/0 over the last 4
        k = n - 1 - j
        return 1.0 if k >= 4 else k / 4.0
    pos = j / (n - 1)
    return (1.0 - pos) if fade != "ease" else (1.0 - pos) ** 2


def _ratio_fn(prev, ramp, dim, unpack, pack, shape):
    """ethanfel's matched_noise_ratio: at every step the noised overlap values are scaled by next sigma / current sigma."""
    def fn(sigma, mask, extra_options):
        if prev is not None:
            mask = prev(sigma, mask, extra_options)
        sigmas = extra_options.get("sigmas")
        if sigmas is None or tuple(mask.shape) != tuple(shape):
            return mask
        sg = sigmas.float().cpu()
        sv = float(torch.as_tensor(sigma).flatten()[0])
        i = int(torch.argmin((sg - sv).abs()))
        nxt = float(sg[i + 1]) if i + 1 < len(sg) else 0.0
        r = max(0.0, min(1.0, nxt / sv)) if sv > 0 else 0.0
        parts = unpack(mask)
        for t, v in ramp:
            parts[0][_ix(dim, [t])] = v * r
        return pack(parts)
    return fn


def _noise_ramp(mask, pinned, new, strength, dim, fade="linear"):
    """ethanfel's (Lumifel's) context noise on a window's pinned overlap: instead of locking the overlap fully clean, give
    it part of the current noise, most at the start of the overlap and fading to none at the join with the new latents.
    The sampler and the model both read the partial mask (a value m = m x the current noise level). Window positions
    already made by an earlier window are never changed by this window, so the real overlap is kept."""
    pinned = sorted(pinned)
    if not pinned:
        return []
    join = min(new) if new else pinned[-1] + 1
    before = [i for i in pinned if i < join]          # the run of pinned latents leading up to the new ones
    if not before:
        return []
    out = []
    n = len(before)
    for j, i in enumerate(before):
        w = _fade_weight(j, n, fade)                    # 1 at the start of the overlap, 0 at the join
        v = float(strength) * w
        if v <= 0:
            continue
        mask[_ix(dim, [i])] = v
        out.append((i, round(v, 3)))
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
    try:
        h.cond_sig = _cond_sig(conds)
    except Exception:
        h.cond_sig = None
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
    """Geiru Hybrid Windows (like JalenBrunson's HybridWindows / Rolling Diffusion, plus keystone and capstone).
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
        extra = (len(windows) - 1) * (steps - k)       # the last window has no next window to hand off to
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
        ramp = []
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
                if mi == 0 and h.context_noise > 0:
                    ramp.extend(_noise_ramp(m[mi], pinned, new, h.context_noise, dims[mi], h.context_noise_fade))
        if slot is not None:
            li[0][:, :, slot:slot + 1] = slot_src.to(li[0].dtype)
            m[0][:, :, slot:slot + 1] = 0.0
            ramp = [(i, v) for i, v in ramp if i != slot]     # the keystone slot stays locked and clean
        li_p, m_p = pack(li), pack(m)
        s0 = float(sig[0])
        noise_w = (pack(state) - (1.0 - s0) * li_p) / (s0 * nscale)
        if ramp:
            # the noised overlap latents need real noise, like any other noisy latent
            nw = unpack(noise_w)
            g = torch.Generator(device="cpu").manual_seed(int(extra_args.get("seed", 0) or 0) + 104729 * (wi + 1))
            sl = _ix(dims[0], [i for i, _ in ramp])
            nw[0][sl] = torch.randn(nw[0][sl].shape, generator=g).to(device=nw[0].device, dtype=nw[0].dtype)
            noise_w = pack(nw)
            if counter.get("noted") != wi:
                counter["noted"] = wi
                logging.info(f"[{WHO}] context noise on window {wi + 1}'s overlap: "
                             + ", ".join(f"latent {i} {v:.2f}" for i, v in ramp))
        h.only_window, h.window_subset = wi, None
        h.mask_conds = model._denoise_mask_values(m_p, shapes) if multi else {}
        guess.pop("x0", None)
        args = _pass_args(extra_args, counter["pass"])
        if ramp and h.context_noise_fade == "ethanfel":
            args = dict(args)
            mo2 = dict(args["model_options"])
            mo2["denoise_mask_function"] = _ratio_fn(mo2.get("denoise_mask_function"), ramp, dims[0], unpack, pack,
                                                     m_p.shape)
            args["model_options"] = mo2
        try:
            out = executor(guider, sig, args, progress(), noise_w, li_p, m_p,
                           disable_pbar)
        finally:
            h.only_window, h.window_subset, h.mask_conds = None, None, None
        counter["n"] += len(sig) - 1
        counter["pass"] += 1
        x0 = unpack(guess["x0"]) if "x0" in guess else None
        return unpack(out * (1.0 - float(sig[-1]))), x0

    def run_win(a, b, wi, state, slot=None, slot_src=None):
        """Run window wi on steps a+1..b. The keystone (slot) is only locked in from keystone_step on: before that,
        its slot is treated like the rest of the overlap."""
        ks = int(getattr(h, "keystone_step", 0))
        if slot is None or ks <= a:
            return one_window(sigmas[a:b + 1], wi, state, slot, slot_src)
        if ks >= b:
            return one_window(sigmas[a:b + 1], wi, state)
        out, _ = one_window(sigmas[a:ks + 1], wi, state)
        mid = [p.clone() for p in state]
        for mi in range(len(shapes)):
            sl = _ix(dims[mi], per_window[wi][mi])
            mid[mi][sl] = out[mi][sl].to(mid[mi].dtype)
        del out
        return one_window(sigmas[ks:b + 1], wi, mid, slot, slot_src)

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
    run_id = _run_id(extra_args, shapes, sigmas, windows, k, h)
    start = 0
    if h.crash_protection:
        saved = _load_progress(run_id)
        if saved is not None:
            dev = st[0].device
            st = [t.to(device=dev, dtype=p.dtype) for t, p in zip(saved["st"], st)]
            pins = [t.to(device=dev, dtype=p.dtype) for t, p in zip(saved["pins"], pins)]
            done = [set(d) for d in saved["done"]]
            key_src = None if saved["key_src"] is None else saved["key_src"].to(device=dev, dtype=st[0].dtype)
            cap_src = None if saved["cap_src"] is None else saved["cap_src"].to(device=dev, dtype=st[0].dtype)
            start = int(saved["next_window"])
            counter["pass"] = int(saved.get("pass", 0))
            logging.info(f"[{WHO}] crash protection: picking up the crashed run from window {start + 1} of "
                         f"{len(windows)}" if start < len(windows) else
                         f"[{WHO}] crash protection: picking up the crashed run - all windows done, going to the "
                         f"blended finish")

    def checkpoint(next_window):
        if not h.crash_protection:
            return
        try:
            path = _save_progress(run_id, st=[_cpu(t) for t in st], pins=[_cpu(t) for t in pins],
                                  done=[sorted(d) for d in done], key_src=_cpu(key_src), cap_src=_cpu(cap_src),
                                  next_window=next_window, **{"pass": counter["pass"]})
            logging.info(f"[{WHO}] crash protection: progress saved after window {next_window} ({path})")
        except Exception as e:
            logging.warning(f"[{WHO}] crash protection: couldn't save progress: {e}")

    try:
        for wi, idxs in enumerate(windows):
            if wi < start:
                continue
            h.capstone, slot, note = None, None, ""
            if wi > 0:
                if h.detail_swap and key_src is not None and idxs[0] in done[0]:
                    slot = idxs[0]
                    note += f", keystone in latent {slot}" + (f" from step {h.keystone_step + 1}"
                                                                    if h.keystone_step > 0 else "")
                if h.back_capstone and cap_src is not None:
                    h.capstone = cap_src
                    note += f", capstone on the back from step {h.capstone_step + 1}"
            n_pin = sum(1 for i in idxs if i in done[0])
            logging.info(f"[{WHO}] hybrid: window {wi + 1} steps 1-{k}"
                         + (f", {n_pin} overlap latents pinned to the windows before"
                            + (" (finished frames)" if handoff else "") if n_pin else "") + note)
            out, x0 = run_win(0, k, wi, st, slot, key_src)
            if handoff and wi < len(windows) - 1:
                # run this window on to the end from its step-k snapshot, same pins / keystone / capstone; the
                # finished frames are what the next window's overlap gets pinned to
                logging.info(f"[{WHO}] hybrid: window {wi + 1} finished on steps {k + 1}-{steps} for the handoff, "
                             f"then rewound to step {k}")
                snap = [p.clone() for p in st]
                for mi in range(len(shapes)):
                    sl = _ix(dims[mi], per_window[wi][mi])
                    snap[mi][sl] = out[mi][sl].to(snap[mi].dtype)
                fin, _ = run_win(k, steps, wi, snap, slot, key_src)
                if wi == 0:
                    k0 = idxs[0]
                    key_src = detail if detail is not None else fin[0][:, :, k0:k0 + 1].clone()
                    ci = idxs[5 * ((len(idxs) - 1) // 5)]
                    cap_src = fin[0][:, :, ci:ci + 1].clone()
                keep(wi, out, fin)
                del out, fin, snap
                _free_vram()
                checkpoint(wi + 1)
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
            del out, x0
            _free_vram()
            checkpoint(wi + 1)
        h.capstone = None
        if k >= steps:
            result = pack(st) / max(1.0 - float(sigmas[-1]), 1e-8)
        else:
            logging.info(f"[{WHO}] hybrid: steps {k + 1}-{steps} blended over all {len(windows)} windows")
            li_p, m_p = pack(clean_parts), pack(base_parts)
            sk = float(sigmas[k])
            noise_w = (pack(st) - (1.0 - sk) * li_p) / (sk * nscale)
            _free_vram()
            result = executor(guider, sigmas[k:], _pass_args(extra_args, counter["pass"]), progress(), noise_w, li_p,
                              m_p, disable_pbar)
        if h.crash_protection:
            _clear_progress()
            logging.info(f"[{WHO}] crash protection: run finished, saved progress removed")
        return result
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
                "fuse_method": (fuse, {"default": cw.ContextFuseMethods.PYRAMID,
                                       "tooltip": "How the blended steps average the windows."}),
                "freenoise": ("BOOLEAN", {"default": True}),
                "split_conds_to_windows": ("BOOLEAN", {"default": False,
                                                       "tooltip": "Give each window its own prompt (combine prompts "
                                                                  "with Conditioning Combine)."}),
            },
            "optional": {
                "detail_latent": ("LATENT", {"tooltip": "Optional: use this latent's first latent as the keystone "
                                                        "instead of window 1's finished frame 0."}),
                "crash_protection": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Saves the progress after every window to "
                                                            "output/geiru_hybrid_progress/crash_protection.pt and clears "
                                                            "unused GPU memory between windows. If ComfyUI crashes or runs "
                                                            "out of memory, press Run again on the same run (same seed, "
                                                            "prompts, size, length and settings) and it picks up from the "
                                                            "last saved window. The file is deleted when the run finishes. "
                                                            "It can't see a changed LoRA, turbo or sampler: change the seed "
                                                            "if you change those after a crash."}),
                # advanced: tucked away until "show advanced inputs" is opened on the node
                "keystone_swap": ("BOOLEAN", {"default": False,
                                              "tooltip": "Window 1's finished frame 0 (or detail_latent's) is locked into "
                                                         "the first slot of every later window during its own steps.",
                    "advanced": True}),
                "keystone_step": ("INT", {"default": 0, "min": 0, "max": 200,
                                          "tooltip": "Steps before the keystone is locked in (late lock). 0 = locked from the first step (the usual keystone). 2 with switch_step 6 = locked for steps 3-6.",
                    "advanced": True}),
                "back_capstone": ("BOOLEAN", {"default": False,
                                              "tooltip": "Window 1's finished last single-frame latent is tacked onto the "
                                                         "back of every later window (locked) from capstone_step, and cut "
                                                         "off again.",
                    "advanced": True}),
                "capstone_step": ("INT", {"default": 4, "min": 0, "max": 200,
                                          "tooltip": "Steps before the capstone is tacked on. 4 with switch_step 6 = "
                                                     "on for steps 5-6.",
                    "advanced": True}),
                "context_noise": ("BOOLEAN", {"default": False,
                                              "tooltip": "ethanfel's context noise: during each window's own steps, its "
                                                         "pinned overlap gets part of the current noise instead of being "
                                                         "locked fully clean, most at the start of the overlap and fading "
                                                         "to none at the join, so H3 doesn't copy the overlap so exactly. "
                                                         "The keystone slot stays clean, and the overlap the windows before "
                                                         "made is kept as it was.",
                    "advanced": True}),
                "context_noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                     "tooltip": "How much of the current noise the first overlap latent "
                                                                "gets (1 = the same as the new latents). Fades to 0 at the join.",
                    "advanced": True}),
                "context_noise_fade": (FADES, {"default": "linear",
                                               "tooltip": "How the context noise fades to clean at the join. linear = an "
                                                          "even fade (best in testing). ease = drops off faster near the "
                                                          "start. ethanfel = his drift_control recipe: full on the overlap, "
                                                          "a 4-latent taper to clean at the join, times next/current noise "
                                                          "level at every step.",
                    "advanced": True}),
                "sound_first": ("BOOLEAN", {"default": False,
                                            "tooltip": "Render the sound first on its own (blended windows, picture at "
                                                       "64x64), then lock it in and render the picture.",
                    "advanced": True}),
                "finished_handoff": ("BOOLEAN", {"default": False,
                                                 "tooltip": "On: every window is run all the way to the last step, and "
                                                            "the next window's overlap is locked to those finished frames "
                                                            "(like context motion). Each window is saved at switch_step "
                                                            "and goes back to that save before the blended finish. Costs "
                                                            "the last steps once more per window. Off: the overlap is "
                                                            "locked to the window before's clean guess at switch_step.",
                    "advanced": True}),
            },
        }

    def patch(self, model, context_length=124, context_overlap=39, switch_step=6, keystone_swap=False,
              back_capstone=False, capstone_step=4, fuse_method=cw.ContextFuseMethods.PYRAMID, freenoise=True,
              split_conds_to_windows=False, detail_latent=None, finished_handoff=False, sound_first=False,
              context_noise=False, keystone_step=0, context_noise_strength=1.0, context_noise_fade="linear",
              crash_protection=True, **_old):
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
        handler.keystone_step = int(keystone_step)
        handler.switch_step = int(switch_step)
        handler.finished_handoff = bool(finished_handoff)
        handler.context_noise = float(context_noise_strength) if context_noise else 0.0
        handler.context_noise_fade = context_noise_fade if context_noise_fade in FADES else "linear"
        handler.crash_protection = bool(crash_protection)
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
              + (", finished handoff" if finished_handoff else "") + ((", keystone" + (f" from step {int(keystone_step) + 1}" if int(keystone_step) > 0 else ""))
                                                   if keystone_swap else "")
              + (f", capstone from step {int(capstone_step) + 1}" if back_capstone else "")
              + (", sound first" if sound_first else "")
              + (f", context noise {float(context_noise_strength):.2f} {handler.context_noise_fade}"
                 if handler.context_noise > 0 else "")
              + (", crash protection" if crash_protection else ""))
        return (m,)


NODE_CLASS_MAPPINGS = {"H3ClownHybridWindows": H3ClownHybridWindows}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownHybridWindows": "Geiru Hybrid Windows"}
