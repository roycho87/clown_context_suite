"""Clown Carryover.

On the shots Clown Prophecy runs, adds a frame from an earlier shot (the previous shot or a chosen clip; its last or
first frame; optionally turned into a canny / depth / pose map) as one more reference picture, with prompt lines saying
what it shows. Sits after MiniMax H3 Reference to Video: positive + latent in, positive +
latent out (the latent passes through). The Prophecy Sampler sees the extra picture in the conditioning and adds the
same line to its own rewritten prompt, so the ending is made with it too.

H3 numbers pictures in order (<Picture 1>, <Picture 2> ...), so the new picture is the next free number. With 9 pictures
already there it can't add one and passes everything through.
"""
import math

CATEGORY = "Clown Context Suite"
FPS = 24
MAX_PICTURES = 9
CANVAS_MULTIPLE = 32
SOURCES = ["previous shot", "selected clip"]
MAPS = {"canny attribute_transfer": "canny", "depth map attribute_transfer": "depth", "pose attribute_transfer": "pose"}
MARKERS = ["fully_preserved", "partially_preserved", "weak_reference"] + list(MAPS)
SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
            "non_diegetic_music")


def _join(items):
    return items[0] if len(items) == 1 else (" and ".join(items) if len(items) == 2 else ", ".join(items[:-1]) + ", and " + items[-1])


def _map_lines(n, kind, positions=False):
    """Lines for a control map picture: it only carries shapes / layout / poses (attribute_transfer)."""
    rest = ("colours, textures and lighting come from the other references and the description, and the subjects' "
            "appearance comes from their own references.")
    if kind == "pose":
        return {"subject_definitions": f"<Picture {n}> is a pose map for the target video, showing where the subjects stand "
                                       "and how their bodies are posed, as stick figures on black; it has no appearance, "
                                       "background or lighting.",
                "retention_analysis": f"<Picture {n}> (subject poses of the target video): attribute_transfer - follow only "
                                      "where the subjects stand and how they are posed; their appearance comes from their own "
                                      "references, and the setting comes from the description."}
    if kind == "canny":
        definition = (f"<Picture {n}> is an edge map of the target video's scene, showing the outlines of the location and "
                      "its background as white lines on black")
        what, keep = "scene outlines", "the shapes and layout of its outlines"
    else:
        definition = (f"<Picture {n}> is a depth map of the target video's scene, showing the layout of the location and its "
                      "background as grey shades, near is light and far is dark")
        what, keep = "scene depth layout", "its layout and depth"
    if positions:
        definition += ", and where the subjects are positioned within it"
        keep += " and where the subjects stand"
    definition += "; it has no colour, texture or lighting."
    return {"subject_definitions": definition,
            "retention_analysis": f"<Picture {n}> ({what} of the target video): attribute_transfer - follow only {keep}; {rest}"}


def carryover_lines(n, lighting=False, positions=False, marker="partially_preserved"):
    """The two Ref2V lines for the carryover picture (MiniMax H3 prompting rules): what it is, and how much to keep."""
    if marker in MAPS:
        return _map_lines(n, MAPS[marker], positions)
    shown = ["the location", "its background"] + (["its lighting"] if lighting else [])
    definition = f"<Picture {n}> is the scene layout reference for the target video, showing the environment: {_join(shown)}"
    if positions:
        definition += ", and where the subjects are positioned within it"
    definition += "."
    what = _join(["environment"] + (["lighting"] if lighting else []) + (["subject positions"] if positions else []))
    keep = _join(["location", "background"] + (["lighting"] if lighting else []) + (["where the subjects stand"] if positions else []))
    retention = (f"<Picture {n}> ({what} of the target video): {marker} - follow its {keep}; "
                 "the subjects' appearance comes from their own references.")
    return {"subject_definitions": definition, "retention_analysis": retention}


def _make_map(kind, img):
    """Turn the picture into a canny / depth / pose map. Canny is ComfyUI's own (kornia); depth and pose use
    comfyui_controlnet_aux's Depth Anything V2 and DWPose nodes (models download on first use)."""
    import torch
    import comfy.model_management as mm
    if kind == "canny":
        from kornia.filters import canny
        x = img[..., :3].to(device=mm.get_torch_device(), dtype=torch.float32).movedim(-1, 1)
        edges = canny(x, 0.4, 0.8)[1]
        return edges.repeat(1, 3, 1, 1).movedim(1, -1).float().cpu()
    import nodes
    name = {"depth": "DepthAnythingV2Preprocessor", "pose": "DWPreprocessor"}[kind]
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"Clown Carryover: the {kind} option needs the comfyui_controlnet_aux pack ({name} not found). "
                           "Install 'ComfyUI's ControlNet Auxiliary Preprocessors' from ComfyUI Manager and restart; its "
                           "model downloads by itself the first time. canny works without it.")
    h, w = int(img.shape[1]), int(img.shape[2])
    res = max(512, min(1024, round(min(h, w) / 64) * 64))
    if kind == "depth":
        out = cls().execute(img, ckpt_name="depth_anything_v2_vitl.pth", resolution=res)[0]
    else:
        out = cls().estimate_pose(img, detect_hand="enable", detect_body="enable", detect_face="enable", resolution=res,
                                  bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt")
        out = out["result"][0] if isinstance(out, dict) else out[0]
    return out[..., :3].float().cpu()


def _blank():
    """Nothing for the frame output on shots it's off: a Preview Image on it just doesn't run."""
    from comfy_execution.graph import ExecutionBlocker
    return ExecutionBlocker(None)


def _has_samples(latent):
    return isinstance(latent, dict) and latent.get("samples") is not None


def _add_to_section(prompt, name, line):
    """Put the line at the end of a section. A missing subject_definitions goes on top; a missing other section is skipped."""
    import re
    if line in prompt:
        return prompt
    m = re.search(r"(?im)^[ \t]*" + name + r"[ \t]*:", prompt)
    if not m:
        return line + "\n\n" + prompt if name == "subject_definitions" else prompt
    others = "|".join(x for x in SECTIONS if x != name)
    nxt = re.search(r"(?im)^[ \t]*(" + others + r")[ \t]*:", prompt[m.end():])
    end = m.end() + (nxt.start() if nxt else len(prompt) - m.end())
    body = prompt[m.end():end].rstrip()
    tail = prompt[end:].lstrip("\r\n")
    return prompt[:m.end()] + body + "\n" + line + ("\n\n" + tail if tail else "\n")


def add_carryover_lines(prompt, lines):
    for name in ("subject_definitions", "retention_analysis"):
        if lines.get(name):
            prompt = _add_to_section(prompt, name, lines[name])
    return prompt


def _video_part(latent):
    s = latent["samples"]
    try:
        from comfy.nested_tensor import NestedTensor
    except Exception:
        NestedTensor = None
    if isinstance(s, (list, tuple)):
        parts = list(s)
    elif NestedTensor is not None and isinstance(s, NestedTensor):
        parts = list(s.unbind())
    else:
        parts = [s]
    return max([p for p in parts if p is not None], key=lambda p: p.ndim)


def _load_clip(latent_folder, k=1):
    """Clip k of the chain (*_0000k.safetensors, as Motion Context / MultiRef Save Latent names it), or None."""
    import os
    import folder_paths
    root = os.path.realpath(folder_paths.get_output_directory())
    p = (latent_folder or "").strip().strip('"').strip("'") or "h3_context"
    folder = os.path.realpath(p if os.path.isabs(p) else os.path.join(root, p))
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    if not (folder == root or folder.startswith(root + os.sep)) or not os.path.isdir(folder):
        return None
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.endswith(("_%05d.safetensors" % int(k), "_clip%03d.safetensors" % int(k)))]
    if not files:
        return None
    from safetensors.torch import load_file
    data = load_file(max(files, key=os.path.getmtime))
    if "video" not in data:
        return None
    return {"samples": [data["video"], data.get("audio")]}


def _decode_frame(shot, vae, which="last frame"):
    """First frame: the first latent alone decodes to exactly the first frame. Last frame: see _decode_last_frame."""
    if which == "first frame":
        v = _video_part(shot)
        img = vae.decode(v[:, :, :1])
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]).float().clamp(0, 1)
        return img[:1].contiguous()
    return _decode_last_frame(shot, vae)


def _decode_last_frame(previous_shot, vae):
    """The previous shot's last frame, decoding only its last block: H3 shots are 17n+5 frames = 5n+2 latents, so the
    last 7 latents (one full block + the final 5 frames) start on a block boundary and decode the same as in the full shot."""
    v = _video_part(previous_shot)
    t = int(v.shape[2])
    n = 7 if (t - 2) % 5 == 0 and t >= 7 else t
    img = vae.decode(v[:, :, t - n:])
    img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]).float().clamp(0, 1)
    return img[-1:].contiguous()


def _refs_of(conditioning):
    for c in conditioning:
        if len(c) > 1 and isinstance(c[1], dict) and c[1].get("minimax_refs"):
            return c[1]["minimax_refs"]
    return []


def _items_from_blocks(vae, blocks):
    """What Reference to Video showed the text encoder, rebuilt from the reference latents (as Clown Prophecy does)."""
    items = []
    for b in blocks or []:
        kind = b.get("kind")
        if kind == "image":
            img = b.get("_image")
            if img is None:
                img = vae.decode(b["latent"])
                img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]).float().clamp(0, 1)[:1]
            items.append({"type": "image", "data": img})
        elif kind in ("video", "video_audio"):
            if kind == "video_audio":
                items.append({"type": "audio"})
            fr = vae.decode(b["latent"])
            fr = fr.reshape(-1, fr.shape[-3], fr.shape[-2], fr.shape[-1]).float().clamp(0, 1)
            idx = list(range(0, fr.shape[0], FPS // 2))
            items.append({"type": "video", "data": fr[idx], "timestamps": [i / 2.0 for i in range(len(idx))]})
        elif kind == "audio":
            items.append({"type": "audio"})
    return items


class H3ClownCarryover:
    """A frame from an earlier shot as an extra reference picture, on Clown Prophecy's shots."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE")
    RETURN_NAMES = ("positive", "latent", "frame")
    OUTPUT_TOOLTIPS = ("Positive with the extra picture.", "Latent, passed through.",
                       "The picture that was added (the frame or its map) - for a Preview Image. Nothing on shots it's off.")
    FUNCTION = "carry"
    DESCRIPTION = ("Wire MiniMax H3 Reference to Video's positive and latent through this. On the shots Clown Prophecy runs, "
                   "a frame of the previous shot or a chosen clip (or optional_frame if plugged in) is added as the next reference picture, with a "
                   "line in the prompt saying it shows the scene and layout. The Prophecy Sampler picks it up for the "
                   "ending too. bypass / run_every are copied from the Prophecy Sampler when you queue.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "MiniMax H3 Reference to Video's positive."}),
                "latent": ("LATENT", {"tooltip": "MiniMax H3 Reference to Video's latent (passed through)."}),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "H3 video VAE."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "The same prompt that goes into Reference to Video."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "on - add the frame as a reference",
                                        "label_off": "off - pass through"}),
                "bypass": ("BOOLEAN", {"default": False, "tooltip": "Copied from the Clown Prophecy Sampler when you queue."}),
                "run_every": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Copied from the Clown Prophecy Sampler when you queue."}),
                "first_shot": ("INT", {"default": 1, "min": -9999, "max": 99999,
                                       "tooltip": "On this shot (or earlier) there's no previous shot: off."}),
                "carry_lighting": ("BOOLEAN", {"default": False, "label_on": "lighting: follow it", "label_off": "lighting: free",
                                               "tooltip": "Add the picture's lighting to what the prompt says to follow."}),
                "carry_positions": ("BOOLEAN", {"default": False, "label_on": "subject positions: follow them",
                                                "label_off": "subject positions: free",
                                                "tooltip": "Add where the subjects are standing to what the prompt says to follow."}),
                "retention": (MARKERS, {"default": "partially_preserved",
                                        "tooltip": "How closely H3 follows the picture (retention_analysis marker): fully_preserved = copy "
                                                   "closely, partially_preserved = take some of it, weak_reference = a light hint. "
                                                   "canny / depth map / pose attribute_transfer = the frame is turned into that map "
                                                   "first and only its outlines / layout / poses are followed (depth and pose need "
                                                   "comfyui_controlnet_aux; lighting is ignored, and positions too for pose)."}),
                "frame_source": (SOURCES, {"default": "previous shot",
                                           "tooltip": "previous shot = the shot before (follows the scene, but "
                                                      "copies its drift). selected clip = always clip_number "
                                                      "(1 = the first shot). A fixed clip doesn't build up drift. Ignored when optional_frame is plugged in."}),
                "clip_number": ("INT", {"default": 1, "min": 1, "max": 9999,
                                        "tooltip": "Which saved clip to take the frame from when frame_source = selected clip."}),
                "latent_folder": ("STRING", {"default": "h3_ref2va",
                                             "tooltip": "Where the chain's latents are saved (copied from the Load Latent wired "
                                                        "into previous_shot when you queue). Used by selected clip."}),
                "frame_position": (["last frame", "first frame"], {"default": "last frame",
                                   "tooltip": "Which frame of the chosen shot to use. Ignored when optional_frame is plugged in."}),
            },
            "optional": {
                "shot_number": ("INT", {"forceInput": True, "tooltip": "From Clown Shot Counter (same as Prophecy)."}),
                "previous_shot": ("LATENT", {"lazy": True, "tooltip": "The previous shot's latent (H3 Motion Context Load Latent). "
                                                                     "Its frame is decoded here and used as the picture."}),
                "optional_frame": ("IMAGE", {"lazy": True, "tooltip": "Optional: your own picture to use instead of the previous "
                                                                     "shot's frame. Unplugged = the frame is decoded from previous_shot / the selected clip."}),
            },
        }

    @staticmethod
    def _active(enabled, bypass, run_every, first_shot, shot_number):
        if not bool(enabled):
            return False, "switched off"
        if bool(bypass):
            return False, "Prophecy is bypassed"
        if shot_number is None:
            return True, ""
        shot = int(shot_number)
        if shot <= int(first_shot):
            return False, f"shot {shot} is the first shot (no previous shot)"
        if shot % max(1, int(run_every)) != 0:
            return False, f"Prophecy skips shot {shot} (run_every {int(run_every)})"
        return True, ""

    def check_lazy_status(self, enabled=True, bypass=False, run_every=1, first_shot=1, shot_number=None,
                          frame_source="previous shot", **kw):
        on, _ = self._active(enabled, bypass, run_every, first_shot, shot_number)
        if not on:
            return []
        if "optional_frame" in kw:
            return ["optional_frame"] if kw["optional_frame"] is None else []
        if frame_source in ("first shot", "selected clip"):
            return []  # read from disk, the previous shot isn't loaded
        if "previous_shot" in kw and not _has_samples(kw["previous_shot"]):
            return ["previous_shot"]
        return ["optional_frame"] if "optional_frame" in kw and kw["optional_frame"] is None else []

    def carry(self, positive, latent, clip, vae, prompt, enabled=True, bypass=False, run_every=1, first_shot=1,
              shot_number=None, previous_shot="not wired", optional_frame=None, carry_lighting=False, carry_positions=False,
              retention="partially_preserved", frame_source="previous shot", clip_number=1, latent_folder="h3_ref2va",
              frame_position="last frame"):
        import comfy.utils
        import node_helpers
        who = "Clown Carryover"
        on, why = self._active(enabled, bypass, run_every, first_shot, shot_number)
        if not on:
            print(f"[{who}] off: {why} - passed through")
            return (positive, latent, _blank())
        if optional_frame is not None:
            frame, src = optional_frame, "the optional frame"
        elif frame_source in ("first shot", "selected clip"):
            k = 1 if frame_source == "first shot" else max(1, int(clip_number))
            if shot_number is not None and k >= int(shot_number):
                print(f"[{who}] off: clip {k} isn't made yet on shot {shot_number} - passed through")
                return (positive, latent, _blank())
            saved = _load_clip(latent_folder, k)
            if saved is None:
                print(f"[{who}] off: no saved clip {k} in output/{latent_folder} - passed through")
                return (positive, latent, _blank())
            frame, src = _decode_frame(saved, vae, frame_position), f"clip {k}'s {frame_position}"
        elif not isinstance(previous_shot, str) and not _has_samples(previous_shot):
            print(f"[{who}] off: no saved previous shot - passed through")
            return (positive, latent, _blank())
        elif isinstance(previous_shot, str):
            print(f"[{who}] off: neither previous_shot nor optional_frame is plugged in - passed through")
            return (positive, latent, _blank())
        else:
            frame, src = _decode_frame(previous_shot, vae, frame_position), f"previous shot's {frame_position}"

        blocks = list(_refs_of(positive))
        n_pics = sum(1 for b in blocks if b.get("kind") == "image")
        if n_pics >= MAX_PICTURES:
            print(f"[{who}] off: already {n_pics} reference pictures (H3's limit is {MAX_PICTURES}) - passed through")
            return (positive, latent, _blank())
        n = n_pics + 1

        # size it like Reference to Video's 'match': down only, to the generation's pixel area, in 32-pixel steps
        vid = _video_part(latent)
        width, height = int(vid.shape[-1]) * 16, int(vid.shape[-2]) * 16
        img = frame[-1:, :, :, :3]
        if retention in MAPS:
            img = _make_map(MAPS[retention], img)
            src += f" as a {MAPS[retention]} map"
        h, w = int(img.shape[1]), int(img.shape[2])
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = comfy.utils.common_upscale(img.movedim(-1, 1), tw, th, "bilinear", "disabled").movedim(1, -1)
        new_block = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": vae.encode(resized),
                     "clown_carryover": True}

        # pictures come first in H3's reference order: the new one goes right after the last picture
        last_img = max([i for i, b in enumerate(blocks) if b.get("kind") == "image"], default=-1)
        blocks.insert(last_img + 1, new_block)
        items = _items_from_blocks(vae, [dict(b, _image=resized) if b is new_block else b for b in blocks])

        lines = carryover_lines(n, carry_lighting, carry_positions, retention if retention in MARKERS else "partially_preserved")
        new_prompt = add_carryover_lines(prompt, lines)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(new_prompt, minimax_ref_items=items))
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks, "clown_carryover_lines": lines})
        print(f"[{who}] shot {shot_number}: {src} added as <Picture {n}> ({tw}x{th}), {retention}, lighting {'on' if carry_lighting and retention not in MAPS else 'off'}, positions {'on' if carry_positions else 'off'}")
        return (cond, latent, resized.float().cpu())


NODE_CLASS_MAPPINGS = {"H3ClownCarryover": H3ClownCarryover}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownCarryover": "Clown Carryover"}
