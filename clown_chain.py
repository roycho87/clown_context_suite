"""Clown Chain: a copy of the Chain node from NikoDemon80's ComfyUI-H3-Motion-Context, renamed.

Original: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context  (Copyright (C) 2026 NikoDemon80, GPL-3.0)

Why it's here: Motion Context and seitanism's ComfyUI-H3-Motion-Context-MultiRef register the same node
names, and MultiRef has no Chain node. A renamed copy lets the Clown nodes chain shots with either pack's
Load / Save Latent nodes.

Changes from the original (GPL-3.0 section 5a):
  - node renamed MiniMaxH3MotionContextChain -> H3ClownChain, shown as "Clown Chain"
  - HTTP routes moved from /h3_motion_context/... to /clown_chain/... so they never clash
  - path, slot and clear helpers and the same-origin check copied into this one file
The buttons themselves live in web/clown_chain.js (also a renamed copy).
"""
import gc
import os
import re
from functools import wraps
from urllib.parse import urlparse, urlsplit

import folder_paths

CATEGORY = "Clown Context Suite"


# ---------------------------------------------------------------- paths (from Motion Context nodes.py)
def _under_output(path):
    """Resolve `path` inside ComfyUI's output folder. None if it would escape."""
    root = os.path.realpath(folder_paths.get_output_directory())
    p = (path or "").strip().strip('"').strip("'") or "h3_context"
    resolved = os.path.realpath(p if os.path.isabs(p) else os.path.join(root, p))
    if resolved != root and not resolved.startswith(root + os.sep):
        return None
    return resolved


def _clip_slot_exists(latent_path, clip_index=1):
    """True if clip `clip_index` is saved (*_00001.safetensors style) in latent_path."""
    c = _under_output(latent_path)
    if not c:
        return False
    if os.path.isfile(c):
        return True
    if not os.path.isdir(c):
        return False
    idx = int(clip_index)
    if idx <= 0:
        return False
    endings = ("_%05d.safetensors" % idx, "_clip%03d.safetensors" % idx)
    return any(f.endswith(endings) for f in os.listdir(c))


def _latent_folder(path):
    c = _under_output(path)
    if not c:
        return None
    if os.path.isdir(c):
        return c
    if os.path.isfile(c):
        return os.path.dirname(c)
    return None


_CHAIN_SLOT_FILE = re.compile(r"(?:_\d{5}|_\d{5}_|_clip\d{3})\.safetensors(?:\.tmp)?$")


def _clear_clip_slots(latent_path):
    """Delete numbered chain slots. Custom filenames are left alone."""
    folder = _latent_folder(latent_path)
    if not folder:
        return 0
    removed = 0
    for name in os.listdir(folder):
        if not _CHAIN_SLOT_FILE.search(name):
            continue
        fp = os.path.join(folder, name)
        try:
            os.remove(fp)
        except OSError:
            gc.collect()
            os.remove(fp)
        removed += 1
    if removed:
        print(f"[Clown Chain] cleared {removed} chain slot(s) from {folder}")
    return removed


# ---------------------------------------------------------------- same-origin check (from Motion Context csrf_guard.py)
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _origin_parts(value):
    raw = (value or "").strip()
    if not raw or raw.lower() == "null":
        return "", None
    parsed = urlparse(raw)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return "", None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return hostname, port


def _host_parts(host_header):
    parsed = urlsplit("//" + (host_header or "").strip().lower())
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    if port is None:
        port = 80
    return hostname, port


def _reject_csrf(request, web):
    if request.method not in _MUTATING:
        return None
    site = (request.headers.get("Sec-Fetch-Site") or "").strip().lower()
    if site == "cross-site":
        return web.json_response({"error": "Cross-site request blocked."}, status=403)
    hh, hp = _host_parts(request.headers.get("Host") or "")
    if not hh:
        return web.json_response({"error": "Missing Host."}, status=403)
    oh, op = _origin_parts(request.headers.get("Origin") or "")
    if not oh:
        oh, op = _origin_parts(request.headers.get("Referer") or "")
        if not oh:
            return web.json_response({"error": "Missing Origin. Same-origin UI requests send it."}, status=403)
    if oh != hh or op != hp:
        return web.json_response({"error": "Origin does not match Host."}, status=403)
    return None


def register_routes():
    """POST /clown_chain/slot_exists and /clown_chain/clear_latents for the buttons."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(register_routes, "_done", False):
        return

    def same_origin(handler):
        @wraps(handler)
        async def wrapped(request):
            blocked = _reject_csrf(request, web)
            if blocked is not None:
                return blocked
            return await handler(request)
        return wrapped

    @server.routes.post("/clown_chain/slot_exists")
    @same_origin
    async def _slot_exists_route(request):
        data = await request.json()
        exists = _clip_slot_exists(data.get("latent_path") or "h3_context", int(data.get("clip_index") or 1))
        return web.json_response({"exists": exists})

    @server.routes.post("/clown_chain/clear_latents")
    @same_origin
    async def _clear_latents_route(request):
        data = await request.json()
        n = _clear_clip_slots(data.get("latent_path") or "h3_context")
        return web.json_response({"removed": n})

    register_routes._done = True


class H3ClownChain:
    """Approve, Run/Re-roll, auto-chain, reset indices, or clear slots (renamed Motion Context Chain)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1,
                                     "tooltip": "How many clips Chain generates before it stops. 0 keeps going until you click Stop."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Copy of NikoDemon80's H3 Motion Context Chain, renamed so it works with either Motion Context or "
                   "Motion Context MultiRef. Approve advances Load/Save and runs the next clip. Run/Re-roll queues the "
                   "current slot (use this instead of ComfyUI's Run). Chain is Approve on a loop; at Load 0 / Save 1 with "
                   "no clip 1 saved it generates clip 1 first. segments = clips before stopping (0 = until Stop). Reset sets "
                   "Load 0 / Save 1. Clear latents deletes numbered chain slots. Load, Save and this node must sit in the "
                   "same canvas group.")

    def noop(self, segments=0):
        return ()


register_routes()

NODE_CLASS_MAPPINGS = {"H3ClownChain": H3ClownChain}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownChain": "Clown Chain"}
