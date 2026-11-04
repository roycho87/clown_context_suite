"""
Geiru Protection LoRA: a model-only LoRA loader tied to Geiru Context's degradation_protection switch.

Made for Imagineer13's H3 Anti-Burn LoRA (https://huggingface.co/Imagineer13/H3-Anti-Burn-LoRA), but any LoRA works.
The default, 'Anti-Burn v1 (auto-download)', uses the copy in your loras folder, or downloads it there the first time.
The LoRA is applied only when:
- degradation protection is on (Geiru Context's switch, read from the latent), and
- this isn't the first shot (Geiru Context held frames from a previous shot, or shot_number is past first_shot).
Otherwise the model passes through untouched. The latent always passes through unchanged.
"""
import folder_paths
import comfy.sd
import comfy.utils

import os

WHO = "Geiru Protection LoRA"
_cache = {}

AB_FILE = "h3_antiburn_universal_v1.safetensors"
AB_REPO = "Imagineer13/H3-Anti-Burn-LoRA"
AB_URL = f"https://huggingface.co/{AB_REPO}/resolve/main/{AB_FILE}"
AUTO = "Anti-Burn v1 (auto-download)"


def _find_antiburn():
    """An h3_antiburn_universal_v1 already anywhere in the loras folders, or None."""
    for n in folder_paths.get_filename_list("loras"):
        if os.path.basename(n.replace("\\", "/")) == AB_FILE:
            return folder_paths.get_full_path("loras", n)
    return None


def _download_antiburn():
    """Downloads Imagineer13's Anti-Burn LoRA into <first loras folder>/antiburn/ (once)."""
    dest_dir = os.path.join(folder_paths.get_folder_paths("loras")[0], "antiburn")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, AB_FILE)
    if os.path.isfile(dest):                       # downloaded earlier (the loras list may not have refreshed yet)
        return dest
    print(f"[{WHO}] downloading Imagineer13's H3 Anti-Burn LoRA (about 310 MB) from {AB_URL}")
    try:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            hf_hub_download = None
        if hf_hub_download is not None:
            hf_hub_download(repo_id=AB_REPO, filename=AB_FILE, local_dir=dest_dir)
        else:
            import urllib.request
            part = dest + ".part"
            with urllib.request.urlopen(AB_URL) as r, open(part, "wb") as f:
                total, done, step = int(r.headers.get("Content-Length") or 0), 0, 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total and done * 10 // total > step:
                        step = done * 10 // total
                        print(f"[{WHO}] download {step * 10}%")
            os.replace(part, dest)
    except Exception as e:
        raise RuntimeError(f"{WHO}: couldn't download the Anti-Burn LoRA ({e}). Download {AB_FILE} yourself from "
                           f"https://huggingface.co/{AB_REPO} and put it in your loras folder.")
    print(f"[{WHO}] saved to {dest}")
    return dest


def _path(name):
    if name == AUTO:
        return _find_antiburn() or _download_antiburn()
    return folder_paths.get_full_path_or_raise("loras", name)


def _load(name):
    path = _path(name)
    if _cache.get("path") != path:
        _cache.clear()
        _cache.update(path=path, lora=comfy.utils.load_torch_file(path, safe_load=True))
    return _cache["lora"]


class H3ClownProtectionLora:
    CATEGORY = "Geiru Context Suite"
    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "apply"
    DESCRIPTION = ("Model-only LoRA that only runs while Geiru Context's degradation_protection is on, and never on the "
                   "first shot. Made for Imagineer13's H3 Anti-Burn LoRA (0.6 recommended, adjust to taste). Plug the "
                   "latent from Geiru Context through it so it can read the switch.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT", {"tooltip": "The latent from Geiru Context (passed through unchanged). "
                                                 "Used to read the degradation_protection switch."}),
                "lora_name": ([AUTO] + folder_paths.get_filename_list("loras"),
                              {"default": AUTO,
                               "tooltip": "Anti-Burn v1 (auto-download) uses Imagineer13's H3 Anti-Burn LoRA "
                                          "(huggingface.co/Imagineer13/H3-Anti-Burn-LoRA). If it isn't in your loras "
                                          "folder it's downloaded the first time it's needed, into loras/antiburn. "
                                          "Or pick any other LoRA."}),
                "strength": ("FLOAT", {"default": 0.6, "min": -10.0, "max": 10.0, "step": 0.05,
                                       "tooltip": "0.6 recommended for the Anti-Burn LoRA. Adjust to taste."}),
                "enabled": ("BOOLEAN", {"default": True}),
                "first_shot": ("INT", {"default": 1, "min": -9999, "max": 99999,
                                       "tooltip": "Number of the first shot of your run. The LoRA is never used on it."}),
            },
            "optional": {
                "shot_number": ("INT", {"forceInput": True,
                                        "tooltip": "Connect your Shot Number (same as Geiru Context)."}),
            },
        }

    def apply(self, model, latent, lora_name, strength=0.6, enabled=True, first_shot=1, shot_number=None):
        lat = latent if isinstance(latent, dict) else {}
        if not enabled or float(strength) == 0:
            print(f"[{WHO}] off - model passed through")
            return (model, latent)
        if lat.get("geiru_protection", True) is False:
            print(f"[{WHO}] degradation protection is off on Geiru Context - model passed through")
            return (model, latent)
        if shot_number is not None and int(shot_number) <= int(first_shot):
            print(f"[{WHO}] shot {shot_number} is the first shot - model passed through")
            return (model, latent)
        if lat.get("geiru_held", True) is False:
            print(f"[{WHO}] Geiru Context held nothing (first shot) - model passed through")
            return (model, latent)
        m, _ = comfy.sd.load_lora_for_models(model, None, _load(lora_name), float(strength), 0)
        print(f"[{WHO}] {AB_FILE if lora_name == AUTO else lora_name} at {float(strength):g}" + (f" on shot {shot_number}" if shot_number is not None else ""))
        return (m, latent)


NODE_CLASS_MAPPINGS = {"H3ClownProtectionLora": H3ClownProtectionLora}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownProtectionLora": "Geiru Protection LoRA"}
