from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .clown_chain import NODE_CLASS_MAPPINGS as _CHAIN_NODES, NODE_DISPLAY_NAME_MAPPINGS as _CHAIN_NAMES
from .clown_carryover import NODE_CLASS_MAPPINGS as _CARRY_NODES, NODE_DISPLAY_NAME_MAPPINGS as _CARRY_NAMES
from .clown_keyframe import NODE_CLASS_MAPPINGS as _KEY_NODES, NODE_DISPLAY_NAME_MAPPINGS as _KEY_NAMES
try:
    from .clown_hybrid_windows import NODE_CLASS_MAPPINGS as _HW_NODES, NODE_DISPLAY_NAME_MAPPINGS as _HW_NAMES
except Exception as _e:  # an older ComfyUI (no MiniMax H3 context windows) shouldn't stop the rest of the pack loading
    print(f"[Clown Context Suite] Clowntext Hybrid Windows not loaded (update ComfyUI): {_e}")
    _HW_NODES, _HW_NAMES = {}, {}

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_CHAIN_NODES, **_CARRY_NODES, **_KEY_NODES, **_HW_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_CHAIN_NAMES, **_CARRY_NAMES, **_KEY_NAMES, **_HW_NAMES}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
