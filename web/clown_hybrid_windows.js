// Geiru Hybrid Windows:
// - workflows saved with an older widget order get their values put back on the right widgets (by name).
//   now:    context_length, context_overlap, switch_step, fuse_method, freenoise, split_conds_to_windows,
//           crash_protection, then (advanced) keystone_swap, keystone_step, back_capstone, capstone_step,
//           context_noise, context_noise_strength, context_noise_fade, sound_first, finished_handoff
//   prev:   ..., keystone_swap, back_capstone, finished_handoff, capstone_step, fuse_method, freenoise,
//           split_conds_to_windows, sound_first, context_noise, context_noise_strength, context_noise_fade,
//           crash_protection
//   old A/B: ..., back_capstone, capstone_step, fuse_method, freenoise, split_conds_to_windows, finished_handoff
//            (, sound_first)
// - settings that do nothing are greyed out: keystone_step when keystone_swap is off, capstone_step when
//   back_capstone is off, context noise strength / fade when context_noise is off.
import { app } from "../../scripts/app.js";

const PREV = ["context_length", "context_overlap", "switch_step", "keystone_swap", "back_capstone", "finished_handoff",
              "capstone_step", "fuse_method", "freenoise", "split_conds_to_windows", "sound_first", "context_noise",
              "context_noise_strength", "context_noise_fade", "crash_protection"];
const OLD = ["context_length", "context_overlap", "switch_step", "keystone_swap", "back_capstone", "capstone_step",
             "fuse_method", "freenoise", "split_conds_to_windows", "finished_handoff", "sound_first"];
// switch -> the settings that only matter while it's on
const LINKS = {
    keystone_swap: ["keystone_step"],
    back_capstone: ["capstone_step"],
    context_noise: ["context_noise_strength", "context_noise_fade"],
};

app.registerExtension({
    name: "clown.context.suite.hybrid_windows",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3ClownHybridWindows") return;

        const get = (node, name) => node.widgets?.find((w) => w.name === name);

        const sync = (node) => {
            for (const [sw, deps] of Object.entries(LINKS)) {
                const on = get(node, sw)?.value !== false;
                for (const d of deps) {
                    const w = get(node, d);
                    if (w) w.disabled = !on;
                }
            }
            node.setDirtyCanvas?.(true, true);
        };

        const hook = (node) => {
            for (const sw of Object.keys(LINKS)) {
                const w = get(node, sw);
                if (!w || w.__geiruHooked) continue;
                const prevCb = w.callback;
                w.callback = function () { const out = prevCb?.apply(this, arguments); sync(node); return out; };
                w.__geiruHooked = true;
            }
        };

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            hook(this);
            sync(this);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            const v = info?.widgets_values;
            let layout = null, fill = {};
            if (Array.isArray(v) && v.length >= 10) {
                if (typeof v[3] === "boolean" && typeof v[6] === "number") {
                    layout = PREV;                         // keystone_swap in slot 3, capstone_step in slot 6
                } else if (typeof v[5] === "number") {
                    layout = OLD;                          // capstone_step in slot 5
                    if (v.length < 11) fill.sound_first = false;
                }
            }
            if (layout) {
                const saved = { ...fill };
                layout.forEach((name, i) => { if (i < v.length) saved[name] = v[i]; });
                for (const w of this.widgets || []) {
                    if (w.name in saved) w.value = saved[w.name];
                    else if (w.name === "keystone_step") w.value = 0;
                }
            }
            hook(this);
            sync(this);
            return r;
        };
    },
});
