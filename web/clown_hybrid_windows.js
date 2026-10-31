// Clowntext Hybrid Windows: workflows saved with an older widget order get their values put back on the right
// widgets. Now: ..., back_capstone, finished_handoff, capstone_step, ..., sound_first (last).
//   old A: ..., back_capstone, capstone_step, fuse_method, freenoise, split_conds_to_windows, finished_handoff
//   old B: ..., back_capstone, capstone_step, fuse_method, freenoise, split_conds_to_windows, finished_handoff, sound_first
import { app } from "../../scripts/app.js";

const NEW = ["context_length", "context_overlap", "switch_step", "keystone_swap", "back_capstone", "finished_handoff",
             "capstone_step", "fuse_method", "freenoise", "split_conds_to_windows", "sound_first"];
const OLD = ["context_length", "context_overlap", "switch_step", "keystone_swap", "back_capstone", "capstone_step",
             "fuse_method", "freenoise", "split_conds_to_windows", "finished_handoff", "sound_first"];

app.registerExtension({
    name: "clown.context.suite.hybrid_windows",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3ClownHybridWindows") return;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            const v = info?.widgets_values;
            // the old order has capstone_step (a number) in slot 5; the new one has finished_handoff (true/false)
            if (Array.isArray(v) && v.length >= 10 && typeof v[5] === "number") {
                const saved = {};
                OLD.forEach((name, i) => { if (i < v.length) saved[name] = v[i]; });
                if (v.length < 11) saved.sound_first = false;
                for (const w of this.widgets || []) {
                    if (w.name in saved) w.value = saved[w.name];
                }
            }
            return r;
        };
    },
});
