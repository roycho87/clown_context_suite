// Clown Keystone: show the keystone / capstone terms on the node (the saved input names stay the same), and put
// workflows saved before keyframe_mask and source_clip were removed back on the right widgets.
//   old: enabled, keyframe_mask, first_shot (source_clip), placement      now: enabled, placement
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "clown.context.suite.keystone",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3ClownKeyframe") return;
        const label = (node) => {
            const inp = node.inputs?.find((i) => i.name === "keyframe");
            if (inp) inp.label = "clip 1 (keystone / capstone)";
        };
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            label(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            const v = info?.widgets_values;
            if (Array.isArray(v) && v.length >= 4 && typeof v[1] === "number") {
                const en = this.widgets?.find((w) => w.name === "enabled");
                const pl = this.widgets?.find((w) => w.name === "placement");
                if (en) en.value = v[0];
                if (pl && typeof v[3] === "string") pl.value = v[3];
            }
            label(this);
            return r;
        };
    },
});
