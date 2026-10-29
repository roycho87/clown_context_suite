// Clown Keystone: show the keystone / capstone terms on the node (the saved input names stay the same).
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "clown.context.suite.keystone",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3ClownKeyframe") return;
        const label = (node) => {
            const inp = node.inputs?.find((i) => i.name === "keyframe");
            if (inp) inp.label = "clip 1 (keystone / capstone)";
            const w = node.widgets?.find((x) => x.name === "keyframe_mask");
            if (w) w.label = "graft_mask";
            const fs = node.widgets?.find((x) => x.name === "first_shot");
            if (fs) fs.label = "source_clip";
        };
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            label(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            label(this);
            return r;
        };
    },
});
