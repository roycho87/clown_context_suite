// Clown Carryover follows the Clown Prophecy Sampler: bypass and run_every are copied from it when you
// queue, and hidden. If the node isn't wired to a Prophecy Sampler, the one Prophecy Sampler in the graph
// is used; with none (or several, unwired) the node's own values are used.
import { app } from "../../scripts/app.js";

const PROPHECY = "H3ClownProphecySampler";

function nodes() {
    return app.graph?._nodes || app.graph?.nodes || [];
}

function prophecyFor(node) {
    const g = node.graph || app.graph;
    for (const o of node.outputs || []) {
        for (const id of o.links || []) {
            const l = g.links?.[id] ?? g.links?.get?.(id);
            const dst = l ? g.getNodeById(l.target_id) : null;
            if (dst && (dst.comfyClass || dst.type) === PROPHECY) return dst;
        }
    }
    const all = nodes().filter((n) => (n.comfyClass || n.type) === PROPHECY);
    return all.length === 1 ? all[0] : null;
}

app.registerExtension({
    name: "clown.context.suite.carryover",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3ClownCarryover") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            for (const name of ["bypass", "run_every"]) {
                const w = node.widgets?.find((x) => x.name === name);
                if (!w) continue;
                const prev = w.serializeValue;
                w.serializeValue = async function () {
                    const p = prophecyFor(node);
                    const src = p?.widgets?.find((x) => x.name === name);
                    if (src) w.value = src.value;
                    return prev ? await prev.apply(this, arguments) : w.value;
                };
                // copied from Prophecy: hidden, but still sent with the prompt
                w.hidden = true;
                w.computeSize = () => [0, -4];
                w.draw = () => {};
            }
            // Clown Carryover: latent_folder follows the Load Latent wired into previous_shot
            const lf = node.widgets?.find((x) => x.name === "latent_folder");
            if (lf) {
                const prevLf = lf.serializeValue;
                const loadNode = () => {
                    const inp = node.inputs?.find((i) => i.name === "previous_shot");
                    const g = node.graph || app.graph;
                    const l = inp?.link != null ? (g.links?.[inp.link] ?? g.links?.get?.(inp.link)) : null;
                    return l ? g.getNodeById(l.origin_id) : null;
                };
                lf.serializeValue = async function () {
                    const src = loadNode()?.widgets?.find((x) => x.name === "latent_path");
                    if (src && src.value) lf.value = src.value;
                    return prevLf ? await prevLf.apply(this, arguments) : lf.value;
                };
                // copied from Load Latent: hidden, but still sent with the prompt
                lf.hidden = true;
                lf.computeSize = () => [0, -4];
                lf.draw = () => {};
            }
            // Clown Carryover: clip_number only matters for frame_source = selected clip
            const fs = node.widgets?.find((x) => x.name === "frame_source");
            const cn = node.widgets?.find((x) => x.name === "clip_number");
            if (fs && cn) {
                // old workflows: "first shot" became selected clip 1
                const prevConf0 = node.onConfigure;
                node.onConfigure = function () {
                    const out = prevConf0?.apply(this, arguments);
                    if (fs.value === "first shot") { fs.value = "selected clip"; cn.value = 1; }
                    return out;
                };
                const sync = () => { cn.disabled = fs.value !== "selected clip"; node.setDirtyCanvas?.(true, true); };
                const prevCb = fs.callback;
                fs.callback = function () { const out = prevCb?.apply(this, arguments); sync(); return out; };
                const prevConf = node.onConfigure;
                node.onConfigure = function () { const out = prevConf?.apply(this, arguments); sync(); return out; };
                sync();
            }
            // Clown Carryover: map options ignore lighting (and positions, for pose)
            const ret = node.widgets?.find((x) => x.name === "retention");
            const lit = node.widgets?.find((x) => x.name === "carry_lighting");
            const posw = node.widgets?.find((x) => x.name === "carry_positions");
            if (ret && lit && posw) {
                const syncRet = () => {
                    const isMap = String(ret.value).includes("attribute_transfer");
                    lit.disabled = isMap;
                    posw.disabled = String(ret.value).startsWith("pose");
                    node.setDirtyCanvas?.(true, true);
                };
                const prevRetCb = ret.callback;
                ret.callback = function () { const out = prevRetCb?.apply(this, arguments); syncRet(); return out; };
                const prevConf2 = node.onConfigure;
                node.onConfigure = function () { const out = prevConf2?.apply(this, arguments); syncRet(); return out; };
                syncRet();
            }
            requestAnimationFrame(() => node.setSize?.([node.size[0], node.computeSize()[1]]));
            return r;
        };
    },
});
