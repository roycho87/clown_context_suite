// Frame math + icons on the clown nodes' frame selectors.
//   🎈 = Clown Context settings, 🤡 = Clown Anchoring settings
//   e.g. frames 73, drop 17  ->  "🎈 context_drop_last · 17 : 56 total"
import { app } from "../../scripts/app.js";

const CTX = "🎈", ANC = "🤡";
const GROUPS = {
    H3ClownContextExperimental: [{ icon: CTX, total: "head_frames", drops: ["drop_first", "drop_last"] }],
    H3BlockTailAnchorExperimental: [{ icon: ANC, total: "tail_frames", drops: ["drop_first", "drop_last"] }],
};

function refresh(node) {
    const groups = GROUPS[node.comfyClass || node.type];
    if (!groups) return;
    const W = (n) => node.widgets?.find((w) => w.name === n);
    for (const g of groups) {
        const t = W(g.total);
        if (!t) continue;
        const total = Number(t.value) || 0;
        const dropSum = g.drops.reduce((a, n) => a + (Number(W(n)?.value) || 0), 0);
        const kept = Math.max(0, total - dropSum);
        for (const n of g.drops) {
            const w = W(n);
            if (w) w.label = `${g.icon} ${n} · ${Number(w.value) || 0} : ${kept} total`;
        }
        t.label = `${g.icon} ${g.total} (${kept} held)`;
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "clown.frame.labels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!GROUPS[nodeData.name]) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            const groups = GROUPS[nodeData.name];
            const names = groups.flatMap((g) => [g.total, ...g.drops]);
            for (const w of node.widgets || []) {
                if (!names.includes(w.name)) continue;
                const isTotal = groups.some((g) => g.total === w.name);
                const cb = w.callback;
                w.callback = function () {
                    const v = Number(w.value) || 0;
                    if (isTotal) w.value = Math.min(124, Math.max(5, 5 + 17 * Math.round((v - 5) / 17)));   // 5, 22, 39 ... 124
                    else w.value = Math.max(0, 17 * Math.round(v / 17));                                     // 0, 17, 34 ...
                    const res = cb?.apply(this, arguments);
                    refresh(node);
                    return res;
                };
            }
            setTimeout(() => refresh(node), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => refresh(this), 0);
            return r;
        };
    },
});
