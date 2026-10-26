// Clown Shot Counter + Clown Prophecy Sampler helpers.
//  increment              -> the counter goes up by 1 each time you hit run (like a seed set to increment)
//  increment after accept -> the counter goes up by 1 after the Prophecy Sampler finishes an accepted run
//  decrement              -> the counter goes down by 1 each time you hit run (repair walks back)
//  fixed                  -> never changes
//  use_classic_counter off -> the shot is worked out in Python (saved latents + 1); the box just shows it
// After an accepted run the sampler's 'accept' switch goes back off.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function counterFor(node) {
    const inp = node.inputs?.find((i) => i.name === "shot_number");
    if (inp?.link != null) {
        const link = node.graph.links[inp.link] ?? node.graph.links.get?.(inp.link);
        const src = link ? node.graph.getNodeById(link.origin_id) : null;
        if (src?.type === "H3ClownShotCounter") return src;
    }
    const all = (app.graph._nodes || app.graph.nodes || []).filter((x) => x.type === "H3ClownShotCounter");
    return all.length === 1 ? all[0] : null;
}

function isClassic(counter) {
    const c = counter?.widgets?.find((x) => x.name === "use_classic_counter");
    return c ? c.value !== false : true;
}

// ask ComfyUI how many latents are saved under latent_prefix (count mode display)
async function checkFolder(node) {
    if (isClassic(node)) { node._clownSaved = null; node.setDirtyCanvas?.(true, true); return; }
    const prefix = node.widgets?.find((x) => x.name === "latent_prefix")?.value ?? "";
    try {
        const r = await api.fetchApi(`/clown/saved_shot?prefix=${encodeURIComponent(prefix)}`);
        const j = await r.json();
        node._clownSaved = j.error ? { error: j.error } : j;
    } catch (e) {
        node._clownSaved = { error: "can't check (restart ComfyUI)" };
    }
    node.setDirtyCanvas?.(true, true);
}

function counters() {
    return (app.graph?._nodes || app.graph?.nodes || []).filter((x) => x.type === "H3ClownShotCounter");
}

api.addEventListener("executing", (e) => {
    if (e?.detail == null) for (const n of counters()) checkFolder(n);  // a run finished: recount
});

function bump(counter, by = 1) {
    if (!isClassic(counter)) return;
    const w = counter?.widgets?.find((x) => x.name === "shot_number");
    if (!w) return;
    w.value = Math.max(1, Number(w.value) + by);
    w.callback?.(w.value);
    counter.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "clown.prophecy.counter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "H3ClownShotCounter") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onCreated?.apply(this, arguments);
                const node = this;
                node._clownShot = null;
                const w = node.widgets?.find((x) => x.name === "shot_number");
                if (w) {
                    const prev = w.afterQueued;
                    w.afterQueued = function () {
                        prev?.apply(this, arguments);
                        const mode = node.widgets?.find((x) => x.name === "shot_counter")?.value;
                        if (mode === "increment") bump(node, 1);
                        else if (mode === "decrement") bump(node, -1);
                    };
                }
                // display (not saved with the workflow): the shot this counter last put out
                node.addCustomWidget({
                    name: "clown_shot_display",
                    type: "clown_shot_display",
                    value: "",
                    serialize: false,
                    options: { serialize: false },
                    computeSize(width) { return [width, isClassic(node) ? 34 : 54]; },
                    draw(ctx, n, width, y) {
                        const classic = isClassic(n);
                        const lines = [];
                        if (n._clownShot != null) lines.push([`OUTPUT: SHOT ${n._clownShot}`, "#ffd75e", "bold 15px sans-serif"]);
                        if (classic) {
                            if (n._clownShot == null)
                                lines.push([`next run: shot ${n.widgets?.find((x) => x.name === "shot_number")?.value}`, "#aaa", "bold 14px sans-serif"]);
                        } else {
                            const sv = n._clownSaved;
                            const t = !sv ? "checking folder..." : sv.error ? sv.error
                                : `${sv.last} saved  ->  next run: shot ${sv.next}`;
                            lines.push([t, "#8fd3ff", "bold 13px sans-serif"]);
                        }
                        const h = classic ? 30 : 50;
                        ctx.save();
                        ctx.fillStyle = "#2a1f3d";
                        ctx.beginPath();
                        ctx.roundRect ? ctx.roundRect(10, y + 2, width - 20, h, 6) : ctx.rect(10, y + 2, width - 20, h);
                        ctx.fill();
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        const step = h / (lines.length + 1);
                        lines.forEach(([t, c, f], i) => {
                            ctx.fillStyle = c; ctx.font = f;
                            ctx.fillText(t, width / 2, y + 2 + step * (i + 1));
                        });
                        ctx.restore();
                    },
                });
                // count mode: grey out the box and the increment selector (they do nothing)
                const sync = () => {
                    const classic = isClassic(node);
                    for (const name of ["shot_number", "shot_counter"]) {
                        const x = node.widgets?.find((v) => v.name === name);
                        if (x) x.disabled = !classic;
                    }
                    const sz = node.computeSize();
                    node.setSize([node.size[0], Math.max(sz[1], classic ? sz[1] : node.size[1])]);
                    checkFolder(node);
                };
                node._clownSync = sync;
                const tog = node.widgets?.find((x) => x.name === "use_classic_counter");
                if (tog) {
                    const cb = tog.callback;
                    tog.callback = function () { const rr = cb?.apply(this, arguments); node._clownShot = null; sync(); return rr; };
                }
                const pre = node.widgets?.find((x) => x.name === "latent_prefix");
                if (pre) {
                    const cb = pre.callback;
                    pre.callback = function () { const rr = cb?.apply(this, arguments); checkFolder(node); return rr; };
                }
                const sz = node.computeSize();
                node.setSize([Math.max(node.size[0], sz[0]), Math.max(node.size[1], sz[1])]);
                setTimeout(sync, 0);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure?.apply(this, arguments);
                setTimeout(() => this._clownSync?.(), 0);
                return r;
            };
            const onExec = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (output) {
                onExec?.apply(this, arguments);
                const shot = output?.clown_shot?.[0];
                if (shot == null) return;
                this._clownShot = Number(shot);
                if (!isClassic(this)) {
                    const w = this.widgets?.find((x) => x.name === "shot_number");
                    if (w) w.value = Number(shot);
                }
                this.setDirtyCanvas?.(true, true);
            };
        }
        if (nodeData.name === "H3ClownProphecySampler") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (output) {
                onExecuted?.apply(this, arguments);
                const state = output?.prophecy?.[0];
                if (!state?.accepted) return;
                const acc = this.widgets?.find((x) => x.name === "accept");
                if (acc) acc.value = false;
                const counter = counterFor(this);
                const mode = counter?.widgets?.find((x) => x.name === "shot_counter")?.value;
                if (mode === "increment after accept") bump(counter);
                this.setDirtyCanvas?.(true, true);
            };
        }
    },
});
