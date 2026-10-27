// Every Clown Context Suite node is pink, so they stand out in any workflow.
// A colour you set yourself on a node (right-click > Colors) is saved with the workflow and kept.
import { app } from "../../scripts/app.js";

const CLOWN_NODES = new Set([
    "H3ClownContextExperimental",
    "H3BlockTailAnchorExperimental",
    "H3ClownProphecySampler",
    "H3ClownProphecyPrompt",
    "H3ClownShotCounter",
    "H3ClownForceAudio",
    "H3ClownChain",
]);
const TITLE_PINK = "#c2407a";
const BODY_PINK = "#6e2347";

app.registerExtension({
    name: "clown.context.suite.colors",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!CLOWN_NODES.has(nodeData.name)) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            this.color = TITLE_PINK;
            this.bgcolor = BODY_PINK;
            return r;
        };
    },
});
