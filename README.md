# Clown Context Suite

ComfyUI nodes for chaining MiniMax H3 clips without the quality slowly falling apart.

## Credits

Clown Context Suite stands on the work of **seitanism** and **nikodemon**, who came before it. I'm just building off their work, so all credit to them for laying the groundwork.

## The problem

When you chain H3 clips, each new clip starts from the tail of the last one. You tell the model where to start, but never where to end. Detail is lost a little in every clip, and because each clip starts from the one before, the loss stacks up over the chain.

## The fix: Clown Prophecy

Give the model a target. Before a shot is rendered, **Clown Prophecy Sampler** renders a short, clean ending for that shot from the shot's own prompt and reference pictures. **Clown Anchoring** pins that ending to the back of the shot, so the shot has to land on good-quality frames. **Clown Context** pins the previous shot's tail to the front, so the motion carries over.

Do this every two shots or so and the quality never drops far enough to notice.

## Nodes

| Node | What it does |
|---|---|
| **Clown Context** | Holds the previous shot's last frames (and audio) at the start of this shot. |
| **Clown Anchoring** | Holds a source's frames (and audio) at the end of this shot. Plug Clown Prophecy's ending into `next_shot`. |
| **Clown Prophecy Sampler** | Renders the clean ending for the shot, before the shot exists. |
| **Clown Prophecy Prompt** | How the ending is worded: an LLM (Ollama, LM Studio…) rewrites the prompt so it only describes the ending, your own text, or the shot prompt as is. |
| **Clown Shot Counter** | The shot number, plus `previous_shot` and `continuation` (true from shot 2 on). |
| **Clown Force Audio** | Puts a real sound track (song, voice) into the latent so the sampler keeps it and lip-syncs to it. |

Everything shows up under the **Clown Context Suite** category.

## Clown Prophecy Sampler

| Setting | Meaning |
|---|---|
| `bypass` | On = Prophecy does nothing and the workflow runs normally, with the main sampler making its own sound. Lets you drop the node into any workflow. |
| `run_every` | 1–10. Prophecy only runs on shots that are multiples of this (3 = shots 3, 6, 9…). Other shots run normally. |
| `prophecy_frames` | Length of the ending (5, 22, 39, 56, 73…). Match Clown Anchoring's `tail_frames`. |
| `seed` | Seed for the ending only. |
| `audio_mask` | How much the main sampler may change the shot's sound: 0 = keep it exactly, 1 = remake it. Ignored when the sound comes from Clown Force Audio with its own mask. |
| `accept` | Off = render a candidate ending, show it, and stop. Run again to reroll. On = use that ending and render the shot. On with no candidate yet = render one and use it straight away. |
| `prophecy_latent_folder` | Folder in `output/` where the endings are saved. |

The shot's sound: if the latent going in already carries sound (from Clown Force Audio), Prophecy uses it. Otherwise it makes the shot's sound with a tiny quick render first, so the ending and the full shot share the same sound.

## Writing prompts for Clown Prophecy

In `ollama` and `manual` mode, Clown Prophecy Prompt rewrites your shot prompt so it only describes the ending. To do that it has to find the parts of your prompt, so **use the standard H3 reference-mode layout**.

**Easiest way to get the format right:** write your prompts with the [H3 Prompt IDE](https://github.com/ethanfel/ComfyUI-H3-Prompt-IDE) node. It outputs the standard layout Clown Prophecy expects. Just make sure every shot after the first has a start time.

- **H3 Prompt IDE:** https://github.com/ethanfel/ComfyUI-H3-Prompt-IDE
- **Official MiniMax H3 prompting rules (Ref2VA):** https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md

The layout looks like this:

```
subject_definitions:
<Subject 1> is the pilot in <Picture 1>.

summary:
[reference generation] What happens in the whole clip.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved.

detailed_description:
One or two sentences of overall style and lighting.

[Shot 1]
What happens in the first shot.

[Shot 2] 00:05.000
What happens from 5 seconds on.

overall_soundscape:
Ambient sound.

non_diegetic_music:
N/A
```

- **Section names on their own line**, ending in a colon, spelled exactly like above. Blank lines between sections are fine either way.
- **Shots labelled `[Shot 1]`, `[Shot 2]`…** inside `detailed_description`.
- **Every shot after the first needs a start time** right after its label: `[Shot 2] 00:05.000` or `[Shot 2] At 00:05.000,`. The node uses these times to work out which shots fall in the ending. Without them, it can only use the last shot.
- **One-shot prompts** just need `[Shot 1]`.

The node cuts out the shots that fall in the ending, renumbers them for the short clip, and fixes the shot numbers in `retention_analysis` to match. In `ollama` mode the LLM then rewrites those shots to show how the action ends.

If your prompts use a different layout, use `as is` mode (your prompt unchanged) or `manual` mode (type the ending yourself).

**How well the ending is written depends on your LLM.** Bigger models that follow instructions well do best. Always look at the candidate before accepting it: reroll, or switch to `manual`, if it's wrong.

**The `llm_sees` switch** on Clown Prophecy Prompt sets how much of your prompt the LLM gets:

| Setting | The LLM gets | Use it for |
|---|---|---|
| **trim all but ending [Shot] (weak LLMs)** (default) | The subject definitions and only the shots that fall in the ending | Smaller or quantized models, which tend to copy the opening shot if they see it |
| **provide full [Shot] description (strong LLMs)** | The whole prompt for context, plus the ending's shots | Strong models that can use the extra context |

If your endings keep looking like the start of the shot, make sure the switch is on the trim setting.

## Clown Shot Counter

| Setting | Meaning |
|---|---|
| `use_classic_counter` on | The `shot_number` box is the shot. `increment` / `decrement` change it each time you hit run, `increment after accept` only after an accepted Prophecy run (one shot at a time only), `fixed` never. |
| `use_classic_counter` off | The shot is worked out when the run starts: highest saved latent in `latent_prefix` + 1. A rejected ending saves nothing, so a batch rerolls the same shot until one is accepted, then moves on. Each new chain needs its own empty folder. |
| `latent_prefix` | The Save Latent `filename_prefix` (count mode). |

The box at the bottom of the node shows the shot the last run used, and in count mode how many latents are saved and which shot is next.

## Clown Context and Clown Anchoring

Both copy real latents into the shot at matching block positions and mask them so the sampler keeps them. Frame counts go 5, 22, 39, 56, 73… (17n + 5); the drop settings go in steps of 17.

- `head_frames` (Context) is the overlap with the previous shot. If you use pre-split audio, set it to the audio overlap (e.g. 39).
- `drop_last` / `drop_first` trim the held frames from either end.
- `hold_audio` holds the audio under the held frames too.

Rules that work:
- Context and Anchoring must never hold the same frames.
- The Anchoring frames must outnumber the Context frames.

## Example workflow

`example_workflows/clown_context_suite_example.json` makes a talking vlog from pre-split audio clips and one prompt file per shot.

1. Set the prompt folder in **Load Text Batch From Dir** and the audio folder in **Shot Audio (one file per shot)**. One `.txt` and one audio file per shot, numbered in order.
2. Load your two reference pictures.
3. Set the batch count to the number of shots and run. With the counter on `increment` and `accept` on, every shot renders in one go.

Video Settings is set to 8-second shots (192 frames), Clown Context to 39 / drop_last 17 to match a 39-frame audio overlap, and Prophecy to run every 2 shots.

### Other packs the example uses

- ComfyUI-H3-Motion-Context (save / load latent)
- ComfyUI-H3-Prompt-IDE
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)
- [ComfyMath](https://github.com/evanspearman/ComfyMath)
- Load Text Batch From Dir, Load Audio Batch From Dir, LoadImageCrop, VideoResolutionNode, and the MiniMax low-VRAM nodes

Clown Prophecy Prompt in `ollama` mode needs a local LLM server: [Ollama](https://ollama.com) (`http://127.0.0.1:11434`), [LM Studio](https://lmstudio.ai) (`http://127.0.0.1:1234`), or any other OpenAI-compatible server. Put its address in `ollama_url`.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/roycho87/clown_context_suite
```

No extra Python packages needed.
