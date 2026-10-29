# Clown Context Suite

ComfyUI nodes for chaining MiniMax H3 clips without the quality slowly falling apart.

## Credits

Clown Context Suite stands on the work of **seitanism** and **nikodemon**, who came before it. I'm just building off their work, so all credit to them for laying the groundwork.

**Clown Chain** is nikodemon's Chain node from [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), copied in and renamed (GPL-3.0, like this pack). See [Clown Chain](#clown-chain) for why.

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
| **Clown Shot Counter** | The shot number, plus `previous_shot` and `continuation` (true from shot 2 on). Can also follow the H3 Motion Context Chain node's count. |
| **Clown Carryover** | On Prophecy shots, adds a frame from an earlier shot as one more reference picture, so the background and layout stay put. Can turn the frame into a canny, depth or pose map first. |
| **Clown Chain** | nikodemon's Motion Context Chain node, renamed: Approve, Run/Re-roll, Chain, Reset and Clear buttons that drive the Load / Save Latent shot numbers. |
| **Clown Keystone** | Puts clip 1's first frame (keystone) and/or its capstone into every shot, so the colours and look don't drift. |
| **Clown Late Lock** | Leaves Keystone's capstone free for the first few steps, then locks it, so the shot picks its own pose first. |
| **Clown Force Audio** | Puts a real sound track (song, voice) into the latent so the sampler keeps it and lip-syncs to it. If the sound is shorter than the shot, the rest of the shot stays silent (stock Concat AV Latent lets the model make up sound there instead). |

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
| `lock_accept` | On = `accept` turns on and stays on, so every Prophecy shot runs hands-free. Off = `accept` turns off; when you turn `accept` on, it's used for one Prophecy shot and then switches itself off, so the next Prophecy shot stops for review. Turning `accept` off by hand also turns the lock off. |
| `continue_audio` | Off by default. On = when Prophecy makes the shot's sound, it holds the previous shot's last few seconds of audio at the start, so the new sound carries on from it. Needs `previous_shot` wired. How much it holds is copied from Clown Context's `head_frames`. Skipped on the first shot or when nothing is saved yet. |
| `prophecy_latent_folder` | Folder in `output/` where the endings are saved. |

| Input | Meaning |
|---|---|
| `previous_shot` | Optional. The same Load Latent that feeds Clown Context. Only used when `continue_audio` is on, and only loaded on Prophecy shots. |

The shot's sound: if the latent going in already carries sound (from Clown Force Audio), Prophecy uses it. Otherwise it makes the shot's sound with a tiny quick render first, so the ending and the full shot share the same sound. With `continue_audio` on, that quick render continues the previous shot's sound.

## Clown Carryover

Sits between **MiniMax H3 Reference to Video** and the samplers: positive and latent in, positive and latent out. On the shots Prophecy runs, it takes a frame from an earlier shot, adds it as the next `<Picture N>`, and adds lines to the prompt saying it's the scene layout reference. The Prophecy Sampler sees the extra picture too, so the ending is made with it. On other shots it passes everything through. `bypass` and `run_every` are copied from the Prophecy Sampler when you queue (hidden on the node).

The picture follows the H3 prompting rules: a line in `subject_definitions` saying what it shows and a line in `retention_analysis` saying how closely to follow it.

| Setting | Meaning |
|---|---|
| `enabled` | On / off. |
| `first_shot` | On this shot and earlier there's no earlier shot to use, so it's off. |
| `carry_lighting` | Also tell H3 to follow the picture's lighting. |
| `carry_positions` | Also tell H3 to follow where the subjects stand. |
| `retention` | How closely H3 follows the picture: `fully_preserved`, `partially_preserved` (default), `weak_reference`. Or `canny` / `depth map` / `pose attribute_transfer`: the frame is turned into that map first and only its outlines / layout / poses are followed. |
| `frame_source` | `previous shot` = the shot before (follows the scene, but copies its drift). `selected clip` = always the clip in `clip_number` (1 = the first shot), so drift doesn't build up. |
| `clip_number` | Which saved clip to use with `selected clip`. |
| `frame_position` | `last frame` (default) or `first frame` of that shot. |

| Input / output | Meaning |
|---|---|
| `previous_shot` | The same Load Latent that feeds Clown Context. Its folder is also where `selected clip` looks. |
| `optional_frame` | Your own picture, used instead. |
| `frame` (output) | The picture that was added (or its map), for a Preview Image. Nothing on shots it's off. |

H3 takes up to 9 pictures. If there are already 9, Carryover passes everything through.

Canny works out of the box. The depth and pose options use [ComfyUI's ControlNet Auxiliary Preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux) (Depth Anything V2 and DWPose); their models download by themselves the first time.

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
| **trim all but ending [Shots] (weak LLMs)** | The subject definitions and only the shots that fall in the ending | Smaller or quantized models, which tend to copy the opening shot if they see it |
| **provide full [Shot] description (strong LLMs)** (default) | The whole prompt for context, plus the ending's shots | Strong models that can use the extra context |

If your endings keep looking like the start of the shot, switch it to the trim setting.

## Clown Shot Counter

| Setting | Meaning |
|---|---|
| `use_classic_counter` on | The `shot_number` box is the shot. `increment` / `decrement` change it each time you hit run, `increment after accept` only after an accepted Prophecy run (one shot at a time only), `fixed` never. `follow motion context` copies the `clip_index` from Motion Context's Save Latent node each run (see below). |
| `use_classic_counter` off | The shot is worked out when the run starts: highest saved latent in `latent_prefix` + 1. A rejected ending saves nothing, so a batch rerolls the same shot until one is accepted, then moves on. Each new chain needs its own empty folder. |
| `latent_prefix` | The Save Latent `filename_prefix` (count mode). |

The box at the bottom of the node shows the shot the last run used, and in count mode how many latents are saved and which shot is next.

## Clown Chain

Clown Chain is a copy of the Chain node from nikodemon's [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), renamed. We put it here because Motion Context and seitanism's [ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef) use the same node names, so you can only have one of them installed, and MultiRef has no Chain node. With Clown Chain you can use the Clown nodes with either pack and don't have to choose.

It works exactly like the original. Keep **Load Latent**, **Save Latent** and **Clown Chain** in the same canvas group. Either pack's Load / Save Latent nodes work.

| Button | What it does |
|---|---|
| **Approve** | Moves Load and Save up one and runs the next shot. |
| **Run/Re-roll** | Runs the current shot again. Use this instead of ComfyUI's Run button. |
| **Chain** | Approve on a loop. `segments` = how many shots before it stops (0 = until you press Stop). |
| **Reset** | Sets Load 0 / Save 1. Doesn't delete anything. |
| **Clear latents** | Deletes the numbered saved shots in the Load folder. Custom file names are kept. |

If you have nikodemon's Motion Context installed you can use its own Chain node instead; everything below works the same with either.

On the first shot Clown Context doesn't run the Load Latent node at all, so MultiRef's Load (which errors on an empty folder at `clip_index` 0) is fine.

## Using it with a Chain node

Set the counter's `shot_counter` to **follow motion context**. The counter then copies the `clip_index` from the Motion Context **Save Latent** node every run, so Prophecy's `run_every` lines up with the Chain node's count (Clown Chain or Motion Context's Chain). The counter only reads that number and never changes it; the Chain node stays in charge. Leave Load and Save Latent's `clip_index` as widgets (not wired) so the Chain buttons can move them.

How a session goes, with `run_every` 4 and `lock_accept` off:

1. Press **Chain**. Shots 1–3 run hands-free.
2. On shot 4 Prophecy renders a candidate ending and the chain stops there (nothing is saved for that shot yet).
3. Not happy? Press **Run/Re-roll** for a new candidate.
4. Happy? Turn `accept` on and press **Run/Re-roll**. Shot 4 renders with that ending, and `accept` switches itself back off.
5. Press **Chain** to carry on. It stops again at shot 8.

Don't press **Approve** or **Chain** right after a candidate: both move to the next shot first, so the shot with the candidate would be skipped.

With `lock_accept` on, Chain runs straight through every shot, Prophecy shots included.

## Clown Context and Clown Anchoring

Both copy real latents into the shot at matching block positions and mask them so the sampler keeps them. Frame counts go 5, 22, 39, 56, 73… (17n + 5); the drop settings go in steps of 17.

- `head_frames` (Context) is the overlap with the previous shot. If you use pre-split audio, set it to the audio overlap (e.g. 39).
- `drop_last` / `drop_first` trim the held frames from either end.
- `hold_audio` holds the audio under the held frames too.
- `shot_number` / `first_shot`: both nodes take an optional `shot_number` from the Clown Shot Counter. On the first shot they don't load their source at all and pass the latent through. Handy when Anchoring's `next_shot` comes from a Load Latent of a clip that isn't made yet. Leave it unplugged and they work as before.

**Recommended settings:**

| Node | Frames | Drop |
|---|---|---|
| **Clown Context** | `head_frames` 39 | `drop_last` 17 |
| **Clown Anchoring** | `tail_frames` 56 | `drop_first` 17 |

Rule that works:
- The Anchoring frames must outnumber the Context frames.

### Joining the clips

Each clip is saved whole (all of its frames, overlap included), so trim them when you join them. `drop_last` doesn't remove the overlap, it only changes what's in it. With Clown Context at 39, `drop_last` 17:

- The new shot's frames 0-21 are exact copies of the previous shot's frames 153-174 (the held part).
- The new shot's frames 22-38 are freshly made, but they cover the same moment as the previous shot's frames 175-191 (the dropped part).

So the two clips still share 39 frames of time: 22 identical, and 17 that are two different takes of the same moment. Butted together untrimmed, that moment plays twice. Cut so the join lands inside the identical part, then it's seamless:

1. **Cut the last 39 frames off every clip except the last one** (the default). The previous clip ends on frame 152 and the next one starts on its frame 0, which is an exact copy of frame 153.
2. **Cut the last 17 frames off each clip and the first 22 off the next.** The previous clip ends on frame 174 and the next one carries on from its frame 22. Use this if the start of each clip has something you don't want to keep.

Don't cut the first 39 frames off the next clip instead: that joins two different takes of the same moment and can show a small jump. The last clip is always kept whole. For other settings, the overlap is `head_frames` and the identical part is `head_frames` minus the drops.

**[Seamless Video Combiner](https://github.com/roycho87/seamless_video_combiner)** does the join for you, sound included (with a short audio crossfade at each seam). `example_workflows/sane_clown_combiner.json` is set up for the Clown examples: 10 clips, 8 seconds (192 frames) each, overlap 39, cutting the end of each clip (End pair: End Overlap 39, End Join 0). For way 2, turn on **Use Beginning pair** (Beginning Overlap 39, Beginning Join 22). Copy the clips from `output/h3_ref2va/` into ComfyUI's `input` folder (or pick them in the ten Load Video nodes), set the number of videos and the duration if yours differ, and run.

## Clown Keystone and Clown Late Lock

Every shot in a chain is made from the one before it, so small colour and sharpness changes build up. Clown Keystone puts a latent from clip 1 into every shot, so each one is pulled back to how clip 1 looked.

H3 latents come in 17-frame blocks of 1 + 4 + 4 + 4 + 4 frames. The first latent of each block is a single frame on its own:

- **keystone** = clip 1's first latent (frame 0).
- **capstone** = clip 1's last single-frame latent (frame 187 of a 192-frame clip).

Keystone only swaps those single-frame latents, so nothing else in the shot moves.

| Setting | Meaning |
|---|---|
| `enabled` | On / off. |
| `graft_mask` | 0 = kept exactly (default). 0.2-0.3 = kept roughly. 1 = ignored. |
| `source_clip` | The clip the keystone and capstone come from (1). On that shot and earlier nothing is grafted. Only used when `shot_number` is plugged in. |
| `placement` | `keystone front` (default), `keystone front + back`, `capstone front + back`, `keystone front + capstone back`. front = the shot's first latent, back = its last single-frame latent. |

| Input | Meaning |
|---|---|
| `clip 1 (keystone / capstone)` | A Load Latent fixed on clip 1. Keep it **outside** the Chain group so the Chain buttons don't change its number. |
| `shot_number` | From Clown Shot Counter. On shot 1 clip 1 isn't loaded (it doesn't exist yet). |

**Clown Late Lock** goes between Keystone and the sampler: latent from Keystone in, latent out to the sampler; model in, model out to the guider. For the first `swap_step` steps (default 4) the grafted slot is free, so the shot decides its own motion and pose. From the next step on it's locked to what Keystone put there. `slots` picks which slot: `capstone slot (back)` (default), `keystone slot (front)`, or both. It only touches slots Keystone actually filled, so on shot 1 it does nothing.

A capstone locked from the start pulls the pose back toward clip 1. Late Lock avoids that.

Comparison, clip 10 of the same 10-clip chain: no Keystone (base), keystone front, and keystone front + Late Lock capstone.

In a 10-clip test (measured about 20 frames from the end of clip 10), colour drift from clip 1 went from 6.4 with no Keystone, to 3.9 with keystone front, to 2.9 with keystone front + Late Lock capstone. Over-sharpening dropped as well.

**Joining:** the grafted frames are clip 1's, not part of the shot, so use the 17 + 22 cut: cut the last 17 frames of each clip and the first 22 of the next. In Seamless Video Combiner: **Use Beginning pair** on, Beginning Overlap 39, Beginning Join 22.

## Example workflows

### Clown Chain example

`example_workflows/sane_person_example.json` is a normal H3 Motion Context-style chain with the Clown nodes in, driven by the **Clown Chain** buttons. It only uses the Load / Save Latent nodes that Motion Context and Motion Context MultiRef both have, so it opens with no missing nodes whichever one you installed.

1. Load your reference pictures and write the prompt in the **H3 Prompt IDE** node.
2. Press **Chain** (or **Run/Re-roll** for one shot). Don't use ComfyUI's Run button.
3. The counter is on `follow motion context` and `lock_accept` is off, so every Prophecy shot stops so you can check the ending. Turn `accept` on and press **Run/Re-roll** to use it, then **Chain** to carry on. Raise `run_every` so Chain runs some shots on its own in between.

Keep Load Latent, Save Latent and Clown Chain inside their group.

### Clown Carryover example

A 10-clip test made with the sane_clown_example workflow. The sound was replaced afterwards.

https://github.com/user-attachments/assets/6e446991-2ac2-4c9a-b476-169040baf0ce



`example_workflows/sane_clown_example.json` is the Clown Chain example with **Clown Carryover** added. The Prophecy Sampler's `previous_shot` is wired, so you can turn `continue_audio` on. It runs the same way: write the prompt, press **Chain**. Carryover adds the first clip's last frame as a scene reference on every Prophecy shot (`frame_source` = `selected clip`, `clip_number` 1). Plug its `frame` output into the Preview Image to see what it added.

The model group also has ComfyUI's **Model Attention Backend** node (comfy kitchen attention), bypassed. It does the same job as the Sage attention patch, so turn on one or the other.

### Mad Clown Inventor example

`example_workflows/mad_clown_inventor_example.json` is the Clown Chain example with Prophecy and Anchoring taken out and **Clown Keystone** + **Clown Late Lock** put in: just Clown Context (39, drop_last 17, holding the sound too), Keystone on `keystone front + capstone back`, and Late Lock on the capstone at step 4. It runs the same way: write the prompt, press **Chain**. Join the clips with the 17 + 22 cut (see above).

### Vlog example

`example_workflows/insane_example.json` makes a talking vlog from pre-split audio clips and one prompt file per shot.

1. Set the prompt folder in **Load Text Batch From Dir** and the audio folder in **Shot Audio (one file per shot)**. One `.txt` and one audio file per shot, numbered in order.
2. Load your two reference pictures.
3. Set the batch count to the number of shots and run. With the counter on `increment` and `lock_accept` on, every shot renders in one go.

Video Settings is set to 8-second shots (192 frames), Clown Context to 39 / drop_last 17 to match a 39-frame audio overlap, and Prophecy to run every 2 shots.

### Other packs the examples use

- [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) or [ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef) (save / load latent). Install one, not both: they use the same node names.
- [ComfyUI-H3-Prompt-IDE](https://github.com/ethanfel/ComfyUI-H3-Prompt-IDE)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)
- [ComfyMath](https://github.com/evanspearman/ComfyMath)
- [ComfyUI's ControlNet Auxiliary Preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux) (only for Clown Carryover's depth / pose options)
- Load Text Batch From Dir, Load Audio Batch From Dir, LoadImageCrop, VideoResolutionNode, and the MiniMax low-VRAM nodes

Clown Prophecy Prompt in `ollama` mode needs a local LLM server: [Ollama](https://ollama.com) (`http://127.0.0.1:11434`), [LM Studio](https://lmstudio.ai) (`http://127.0.0.1:1234`), or any other OpenAI-compatible server. Put its address in `ollama_url`.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/roycho87/clown_context_suite
```

No extra Python packages needed.
