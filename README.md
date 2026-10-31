# Clown Context Suite

ComfyUI nodes for chaining MiniMax H3 clips without the quality slowly falling apart.

## Credits

Clown Context Suite stands on the work of **seitanism** and **nikodemon**, who came before it. I'm just building off their work, so all credit to them for laying the groundwork.

**Clowntext Hybrid Windows** is built on **kijai**'s MiniMax H3 context windows in ComfyUI and works the way **JalenBrunson**'s [ComfyUI-HybridWindows](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows) does. All credit to them; I just added the keystone and capstone to it, to help everyone make longer videos without degradation. See [Clowntext Hybrid Windows](#clowntext-hybrid-windows).

**Clown Chain** is nikodemon's Chain node from [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), copied in and renamed (GPL-3.0, like this pack). See [Clown Chain](#clown-chain) for why.

## The problem

When you chain H3 clips, each new clip starts from the tail of the last one. You tell the model where to start, but never where to end. Detail is lost a little in every clip, and because each clip starts from the one before, the loss stacks up over the chain.

## The fix: Clown Keystone (recommended)

Pull every shot back to clip 1. **Clown Context** pins the previous shot's tail to the front of each shot, so the motion carries over. **Clown Keystone** locks clip 1's first frame into the start of every shot and tacks clip 1's capstone onto the back, and **Clown Late Lock** brings the capstone in only for the last steps, so the shot picks its own pose first. Every shot is pulled back toward how clip 1 looked, so the loss can't stack up, and the real frames are put back afterwards so nothing from clip 1 ends up in the clips.

It's the simplest way to run a long chain: no language model and no extra renders. Start with `example_workflows/mad_clown_inventor_example.json`. See [Clown Keystone and Clown Late Lock](#clown-keystone-and-clown-late-lock).

## Clown Prophecy has moved

Clown Prophecy (with Clown Carryover and the Prophecy example workflows) now lives in its own repo: [Clown-Prophecy-Experimental-](https://github.com/roycho87/Clown-Prophecy-Experimental-). Clown Prophecy was moved there because it is too inconsistent and experimental to be considered a finished node. Keystone and Capstone are a much more promising direction.

## Nodes

| Node | What it does |
|---|---|
| **Clown Context** | Holds the previous shot's last frames (and audio) at the start of this shot. |
| **Clown Anchoring** | Holds a source's frames (and audio) at the end of this shot. Plug any source into `next_shot`, e.g. an ending from Clown Prophecy (in its own repo now). |
| **Clown Shot Counter** | The shot number, plus `previous_shot` and `continuation` (true from shot 2 on). Can also follow the H3 Motion Context Chain node's count. |
| **Clown Chain** | nikodemon's Motion Context Chain node, renamed: Approve, Run/Re-roll, Chain, Reset and Clear buttons that drive the Load / Save Latent shot numbers. |
| **Clown Keystone** | Puts clip 1's first frame (keystone) into every shot and/or tacks its capstone onto the back, so the colours and look don't drift. Puts the real frame back afterwards. |
| **Clown Late Lock** | Picks the step Keystone's latents come in: the capstone is tacked onto the back from that step, so the shot picks its own pose first. |
| **Clowntext Hybrid Windows** | One long render (a minute or more) in overlapping windows, instead of a chain of clips: each window renders its first steps in order, then the last steps are blended over the whole video. Clip 1's keystone and capstone go into every window. |

Everything shows up under the **Clown Context Suite** category.

## Clown Shot Counter

| Setting | Meaning |
|---|---|
| `use_classic_counter` on | The `shot_number` box is the shot. `increment` / `decrement` change it each time you hit run, `increment after accept` only after an accepted run of Clown Prophecy (now in its own repo; one shot at a time only), `fixed` never. `follow motion context` copies the `clip_index` from Motion Context's Save Latent node each run (see below). |
| `use_classic_counter` off | The shot is worked out when the run starts: highest saved latent in `latent_prefix` + 1. Each new chain needs its own empty folder. |
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

Set the counter's `shot_counter` to **follow motion context**. The counter then copies the `clip_index` from the Motion Context **Save Latent** node every run, so the shot number lines up with the Chain node's count (Clown Chain or Motion Context's Chain). The counter only reads that number and never changes it; the Chain node stays in charge. Leave Load and Save Latent's `clip_index` as widgets (not wired) so the Chain buttons can move them.

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

**Workflows that use Clown Anchoring** (in [Clown-Prophecy-Experimental-](https://github.com/roycho87/Clown-Prophecy-Experimental-)):

- [sane_person_example.json](https://github.com/roycho87/Clown-Prophecy-Experimental-/blob/main/example_workflows/sane_person_example.json): Clown Chain + Prophecy + Anchoring.
- [sane_clown_example.json](https://github.com/roycho87/Clown-Prophecy-Experimental-/blob/main/example_workflows/sane_clown_example.json): the same, with Clown Carryover.
- [insane_example.json](https://github.com/roycho87/Clown-Prophecy-Experimental-/blob/main/example_workflows/insane_example.json): the vlog, pre-split audio and one prompt file per shot.

Two more are in this repo's `example_workflows` (moved over from my old packs and switched to this suite's Clown Context and Clown Anchoring, same settings):

- [everyone_wants_a_piece_of_the_clown.json](example_workflows/everyone_wants_a_piece_of_the_clown.json): a Motion Context chain with Clown Context (39, drop_first 17) and Clown Anchoring (39, drop_last 17).
- [withers_has_big_nattys.json](example_workflows/withers_has_big_nattys.json): the static vlog (odd shots fresh, even shots continue). On even shots, Clown Context (39, drop_first 17) holds the end of the shot before at the start, and Clown Anchoring (39, drop_last 17) holds the start of the shot after at the end. Motion Context's Load / Save Latent nodes are still used for the clips.

### Joining the clips

Each clip is saved whole (all of its frames, overlap included), so trim them when you join them. `drop_last` doesn't remove the overlap, it only changes what's in it. With Clown Context at 39, `drop_last` 17:

- The new shot's frames 0-21 are exact copies of the previous shot's frames 153-174 (the held part).
- The new shot's frames 22-38 are freshly made, but they cover the same moment as the previous shot's frames 175-191 (the dropped part).

So the two clips still share 39 frames of time: 22 identical, and 17 that are two different takes of the same moment. Butted together untrimmed, that moment plays twice. Cut so the join lands inside the identical part, then it's seamless:

1. **Cut the last 39 frames off every clip except the last one** (the default). The previous clip ends on frame 152 and the next one starts on its frame 0, which is an exact copy of frame 153.
2. **Cut the last 17 frames off each clip and the first 22 off the next.** The previous clip ends on frame 174 and the next one carries on from its frame 22. Use this if the start of each clip has something you don't want to keep.

Don't cut the first 39 frames off the next clip instead: that joins two different takes of the same moment and can show a small jump. The last clip is always kept whole. For other settings, the overlap is `head_frames` and the identical part is `head_frames` minus the drops.

**[Seamless Video Combiner](https://github.com/roycho87/workflows/blob/main/workflows/seamless_video_combiner/clussy_origin_story.json)** (in my workflows repo) does the join for you, sound included (with a short audio crossfade at each seam). `example_workflows/sane_clown_combiner.json` is set up for Clown chains: 10 clips, 8 seconds (192 frames) each, overlap 39, cutting the end of each clip (End pair: End Overlap 39, End Join 0). For way 2, turn on **Use Beginning pair** (Beginning Overlap 39, Beginning Join 22). Copy the clips from `output/h3_ref2va/` into ComfyUI's `input` folder (or pick them in the ten Load Video nodes), set the number of videos and the duration if yours differ, and run.

## Clown Keystone and Clown Late Lock

Every shot in a chain is made from the one before it, so small colour and sharpness changes build up. Clown Keystone puts a latent from clip 1 into every shot, so each one is pulled back to how clip 1 looked.

H3 latents come in 17-frame blocks of 1 + 4 + 4 + 4 + 4 frames. The first latent of each block is a single frame on its own:

- **keystone** = clip 1's first latent (frame 0).
- **capstone** = clip 1's last single-frame latent (frame 187 of a 192-frame clip).

- **Front:** the keystone (or capstone) is locked into the shot's latent 0. After sampling, the chain's real latent 0 is always put back into the output, so the saved latent and the video keep the real frame.
- **Back:** nothing in the shot is replaced. The latent is tacked on after the shot's last latent for each model call (locked) and cut off again, so it's never in the output. It sits in a 4-frame latent slot.

Restore and the back both need the model, so plug the model through Keystone (model in, model out to the guider) or use Clown Late Lock.

| Setting | Meaning |
|---|---|
| `enabled` | On / off. |
| `graft_mask` | 0 = kept exactly (default). 0.2-0.3 = kept roughly. 1 = ignored. |
| `source_clip` | The clip the keystone and capstone come from (1). On that shot and earlier nothing is grafted. Only used when `shot_number` is plugged in. |
| `placement` | `keystone front` (default), `keystone front + back`, `capstone front + back`, `keystone front + capstone back`. front = written into the shot's first latent, back = tacked on after the shot's end. |

| Input | Meaning |
|---|---|
| `clip 1 (keystone / capstone)` | A Load Latent fixed on clip 1. Keep it **outside** the Chain group so the Chain buttons don't change its number. |
| `model` | Optional. Plug it through here (model out to the guider) if you don't use Late Lock: then the back is tacked on from step 1 and latent 0 is restored. |
| `shot_number` | From Clown Shot Counter. On shot 1 clip 1 isn't loaded (it doesn't exist yet). |

**Clown Late Lock** goes between Keystone and the sampler: latent from Keystone in, latent out to the sampler; model in, model out to the guider. `swap_step` (default 4) is when Keystone's latents come in:

- **Capstone (back):** nothing is there for the first `swap_step` steps, then it's tacked onto the back for the rest, so nothing pulls on her motion and pose while the shot decides them.
- **Keystone (front):** the slot is held empty (left as noise the model doesn't change) for the first `swap_step` steps, then locked in.

`slots` picks which: `capstone slot (back)` (default), `keystone slot (front)`, or both. It only touches what Keystone set up, so on shot 1 it does nothing. Late Lock also does Keystone's restore, so you don't need the model through Keystone as well.

A capstone locked from the start pulls the pose back toward clip 1. Late Lock avoids that.

Comparison, clip 10 of the same 10-clip chain: no Keystone (base), keystone front, and keystone front + Late Lock capstone.

https://github.com/user-attachments/assets/3002019d-7d6d-4afc-9592-09a3c201fb78

In a 10-clip test (measured about 20 frames from the end of clip 10), colour drift from clip 1 went from 6.4 with no Keystone, to 3.9 with keystone front, to 2.9 with keystone front + Late Lock capstone. Over-sharpening dropped as well. (That test used the earlier version, which wrote the capstone into the shot's last single-frame slot instead of tacking it on.)

**Joining:** with the model plugged through Keystone or Late Lock, nothing from clip 1 ends up in the saved clips, so join them the normal way (cut the last 39 frames of each clip). Without the model, latent 0 can't be put back and stays clip 1's frame, so use the 17 + 22 cut instead: cut the last 17 frames of each clip and the first 22 of the next (Seamless Video Combiner: **Use Beginning pair** on, Beginning Overlap 39, Beginning Join 22).

## Clowntext Hybrid Windows

### Credits

This node is not my idea. It's built on **kijai**'s MiniMax H3 context windows, which are in ComfyUI now ([PR #17425](https://github.com/Comfy-Org/ComfyUI/pull/17425)), and it works the way **JalenBrunson**'s [ComfyUI-HybridWindows](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows) does (based on the [Rolling Diffusion](https://arxiv.org/abs/2402.09470) paper). All credit to them. I just added the keystone and capstone feature to it, to help everyone make longer videos without degradation.

### What it does

Instead of chaining clips, it renders the whole video in one go, in overlapping windows, so memory stays at one window:

1. Window 1 renders the first `switch_step` steps, and a snapshot is saved.
2. Window 1 carries on to the last step, only to get its finished first frame (the **keystone**) and its last single-frame latent (the **capstone**). That finish is thrown away and window 1 goes back to the snapshot.
3. Windows 2, 3 and on render the first `switch_step` steps one after another. Each one's overlap with the window before is locked to that window's result. The keystone is locked into each window's first slot, and the capstone is tacked onto its back from `capstone_step` on (and cut off again).
4. The last steps run as blended context windows over the whole video, which smooths the joins.

With keystone and capstone off, it works like plain hybrid windows. In my test (a 60-second render, 17 windows, same seed), the plain version lost a lot of detail by the end and the keystone + capstone version held its detail much better.

Plug the model through it (model in, model out to the guider). It needs a ComfyUI with MiniMax H3 context windows: update ComfyUI if the node is missing.

| Setting | Meaning |
|---|---|
| `context_length` / `context_overlap` | Window size and overlap in frames (17n + 5). 124 / 39 is about 5 seconds per window. |
| `switch_step` | Steps each window renders on its own before the blended finish. 6 of 8 = steps 1-6 window by window, 7-8 blended. |
| `keystone_swap` | Window 1's finished first frame is locked into the first slot of every later window. |
| `back_capstone` | Window 1's finished last single-frame latent is tacked onto the back of every later window. |
| `finished_handoff` | On: every window renders all the way to the last step, and the next window's overlap is locked to those finished frames. Every window goes back to its `switch_step` snapshot before the blended finish. Costs the last steps once more per window. |
| `capstone_step` | Steps before the capstone comes in. Keep it below `switch_step` (4 with 6 = on for steps 5-6), or it never comes in. |
| `fuse_method` | How the blended steps average the windows (pyramid). |
| `freenoise` | Reuses window 1's noise for every window: more consistent, but motion can repeat. Off gives more variety. |
| `split_conds_to_windows` | Give each window its own prompt (see below). |
| `detail_latent` | Optional: use this latent's first frame as the keystone instead of window 1's. |
| `sound_first` | Off by default. On: renders the sound first on its own (picture at 64x64), then locks it in and renders the picture. |

**One prompt per window:** make one prompt node per window, join them in order with ComfyUI's **Conditioning Combine** (chain them for more than two), plug the result into the guider, and turn `split_conds_to_windows` on. Each window takes the prompt for the part of the video under its middle, so with as many prompts as windows, window 1 gets prompt 1, window 2 prompt 2, and so on. One prompt for a whole minute makes every window try to do the whole thing, so the motion repeats.

## Example workflows

### Mad Clown Inventor example

`example_workflows/mad_clown_inventor_example.json` is a Clown Chain workflow with **Clown Keystone** + **Clown Late Lock**: Clown Context (39, drop_last 17, holding the sound too), Keystone on `keystone front + capstone back`, and Late Lock on the capstone at step 4.

1. Load your reference pictures and write the prompt in the **H3 Prompt IDE** node.
2. Press **Chain** (or **Run/Re-roll** for one shot). Don't use ComfyUI's Run button.
3. Keep Load Latent, Save Latent and Clown Chain inside their group, and the clip 1 Load Latent outside it.

Late Lock does the restore, so join the clips the normal way (see [Joining the clips](#joining-the-clips)).

### Clown Coaster example

`example_workflows/clown_coaster_example.json` is a 60-second **Clowntext Hybrid Windows** render in one go: a first-person space roller coaster where only the rider's gloved hands are seen. A rocket launch throws her onto the track, then a loop-de-loop, ghost space stingrays, down to Mars, she hits a HYPERDRIVE button, a ride over the Sun, and around a black hole.

- 1433 frames (60 s) in 8 windows of 226 frames, overlap 39.
- 8 prompt nodes, one per window, joined in order with **Conditioning Combine**, with `split_conds_to_windows` on.
- Keystone, capstone (`capstone_step` 4) and `finished_handoff` on, `freenoise` off.

Load your subject picture into Picture 1 and press ComfyUI's Run. The length and size are on PROMPT 1; if you change the length, the prompts won't line up one per window any more.

### Clowns for Picard example

`example_workflows/clowns_for_picard.json` is the Clown Coaster example as I actually rendered it: widescreen 1328 × 736 (0.98 MP), 60 seconds (1433 frames) in 8 windows of 226 frames, with more detailed, faster-moving prompts. The 8 prompts are in big **WINDOW 1-8 PROMPT** text boxes in the green group at the bottom, in ride order, so you don't have to type into the small prompt nodes.

Keystone, capstone (`capstone_step` 4), `finished_handoff` and `sound_first` on, `freenoise` off. Load your subject picture into Picture 1 and press ComfyUI's Run. It's a big render (a full minute at almost 1 MP); lower the size if you run out of memory.

### Other packs the examples use

- [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) or [ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef) (save / load latent). Install one, not both: they use the same node names.
- [ComfyUI-H3-Prompt-IDE](https://github.com/ethanfel/ComfyUI-H3-Prompt-IDE)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)
- [ComfyMath](https://github.com/evanspearman/ComfyMath)
- LoadImageCrop, VideoResolutionNode, and the MiniMax low-VRAM nodes

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/roycho87/clown_context_suite
```

No extra Python packages needed.
