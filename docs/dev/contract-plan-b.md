# Plan B contract: learned finishing stage

Shared rules for the three parts built in parallel: the runtime, the evaluation tool and the training code.
Everything here is technical. Do not add personal details, absolute paths with a user name, or reasons for wanting
the feature to this file or to anything else that is tracked.

**No training data exists yet.** Every model built now is a stand-in that only proves the plumbing. Nothing may claim that a
learned look is good. A stand-in can never be used on a virtual camera (§6.3).

Keywords: MUST and MUST NOT are binding. Where a number is marked *provisional*, it was chosen without a real model.
Change it only through this file, and say what measurement justified the change.

---

## 1. Measured facts

Measured on 2026-11-06 with throwaway random-weight U-Nets, on one machine: Windows 11, AMD Radeon RX 9070 XT (15.8 GB dedicated, driver 32.0.32015.7029), Ryzen 7 9800X3D (8 cores, 16 threads), Python 3.12.10, `onnxruntime-directml` 1.24.4, `torch` 2.15.0. The GPU was idle apart from the desktop. **Random weights say nothing about any look**, and these numbers do not include a video call app, OBS or a real camera running alongside. Probe scripts: `.test-media/planb/probe/` (not tracked).

### 1.1 Environments

| Folder | Contents | Result |
|---|---|---|
| `.venv-neural` | `requirements-neural.txt`: the app's requirements plus `onnxruntime-directml` | `DmlExecutionProvider` lists and runs on the RX 9070 XT. Outputs match the CPU's within 1.4e-7. All nodes in one DML partition. Other package versions equal `.venv`'s. |
| `.venv-train` | `training/requirements.txt`: CPU `torch` 2.15.0, `onnx`, `onnxscript`, CPU `onnxruntime` 1.31.1 | Works, including `torch.onnx.export`. |
| (not created in the project) | `torch-directml` 0.2.5.dev240914 with `torch` 2.4.1 | **A cp312 wheel does exist.** `pip index versions` hides it because it is a dev release, but `pip install torch-directml` finds it. Tried in a temporary folder: see 1.7. |

### 1.2 Latency alone, on the GPU (ms)

Fresh process per model. `create` and `first` are the first time that model was ever loaded on this machine (the driver's shader cache was not cleared, so a truly first run may be slower; the very first DML session here took 305 ms). 30 Hz is the realistic case: the GPU's clocks drop between frames, so it is slower than flat out. 300 paced and 200 flat-out runs, inputs changing. Exported with the legacy exporter at opset 17.

| Model | Weights MB | GMAC | Create | First run | 30 Hz median | p95 | p99 | max | Flat-out median | Process VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|
| p08 @256 | 0.55 | 0.72 | 166 | 5.9 | 1.57 | 1.76 | 2.08 | 2.11 | 0.52 | 84 |
| d24 @256 (depthwise) | 0.99 | 1.10 | 367 | 8.9 | 1.61 | 1.84 | 2.14 | 2.16 | 0.57 | 185 |
| p16 @256 | 2.15 | 2.81 | 115 | 6.2 | 1.59 | 2.01 | 2.28 | 2.32 | 0.53 | 121 |
| p32 @256 | 8.57 | 11.1 | 122 | 9.9 | 2.38 | 2.62 | 3.83 | 3.88 | 0.93 | 194 |
| p16 @384 | 2.15 | 6.33 | 133 | 10.1 | 2.95 | 3.24 | 4.26 | 4.46 | 1.29 | 190 |
| p24 @384 | 4.83 | 14.1 | 138 | 12.1 | 4.75 | 5.24 | 6.75 | 7.20 | 2.21 | 192 |
| p32 @384 | 8.57 | 25.1 | 122 | 15.8 | 3.18 | 4.81 | 4.87 | 6.16 | 2.14 | 331 |
| p08 @512 | 0.55 | 2.87 | 122 | 10.8 | 3.45 | 3.68 | 4.58 | 4.87 | 1.71 | 192 |
| d24 @512 (depthwise) | 0.99 | 4.38 | 131 | 18.8 | 4.17 | 4.91 | 4.96 | 5.85 | 2.80 | 596 |
| p16 @512 | 2.15 | 11.3 | 129 | 15.3 | 4.21 | 4.49 | 4.52 | 6.46 | 2.04 | 329 |
| d40 @512 (depthwise) | 2.67 | 11.4 | 133 | 23.9 | 6.27 | 8.72 | 8.81 | 11.02 | 5.13 | 597 |
| p24 @512 | 4.83 | 25.1 | 138 | 17.3 | 5.02 | 8.08 | 8.34 | 11.65 | 3.66 | 331 |
| p32 @512 | 8.57 | 44.5 | 120 | 21.9 | 4.52 | 6.94 | 7.16 | 8.54 | 3.28 | 604 |
| p64 @512 | 34.2 | 177 | 195 | 42.6 | 10.63 | 10.96 | 11.16 | 17.15 | 9.72 | 1168 |

Reading it: depthwise convolutions are not cheaper here (d40 @512 is slower than p32 @512 and has a quarter of the MACs). At these sizes, time follows the number of layers and the resolution more closely than MACs. VRAM moves in steps. Process CPU during the paced runs: 4 to 31 % of one logical CPU.

**The table above has no normalization; the students of `training/` use `InstanceNormalization`, and that is much slower.** Measured later the same way (fresh process, 30 Hz, 300 runs, inputs changing, all nodes on DML), and next to the running app (`natural`, 1280x720, 450 frames, round trip through the worker):

| Student | File MB | Alone median / p95 / p99 ms | Next to the app p95 / highest p99 ms | Process VRAM MB |
|---|---|---|---|---|
| p16 @256 float32 | 2.26 | 1.95 / 2.12 / 2.55 | 2.17 / 2.92 | 121 |
| p32 @512 float16, no normalization | 4.37 | 2.25 / 2.46 / 2.61 | 3.37 / 3.65 | 331 |
| p32 @512 float16 | 4.41 | 6.17 / 7.26 / 7.45 | 6.55 to 7.21 / 10.34 | 331 |
| p32 @512 float32 | 8.68 | 9.14 / 9.38 / 9.46 | 9.98 / 10.42 | 604 |
| p48 @384 float16 | 9.76 | 8.95 / 9.07 / 9.20 | 10.2 / 10.86 | 197 |
| p40 @512 float16 | 6.82 | 15.36 / 15.63 / 15.78 | 16.49 / 17.12 | 333 |
| p48 @512 float16 | 9.76 | 15.84 / 16.10 / 16.32 | 18.62 / 21.72 | 336 |
| p64 @512 float16 | 17.25 | 17.88 / 20.78 / 21.82 | 19.84 / 22.05 | 613 |
| p96 @512 float16 | 38.64 | 23.37 / 23.76 / 24.00 | 24.61 / 25.32 | 634 |

There is a step between p32 and p40 at 512; its cause was not looked into. With one GPU contender, p32 @512 float16 was 8.48 ms p95, p16 @256 5.12 ms. These weights had only a few training steps: speed only.

### 1.3 Next to the running app

Real `MakeupPipeline` with real tracker and segmenter, threads arranged as in `runner.py` (a capture thread announcing frames, an output thread rendering the newest), clip `1012_IEO_HAP_HI` looped and upscaled to 1280x720, 30 fps, 600 frames. No camera, no virtual camera. The stage is a prototype of §5 and §7. "Round trip" is through a worker process, copies included.

| Preset | Model | Wait | Delivered fps | `process` median / p95 / max ms | Process CPU (logical) | Round trip median / p95 / p99 / max ms | Stage on camera thread median / p95 ms |
|---|---|---|---|---|---|---|---|
| natural | none | | 30.00 | 30.0 / 32.5 / 40.8 | 1.90 | | |
| natural | p16 @256 | 0 | 30.00 | 32.6 / 34.9 / 42.2 | 2.06 | 1.75 / 1.82 / 1.89 / 2.30 | 3.01 / 3.44 |
| natural | p24 @384 | 0 | 30.00 | 32.9 / 35.0 / 42.5 | 2.09 | 4.83 / 4.96 / 5.09 / 5.82 | 3.33 / 3.75 |
| natural | p32 @512 | 0 | 29.99 | 32.9 / 35.4 / 42.8 | 2.14 | 4.72 / 7.24 / 7.40 / 7.75 | 3.59 / 4.22 |
| natural | p64 @512 | 0 | 29.99 | 33.0 / 35.7 / 44.1 | 2.19 | 10.95 / 11.29 / 11.72 / 15.30 | 3.64 / 4.29 |
| natural | p32 @512 | **8 ms** | **27.92** | 34.9 / 41.3 / 53.1 | 2.12 | 4.69 / 7.32 / 7.52 / 7.86 | 9.82 / 12.54 |
| strong | none | | 29.99 | 32.9 / 36.1 / 42.6 | 2.19 | | |
| strong | p32 @512 | 0 | 29.41 | 33.4 / 38.4 / 45.6 | 2.40 | 4.83 / 7.27 / 7.48 / 7.64 | 3.65 / 4.35 |

- At wait 0 no result was ever older than one frame in any run above.
- The stage's share on the camera thread at p32 @512 (median ms): crop and signal 0.7, weight masks 1.7, warp and add 1.2.
- `process` can exceed 33 ms at 30.0 fps: part of it is the wait for landmarks, which shrinks when the renderer runs late.
- **The frame has about 3 ms to spare at `natural`**, so a synchronous wait is not affordable, and the stage's own cost is what matters. `strong` already loses 0.6 fps to the stage.
- These fps figures are not comparable with earlier figures for reading OBS Virtual Camera: different input and no virtual camera.

Synthetic GPU load (other processes running p64 @512 flat out; DML work appears under the 3D engine at about 94 %), model paced at 30 Hz, no app:

| Load | p32 @512 median / p95 / max ms | p16 @256 median / p95 / max ms |
|---|---|---|
| none | 4.24 / 6.76 / 7.03 | 1.59 / 2.01 / 2.32 |
| one contender | 4.11 / 7.16 / 7.62 | 1.32 / 4.57 / 4.91 |
| two contenders | 6.33 / 10.98 / 11.76 | 3.85 / 8.32 / 9.20 |

With the app and one contender, p32 @512 was 4.45 / 7.64 / 8.46 ms with 30.00 fps delivered.

### 1.4 Worker process, device loss, errors

| Case | What happens |
|---|---|
| Worker process start: spawn, import, session, first run | 350 to 420 ms, ready to serve |
| Extra cost of the process boundary, per frame | 0.2 ms copies at 256, 0.5 ms at 512; round trip at p32 @512 under 1 ms above the bare run |
| Kill a worker | about 25 ms; parent unaffected. Hung and crashed workers both handled the same way. |
| **Device removed** (simulated with `ID3D12Device5::RemoveDevice` in that process only; reason `0x887A0005`) | `session.run` raises at once (0.1 to 0.3 ms). **On a Japanese Windows it is a `UnicodeDecodeError`, not an ORT exception**: ORT cannot decode the system's error text. The process does not crash. |
| New session in that process, afterwards | **Never gets the GPU back**, also after releasing the sessions and our own device handle (35 references to the device remained, not ours). `InferenceSession` **silently falls back to the CPU**, even with `providers=["DmlExecutionProvider"]`, printing `EP Error` to stdout. `get_providers()` is the only way to tell. Tried 3 times, 1 s apart. |
| New worker process, afterwards | Gets the GPU at once, in the same 350 ms. |
| Invalid `device_id` | Same silent fallback to the CPU. |
| Corrupt, truncated or missing model file | ORT exception on load (`InvalidProtobuf`, `NoSuchFile`). |
| Wrong shape, dtype or input name | Exception on `run` (`InvalidArgument`, `ValueError`). The session keeps working afterwards. |
| NaN input | No exception. The output happened to be finite for this model. |
| `run` and the GIL | Released: a Python thread keeps running during `run`. |
| `enable_mem_pattern`, execution mode, `intra_op_num_threads` | No measurable effect on latency, and no error, at this version. The settings of §7.1 follow ORT's DML guidance anyway. Graph optimization off: about 8 % slower. |

**Not tested**: a real driver reset (TDR), a real hang inside `run`, running out of VRAM, a GPU removed physically, sleep and resume. A real TDR is expected to look like the simulated case after up to about 2 s of blocking, which is why §7.4 assumes `run` can stop returning.

### 1.5 Ops and opsets

Each op between two `Conv`s at 256, DirectML against CPU. Results are in §8.

| Finding | Number |
|---|---|
| Opset 11, 13, 17, 18, 20, 21, from both exporters | One DML partition; 1.2 ms (p16 @384, flat out) |
| **Opset 23** | **`Conv` runs on the CPU: 44 ms.** No warning. |
| `Resize` with `cubic` | On the CPU: **202 ms** at this small size |
| `Gelu` (opset 20) | On the CPU |
| `GroupNormalization` | Its small `Div` and `Concat` shape nodes on the CPU |
| Float16 inside the graph (Cast at the edges) | Runs on DML. p32 @512 paced: 2.87 against 4.37 ms median, 3.26 against 6.88 p95. File 4.36 against 8.63 MB. Output differs by about 2e-4 at this random model. The `model.half()` route through the dynamo exporter works. |
| Dynamic axes | Run, but about 3 to 4 times slower than the same model with static shapes (p16 @384: 4 to 5 ms against 1.2), and a new size costs about 15 ms the first time. |
| Depthwise and dilated `Conv`, `ConvTranspose`, `DepthToSpace`, `InstanceNormalization`, `BatchNormalization`, `GridSample`, `MatMul`+`Softmax`, `Pad` | On DML, outputs match the CPU |
| `LayerNormalization` | Not verified (the test model was invalid). Treat as not allowed until checked. |

### 1.6 Face geometry

Landmarks and hair masks on all 44 CREMA-D clips (11 women, 3218 frames, 1280x720 as above) and the NASA stills. Unit: `face_scale`. `a` runs along the eyes, `b` down from the eye midpoint.

| | min | p1 | median | p99 | max |
|---|---|---|---|---|---|
| Face oval, left (`a`) | -1.38 | -1.28 | -1.13 | -1.03 | -0.99 |
| Face oval, right (`a`) | 0.90 | 0.93 | 1.16 | 1.25 | 1.36 |
| Face oval, top (`b`) | -1.06 | -1.03 | -0.93 | -0.86 | -0.83 |
| Face oval, chin (`b`) | 1.66 | 1.69 | 1.86 | 2.18 | 2.31 |
| Hair, top (`b`, 1st percentile of hair pixels) | -2.27 | -2.26 | -1.87 | -1.61 | -1.58 |
| `face_scale` in pixels | 69 | 70 | 114 | 146 | 148 |

- Hence the crop of §4: side 4.5, center 0.4 below the eyes, so `b` covers -1.85 to 2.65 and `a` -2.25 to 2.25, and the part before the edge window starts reaches `b` -1.49 to 2.29. The whole face oval fits for all frames; the chin's maximum is at the start of the window. The crown of the head reaches the crop's edge and fades. Long hair below the shoulders is outside, and is left as it is.
- At the median face the crop side is 513 frame pixels, so a 512 model works at about 1:1.
- Checked by eye on a contact sheet (`.test-media/renders/planb-contract/crops.png`): upright faces, teeth and eyes excluded by the weight of §5.
- Jitter of the transform, against a 5-frame average (motion included, so an upper bound): center 0.41 px, angle 0.05 degrees, scale 0.09 %. No extra smoothing of the transform in schema 1; `flicker_ratio` will show if that is wrong. The fastest step in these clips: 15.8 px and 3.35 degrees in one frame.
- CREMA-D is nearly frontal. Turned heads are not covered by these numbers.

### 1.7 Training on this machine

U-Net, batch 4 at 256, Adam, median step:

| Environment | p16 (2.1 MB) | p32 (8.6 MB) |
|---|---|---|
| `.venv-train`, CPU, 8 threads, `torch` 2.15.0 | 151 ms | 433 ms |
| `torch-directml` 0.2.5.dev240914, `torch` 2.4.1, RX 9070 XT | 21 ms | 32 ms |

- On `torch-directml`, these worked: `grid_sample`, bicubic `interpolate`, `avg_pool2d`, `InstanceNorm2d`, `GroupNorm`. Loss falls as on the CPU.
- **`torch.fft.fft2` kills the process** on `torch-directml` (`dml_util.cc: Invalid or unsupported data type ComplexFloat`, exit 127). It is not an exception, so losses must not use FFT there. Some optimizer steps warn that they fall back to the CPU.
- `torch-directml` pins `torch` 2.4.1, so it needs its own environment. `.venv-train` stays on the current CPU build, the same version a rented GPU machine gets. Export always happens in `.venv-train`.
- Teacher-sized training was not tried locally. A short run does not show that it would be stable for many hours.

---

## 2. Where the stage sits

- One effect named `neural`, `order = 90`, after `teeth` (75): the last step before the pipeline clips and converts the frame.
- **Input**: the frame as the classic effects left it (float32 BGR in [0, 1]). The model finishes the classic result. It does not replace it.
- **Output**: the classic frame plus a weighted delta (§5). With weight 0 the frame is the classic output, bit for bit, which
  is what makes "fade to the classic output" possible and cheap.
- `needs_face = True`, so the pipeline's face fade already reaches the stage through `ctx.intensity`.
- `ctx.landmarks` and the masks are used as `reshape` left them, like any later effect.
- The frame is the unmirrored camera frame. Mirroring happens after the stage.

---

## 3. Model tensors

All models, schema 1:

| | Input | Output |
|---|---|---|
| Name | `input` | `delta` |
| Shape | `[1, 3, S, S]`, fixed, no dynamic axes | `[1, 3, S, S]`, the same `S` |
| Type | float32 | float32 |
| Layout | NCHW | NCHW |
| Channel order | **B, G, R** (the app's order) | B, G, R |
| Range | [0, 1], sRGB-encoded as the camera gave it, no mean or std | [-1, 1] (end the graph with `Tanh` or `Clip`) |
| Meaning | the aligned crop of §4 | the change to add to the input, in the same crop |

- `S` is one of **256, 384, 512**. 512 is the target (§1.6): at the median face the crop side is 513 frame pixels, so
  512 is about 1:1, and the smaller sizes throw away detail the GPU can afford to keep (§1.3).
- **The output is a delta, not a picture.** All zeros means "no change", so the runtime can fade, clamp and sanity-check it, and
  frame detail the model never saw stays in the output. A model MUST be able to represent zero exactly.
- Normalization to any other range belongs inside the graph.
- Float16 is allowed **inside** the graph; the graph edges stay float32 (§1.5, §8).
- Training code that needs RGB (a perceptual loss, for example) swaps channels itself. It never changes the model's order.

### Temporal input: none in schema 1

The model is stateless: one crop in, one delta out. No previous frame, no previous output, no hidden state.

Reasons: frames may be skipped; a feedback loop can drift or blow up over a long call and is much harder to evaluate; and aligned
crops let the runtime do the smoothing. Temporal stability comes from a training loss (§10) and from the runtime's smoothing:

```
smoothed = smoothed + alpha * (delta - smoothed)      # in crop space, reset by Effect.reset()
```

`alpha` is `output.temporal_alpha` from the metadata (default 0.6, range (0, 1], 1 = off). A later schema may add a temporal input; the
runtime refuses any schema it does not know.

### Strength

```
frame_out = frame + clamp(smoothed, ±max_abs_delta) * edge_window  (warped to the frame) * weight * strength * ctx.intensity * health
```

- `strength`: the slider, `settings.neural.strength`, in [0, 1].
- `ctx.intensity`: master intensity x pipeline face fade x preset glide, as for every effect.
- `health`: the stage's own fade level in [0, 1] (§7.4).
- `motion`: how well the delta fits the face right now, in [0, 1] (§7.4). The formula below has it as a further factor.
- `weight` and `edge_window`: §5.
- The model never receives the strength. It always computes the full delta.

---

## 4. The aligned crop

A similarity transform (no shear) from landmarks that do not move with speech, gaze or expression: eye corners and nose base. It MUST NOT use
the lips, the chin or the iris. **Every part uses this one definition**; `crop.definition` in the metadata names it.

Normative code. The runtime agent puts it, exactly this, in `makeup_cam/neural/geometry.py` as its first step. That file imports only
numpy, cv2 and `makeup_cam.regions`, so it also imports under `.venv-train`. Evaluation and training import it from there and never copy it.

```python
CROP_DEFINITION = 1
SIDE_PER_FACE_SCALE = 4.5   # crop side = this x regions.face_scale
CENTER_BELOW_EYES = 0.4     # crop center below the eye midpoint, x face_scale
ALLOWED_SIZES = (256, 384, 512)


def crop_matrix(landmarks: np.ndarray, size: int) -> np.ndarray:
    """2x3 float64: frame pixel index -> crop pixel index, for cv2.warpAffine."""
    right = regions.eye_center(landmarks, "right").astype(np.float64)
    left = regions.eye_center(landmarks, "left").astype(np.float64)
    u = (left - right) / np.linalg.norm(left - right)  # along the eyes: image right for an upright face
    v = np.array([-u[1], u[0]])                        # image down for an upright face
    scale = float(regions.face_scale(landmarks))
    center = (left + right) / 2 + CENTER_BELOW_EYES * scale * v
    k = size / (SIDE_PER_FACE_SCALE * scale)
    a = np.array([u, v]) * k
    t = size / 2 - a @ center                 # continuous frame coordinates -> continuous crop coordinates
    return np.column_stack([a, t + a.sum(axis=1) * 0.5 - 0.5])  # OpenCV indexes pixel centers (regions.py: continuous)


def extract_crop(frame: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray:
    """The crop as float32 S x S x 3 BGR. ``frame`` is float32 BGR in [0, 1]."""
    m = crop_matrix(landmarks, size)
    reduction = SIDE_PER_FACE_SCALE * float(regions.face_scale(landmarks)) / size
    image, level = frame, 0
    while reduction / 2**level > 1.5:           # never sample more than 1.5x sparser: the crop would alias and flicker
        image, level = cv2.pyrDown(image), level + 1
    m = m.copy()
    m[:, :2] *= 2**level                        # cv2.pyrDown: half-size pixel i is centered on full-size pixel 2i
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
```

An implementation may run `pyrDown` on an even-aligned ROI for speed. The geometry MUST stay the same.

Golden values for `tests`, to catch a wrong copy (landmarks all zero except those listed; the matrix rounded to 6 decimals):

| Landmarks | `S` | `crop_matrix` |
|---|---|---|
| 33: (550, 360), 133: (610, 360), 362: (670, 360), 263: (730, 360), 2: (640, 430) | 512 | `[[0.948148, 0, -350.840741], [0, 0.948148, -130.87037]]` |
| 33: (493.6408, 381.246), 133: (564.5469, 393.7487), 362: (635.4531, 406.2513), 263: (706.3593, 418.754), 2: (585.4136, 482.7238) | 256 | `[[0.38906, 0.068602, -133.147926], [-0.068602, 0.38906, -9.558188]]` |

Compare within 1e-3. Checked: this code, run as written from this document, maps each output pixel to its expected source position on a ramp image with 0.000 px error, at reductions of 1.05 and 6.33 (3 levels of `pyrDown`).

---

## 5. Paste-back and what a model may not change

Computed for each frame, from **the current frame's landmarks**, in the frame ROI that the crop covers (on a helper thread when it is ready in time, else on the camera thread, with the same result: §7.2):

```python
s = regions.face_scale(lm)
allowed = regions.face_oval_mask(lm, roi, expand_px=0.03 * s, feather_px=0.05 * s)
for mask in (ctx.hair_mask, ctx.face_skin_mask):              # each may be None: then the oval alone decides
    if mask is not None:
        allowed = np.maximum(allowed, mask_trust * np.clip((mask[roi.slices] - 0.3) / 0.4, 0.0, 1.0))
for polygon in (regions.expand_polygon(lm[regions.LIPS_INNER], 0.01 * s), *eye_polygons_grown_by_0_005_s):   # the eyes unless may_edit_eyes
    touched = pixels_the_polygon_touches(polygon)             # exactly 0, also where it only partly covers a pixel
    weight = allowed * (1.0 - soft_edge_one_pixel_further_out(touched))
```

The normative code is `geometry.edit_weight`. Measured reasons for the two differences from a plain product of the masks:

- **Masks count only where they are sure** (`geometry.MASK_SURE`, *provisional*). MediaPipe's masks hold about 0.01 to 0.017 everywhere, which gave the whole crop's rectangle a weight above 0: 37.6 % of the frame on a large face, and at a large delta one level on 5.7 to 11.6 % of the background. Their soft rim also reaches onto the wall and the collar. Below 0.3 a mask now gives nothing.
- **`mask_trust` (`stage.MaskTrust`, *provisional*) weakens what the masks add while they are behind the face.** They come from a slower model: 78 ms old in the median at 30 Hz, and smoothed over earlier runs as well. In a fast pan their edge stays on the wall with a weight over 0.5, and `motion` (§7.4) does not see a pan. So: the distance of the centre between the eyes from where it was in the masks' frame, plus the change of the face scale, over the face scale; full trust up to 0.07, none from 0.14, moved like `motion` (§7.4). Measured over 3 frames on 8 women's clips: talking, 90th percentile 0.02 to 0.11; a pan of 30 px over 40 frames, median 0.08 to 0.15; of 45 px over 20 frames, 0.21 to 0.43. Offline (`sync`) it is 1. The price is that hair and hairline get no finish during a fast move of the head; the oval follows the current landmarks and keeps its own. Wall pixels (the truth computed afterwards on the same frames) with a weight over 0.5, p95, still / pan 30 / pan 45: 1012 from 39 / 684 / 1296 to 41 / 114 / 104, 1072 from 207 / 831 / 1613 to 107 / 184 / 154. 2D pans, not real heads.
- **The mouth and eye openings are cut out by every pixel they touch.** `expand_polygon` grows radially, so a thin mouth hardly grows across its height, and an anti-aliased edge would let a model change the rim of the teeth.

The delta goes to the frame with `cv2.warpAffine(delta, cv2.invertAffineTransform(m) shifted to the ROI, ..., flags=cv2.INTER_LINEAR,
borderMode=cv2.BORDER_CONSTANT)`, value 0.

**Edge window** (crop space, precomputed once per `S`, applied to the delta on the reader thread): `w(i) = smoothstep(min(i + 0.5, S - i - 0.5) / (0.08 * S))`, clamped to
[0, 1], and `edge_window = outer(w, w)`. The crop's border therefore never shows.

Rules, which the runtime enforces whatever the model outputs and the evaluation tool verifies:

| Region | Rule |
|---|---|
| Mouth interior and teeth (`LIPS_INNER`, grown by 0.01 x face scale) | **Weight exactly 0.** No feather inward: pixels there are the classic output, bit for bit. Teeth never change shape. |
| Eye openings (white, iris, grown by 0.005 x face scale) | Weight exactly 0, unless the metadata says `may_edit.eyes: true` and the evaluation covers it. Lids, lashes and brows are allowed. |
| Everything outside `allowed` (background, neck, clothes, hands) | Weight exactly 0. |
| Lips, skin, brows, lids, hair | Allowed, under the clamp. |
| Clamp | Each channel of the delta is limited to `±max_abs_delta` (metadata, at most **0.35**, default 0.25). |

Soft gates that multiply into `health` (§7.4), so they fade rather than switch (*provisional*; CREMA-D is nearly frontal, so these were not
measured on turned heads):

| Condition | Full strength | Zero strength |
|---|---|---|
| Face size: `face_scale` in frame pixels (a small face has little detail for the model) | ≥ 60 | ≤ 35 |
| Share of `FACE_OVAL` points outside the frame | 0 | ≥ 0.15 |
| Head turned: `min(dL, dR) / max(dL, dR)`, `dL`, `dR` = nose tip (1) to each eye center | ≥ 0.6 | ≤ 0.35 |

---

## 6. Model files

### 6.1 Where they live

```
<user_data_dir()>/neural_models/<name>-<version>/model.onnx
                                                 model.json
                                                 evaluation.json      (written by the evaluation tool)
<user_data_dir()>/neural_models/active.json                           {"model": "<name>-<version>"}
```

- `AI_MAKEUP_CAM_NEURAL_DIR` overrides the `neural_models` folder. It is separate from `AI_MAKEUP_CAM_MODELS_DIR` (MediaPipe).
- Stand-ins and anything else made during development go under `.test-media/planb/` (ignored by git).
- **Nothing here is ever tracked**: no `.onnx`, `.pt`, no weights of any kind in a tracked path.
- A single self-contained `.onnx` file. No external data file (§8).
- Which model is active is not part of a preset: presets stay shareable and the sections stay plain numbers.

### 6.2 `model.json`

```json
{
  "schema": 1,
  "name": "finish-student",
  "version": "0.1.0",
  "file": "model.onnx",
  "sha256": "<hex of model.onnx>",
  "size_bytes": 8570000,
  "opset": 20,
  "stand_in": true,
  "input": {"name": "input", "size": 512, "layout": "NCHW", "colors": "BGR", "range": [0.0, 1.0], "dtype": "float32", "stage": "after_classic"},
  "output": {"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "range": [-1.0, 1.0], "dtype": "float32", "max_abs_delta": 0.25, "temporal_alpha": 0.6},
  "crop": {"definition": 1},
  "may_edit": {"eyes": false},
  "training_data": {"kind": "none", "input_preset": "natural", "note": "technical note, no personal details"},
  "evaluation": {"verdict": "none", "report": null, "report_sha256": null, "model_sha256": null, "date": null}
}
```

- `training_data.kind`: `none`, `classic_pipeline`, `3d_render`, `recording` or `mixed`. `input_preset` is the preset the input side of the pairs went through.
- `stand_in` is `true` for any model not trained on real paired data. Only training on `3d_render`, `recording` or `mixed` may set it to `false`.
- `evaluation.verdict`: `none`, `pass`, `fail` or `incomplete`. Only the evaluation tool writes this block, atomically.
- Unknown extra keys are ignored. An unknown `schema` or `crop.definition` means the model is refused.

### 6.3 The gate

A model may reach a virtual camera only if **all** hold, checked when it loads (on a background thread, never on the camera thread):

1. `model.json` parses, and `schema` and `crop.definition` are known.
2. `sha256(model.onnx)` equals `sha256` and `evaluation.model_sha256`.
3. `evaluation.verdict == "pass"` and `sha256(evaluation.json)` equals `evaluation.report_sha256`.
4. `stand_in` is `false`.

Otherwise the stage stays off and the GUI says why (§11). The override is `allow_unevaluated=True` on the effect, which only the offline modes
(`--image`, `--video`) and tests may pass; `--headless` and the GUI cannot. Offline, the result is labelled as such on stderr.

---

## 7. Runtime

### 7.1 Worker process, and why

Inference runs in **its own process**, started with `subprocess.Popen([sys.executable, "-m", "makeup_cam.neural.worker", ...])`. Do not use `multiprocessing`:
it would import the app's `__main__` (and with it Qt and MediaPipe) in the child.

The reason is measured (§1.4): after a device removal, a new session in the same process never gets the GPU back, and it silently lands on the CPU. A fresh process gets the
GPU back at once. It also means a driver crash or a hang costs the look, not the call: killing the worker takes about 25 ms, and a new one is ready in about 0.4 s.
The process boundary costs about 0.5 ms per frame.

- `makeup_cam/neural/__init__.py` and `worker.py` MUST NOT import Qt, MediaPipe or anything else heavy at import time.
- The worker imports `numpy` at import time, and `onnxruntime` and the result filter (`cv2`, §7.5) only when it runs.
- Transport: one `multiprocessing.shared_memory` block (input tensor, then the filtered result) plus binary pipes on stdin and stdout carrying 8-byte sequence numbers.
  Status and errors go on the same pipe as negative codes. ORT's Python wrapper prints `EP Error` to stdout, so the worker MUST keep the original stdout handle for the protocol and point `sys.stdout` at stderr before importing ORT. Stderr goes to the log.
- **Parent death**: EOF on stdin makes the worker exit, from before the session is made: a driver can hang there too. No orphan processes. Closing the stage also reaches a worker that is still starting.
- `subprocess.CREATE_NO_WINDOW` on Windows. The worker inherits the app's CPU affinity (`pipeline._use_one_thread_per_core`). Check `importlib.util.find_spec("onnxruntime")` before starting anything.
- A `Backend` protocol (`start`, `submit`, `latest`, `close`) lets tests and the `.venv` environment use a fake without any ONNX package.

Session settings in the worker:

```python
so = ort.SessionOptions()
so.enable_mem_pattern = False
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.log_severity_level = 3
session = ort.InferenceSession(path, so, providers=["DmlExecutionProvider"])
if session.get_providers()[0] != "DmlExecutionProvider":
    ...  # report failure and exit: ORT falls back to the CPU silently, even with this provider list
```

**There is no CPU inference during a call.** The stage's fallback is the classic output. Then run 5 warm-up runs on a zero tensor, and only then report ready.

### 7.2 Threads

| Thread | Does | Must never |
|---|---|---|
| Camera/output thread, in `NeuralEffect.apply` | `extract_crop`, copy into shared memory, write 8 bytes to the pipe if no job is in flight, add the prepared paste (or build it, if the helper is not done) | wait for a result, take a lock that another thread holds during slow work, start or kill a process, read files, compute a hash |
| Paste helper (daemon), started by the pipeline through `start_ahead` once landmarks and masks are final, unless the stage is next itself: then nothing would run in the meantime | build the weight, warp the newest delta, multiply. May wait up to 15 ms for the job in flight | touch the frame. Its result is used only if it is for these very landmarks and arrived in time; the camera thread never waits for it |
| Worker process | run the model; check the output (§7.5); clamp, window, smooth, in crop space; write the result to shared memory and signal. Measured: at 512 this takes 1.2 ms, and in the app's process it cost the classic effects next to it 0.5 ms per frame, for the GIL | |
| Reader thread (daemon) | block on the worker's pipe; copy the result out of shared memory; check that it is finite; publish it by swapping one reference | touch the frame |
| Control thread (daemon, started when the stage is first enabled, or by a failure) | check the gate, start, watch, kill and restart the worker | run on a frame's time |

- **At most one job in flight.** If the worker is still busy when a frame comes, that frame submits nothing. So the pipe write of 8 bytes cannot block and the input buffer is never torn.
- **No waiting for the current frame: `NEURAL_WAIT_MS = 0`.** Each frame is drawn with the newest finished delta, normally from the frame before (age 1). The delta lives in crop space, so it follows the face through the current frame's landmarks.
  Measured: waiting up to 8 ms for the current result costs 2 fps at `natural` (§1.3). Evaluation therefore runs at age 1, so any lag artefact shows up there.
- Offline modes (`--image`, `--video`, the evaluation's quality pass where stated) may use `sync=True`: wait for the current result (age 0), with a 2 s timeout, then fall back to classic.

### 7.3 Latency budget

At 1280x720, 30 fps, `natural`, with the app's own load present. The first three are the evaluation's thresholds (§9).

| What | Budget | Measured with the stand-ins (§1) |
|---|---|---|
| Model round trip (copy in, run, copy out, signal back), p95 | ≤ 12 ms | p32 @512: 7.24; p64 @512: 11.29 |
| Same, p99 | ≤ 20 ms | |
| Stage on the camera thread (crop, signal, mask, paste), p95 | ≤ 5 ms | 3.44 to 4.35 |
| Lost frame rate against classic alone | ≤ 0.5 fps | **On the slowest clips only by the stage giving way.** See below |
| GPU memory of the worker | ≤ 1.5 GB | 604 MB at p32 @512; 1168 MB at p64 @512 |
| Worker ready after start | ≤ 3 s | about 0.4 s |

Size guidance: of the students with `InstanceNormalization` (§1.2), p32 @512 float16 is the largest with room to spare. p32 @512 float32 and p48 @384 float16 are inside with little room, and p40 @512 and up are over 12 ms. Without normalization a 34 MB U-Net at 512 is only just inside; that is not what `training/` makes.

**The frame-rate budget is kept by the stage giving way, not by there being room.** On the slowest clips the classic effects take 29 to 31 ms of the 33.3, and what the stage costs is more than is left on a bad minute. The GPU's time plays no part, so a smaller model hardly helps.

What it costs on the large face of 1072 at `natural`, in a run that is out of room: 0.65 to 0.85 ms on the camera thread, and about 1.3 ms more for `hair`, which runs next to the paste helper. Of that 1.3, measured by giving the helper one part at a time: 0.3 for the oval, 0.1 for the masks, 0.3 for the cut-outs, 0.3 for the warp and the multiply. It is neither the size of the paste (no masks: the same), nor the Python steps (the cut-outs in one go: the same), nor OpenCV's thread pool (the warp in strips below its threshold: the same). The rest of what used to be there is gone: the crop's pyramid only where the crop reaches, the numpy clip of the oval on its own box, the copy of the result without the GIL.

So when frames are lost the stage does less (§7.4): a job on every second frame, then a pause. The two steps were chosen over more of the same because the frame rate decides, not the milliseconds: near the limit it stays at 30 and then falls by 1 fps within about 1 ms.

Loss against classic in fps, stand-ins, nothing else running (all CPUs 17 to 20 % busy). The evaluation's clock: 3 rounds of 15 s, with 1.5 ms for the delivery. The harness: 60 s, about 2 ms in the output thread. A pause means that the finish was off for most of the run, and the evaluation fails it.

| | Clip | p16 @256 | p32 @512 float16 |
|---|---|---|---|
| Clock, on its own, first set | 1072 | 0.38, 0.41, 0.37; paused in all 3 | 0.50, 0.44, 0.51; paused in all 3 |
| | 1073 | 0.20, 0.07, 0.16 | 0.53, 0.34, 0.15; paused in 1 |
| Clock, on its own, the same again | 1072 | 0.04, 0.03, 0.01 | |
| | 1073 | -0.02, -0.02, 0.01 | |
| Clock, inside a full evaluation | 1073 | 0.58, 0.65, 0.64; paused in all 3 | |
| | 1082 (middling) | 0.18, 0.27, 0.17 | |
| Harness, 60 s | 1072 (classic 29.97) | 0.33; paused twice | 0.27; paused twice |
| | 1073 (classic 29.68 to 29.80) | 0.37; paused twice | 0.17; paused twice |

The same model on the same clip can still pass in one set and fail in another, with classic itself between 30.0 and 31.4 ms from one set to the next. The cause of that drift is not known: it is not what a quality pass of 16 clips leaves behind in the process. On a middling face there is room, and nothing of this happens. **A classic pipeline faster by about 1.5 ms on the slowest clips would remove the need for all of it**, and that is the open decision. Not measured: the real virtual camera. `DELIVERY_MS` is an estimate.

### 7.4 What happens when something goes wrong

`health` is the stage's fade level. It multiplies the strength. All changes are ramps, so the picture never jumps.

- Rising: `+1/15` per frame (0.5 s).
- Falling: `-1/10` per frame (0.33 s).
- `health` ramps toward a target: 0 while the stage is failing or the soft gates of §5 say so, otherwise the gates' value.
- While falling, the last good delta keeps following the face. At `health == 0` the frame is exactly the classic output and the mask and paste work is skipped; the crop is made only while jobs are being submitted, so that the stage can come back.
- With the section disabled or the strength at 0, `apply` is not called or returns at once, and no jobs are sent (the worker idles without using the GPU).

| Event | Detected by | Action |
|---|---|---|
| Result is late: age 2 or 3 (at most 100 ms) | reader thread's sequence number | Reuse the newest delta. `health` unchanged; `motion` below decides how much of it fits. |
| **The delta does not fit the face** | camera thread: `motion_points` now against those of the delta's frame, in crop space, x (age + (1 - alpha) / alpha) / age, smoothed by 0.5 | `motion` falls by at most 1/4 per frame, from 1 at 0.05 x face scale to 0 at 0.10 (*provisional*), and rises by 1/15. It multiplies the paste like `health`, and is no failure: no event, no restart. |
| Age ≥ 4 | same | `health` falls. It rises again when results return. |
| **Frames lost**: frames the camera announced (`MakeupPipeline.prepare`) that were never rendered, 3 within 5 s (0.6 fps) | `ctx.skipped_frames`, counted while the stage is working | A job on every second frame only (age 1 and 2). Back to every frame after 120 s without such a loss. Not a failure. |
| Frames still lost: 6 within 10 s, after that | same, control thread | The overload action below: the budget of §7.3 comes first, and the classic output is what is left. The evaluation fails a model for this. |
| **Overload**: over 20 % of the frames in the last 5 s submitted nothing, or their round trip exceeded 33 ms | counters on the reader thread | `health` falls; no jobs for 30 s; then try again. Status text (§11). Doubles to at most 5 min. |
| Worker silent: a job in flight for over **1 s** (driver reset, hang) | control thread | `health` falls (it already is); kill the worker; go to restart. |
| **Any exception** in the worker's `run`, or provider not DML | worker reports a code and exits | Go to restart. Catch `Exception`, not ORT's types. After a device removal, on a Japanese Windows `run` raises `UnicodeDecodeError` in 0.2 ms, so text matching is useless too. |
| Worker process died (crash) | EOF on the pipe | Go to restart. |
| Output is bad (§7.5) | worker, which reports the job as rejected | Discard it, age keeps growing. 3 in a row: go to restart. |
| Frame size changed, or `reset()` | pipeline | Drop the delta and the smoothing. Warm up by rising from 0. |
| Restart | control thread | Waits 5 s, then 30 s, then 120 s. After 3 failures within 10 min: `failed` until the user re-enables the stage or restarts the app. A restart needs 10 good results before `health` rises. |
| Model missing or refused, no `onnxruntime`, no DML | control thread at start | Stage stays off. No error dialog. Status text. |

- The camera thread does nothing but read `health` and the newest delta; every action above happens on another thread.
- `motion` and `mask_trust` move by `stage._Ease`: speed limited (`motion`: 1/4 down, 1/15 up), change of speed limited to 1/8 per frame, slowing down before the target. A level that turned within one frame (from falling by 1/4 to rising by 1/15) showed as a flash of 3 levels over a delta of 10 to 12 levels: all 22 breaks of a 30-minute run of a stand-in with a large delta were at such a corner, and they stopped with this (`docs/dev/evaluation.md`). It costs a nod about one frame.
- `motion`, measured: the crop follows the eyes, so a pan costs nothing, while a nod moves the mouth and the chin against them. A 2D warp at 14 px per frame put the paste 0.035 to 0.04 x face scale off the mouth at age 1 with `alpha` 0.6, and 0.07 to 0.09 at age 3. On 8 clips of women talking, the paste stays at 0.977 of its strength on average (0.5 % of the frames under half) at age 1; with that nod 0.634 (31 % under half); with a bow at 23 px per frame 0.054. Real heads were not tried.
- `apply` never raises in normal failure; if it does, the pipeline's isolation skips the step for that frame (a jump only for that frame, so it must be treated as a bug).
- Closing: `MakeupPipeline.close` asks effects that have a `close()`; it closes stdin, kills after 200 ms, on a daemon thread. Never join on the camera thread.

### 7.5 Checks on each result (in the worker)

Discard the result if any value is not finite, or if the mean of `|delta|` over the crop's central 50 % exceeds **0.15** (*provisional*). A NaN input gives no exception and may give finite output, so check the output.

---

## 8. Export and operator rules

- Exporter: `torch.onnx.export(model, (x,), path, dynamo=True, opset_version=20, external_data=False, verbose=False, input_names=["input"], output_names=["delta"])`. Static shape: dynamic axes are 3 to 4 times slower (§1.5). `opset` MUST be in **17..21**.
  - **Opset 23 must not be used: DirectML then runs `Conv` on the CPU and a run takes 44 ms instead of 1.2 ms** (§1.5), and nothing warns about it.
  - Keep `verbose=False`, or set `PYTHONUTF8=1`: on a Japanese Windows the exporter's progress output has a character cp932 cannot encode, and it raises `UnicodeEncodeError`.
  - `external_data` defaults to true and writes a separate `.data` file, which is not allowed (§6.1).
- All kernels MUST run on DirectML. The evaluation verifies this with `SessionOptions.enable_profiling` (§9).
- Allowed ops, verified on this GPU with outputs matching the CPU's: `Conv` (with groups, dilation, stride), `ConvTranspose`, `Resize` (`linear` with `half_pixel`, `pytorch_half_pixel`, `align_corners`; `nearest`), `Concat`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Relu`, `LeakyRelu`, `PRelu`, `Sigmoid`, `Tanh`, `HardSwish`, `HardSigmoid`, `Elu`, `Selu`, `Softplus`, `Softsign`, `Mish`, `Clip`, `BatchNormalization`, `InstanceNormalization`, `AveragePool`, `MaxPool`, `GlobalAveragePool`, `ReduceMean`, `DepthToSpace`, `SpaceToDepth`, `Pad`, `Slice`, `Split`, `Transpose`, `Reshape`, `Softmax`, `MatMul`, `GridSample`, `Einsum`, `Cast` (float16 and float32).
- Forbidden: **`Resize` with `cubic`** (falls to the CPU: 202 ms), **`Gelu`** (CPU), anything that shows CPU kernels in the profile. `GroupNormalization` leaves small shape nodes on the CPU: use `InstanceNormalization` or `BatchNormalization`.
- Float16 inside the graph: half the file, and 30 to 50 % less time (§1.5). Wrap the model so that it casts at the edges. Check the look with it, since the numbers differ slightly from float32.
- `LayerNormalization` is not verified. Check it and add it here before use.
- Export check (training side, CPU ORT in `.venv-train`): loads, shapes match, output equals PyTorch within 1e-3 (float32) on a real crop, ops in the allow-list, single file.

---

## 9. Evaluation tool

`python -m makeup_cam.evaluate` under `.venv-neural`. It only reads the app's code and writes under the model's folder and `--out`. Thresholds marked *provisional* are set without a real model; the numbers will move once one exists, and the tool keeps them in one table at the top of the module.

### 9.1 Interface

```
python -m makeup_cam.evaluate --model DIR [--clips GLOB ...] [--preset NAME] [--out DIR]
       [--frames-per-clip N] [--soak-minutes M] [--identity-model PATH] [--identity-licence TEXT] [--deepfake-report FILE] [--no-update-metadata]
       [--measure-fps] [--fps-seconds S] [--strength X] [--settings FILE] [--rendered FILE] [--reference FILE] [--no-sway]
```

| Argument | Meaning |
|---|---|
| `--model DIR` | folder with `model.onnx` and `model.json` |
| `--clips` | default `.test-media/testvids/*.flv` (all 11 actors); also takes recordings kept outside the repository |
| `--preset` | default: `training_data.input_preset` |
| `--out` | default `DIR`; files `evaluation.json`, plus contact sheets under `.test-media/` or `--out` when given (not tracked) |
| `--frames-per-clip` | default all |
| `--soak-minutes` | real-time run of that length on the real GPU; default 0 (not run, then verdict at best `incomplete`) |
| `--identity-model` | optional face-recognition model; absent means that metric is `not_run` |
| `--deepfake-report` | optional JSON from a separate detector run; absent means `not_run` |
| `--no-update-metadata` | do not write `model.json.evaluation` |

Rules: clips are upscaled to 960x720 and padded to 1280x720 as in the other video checks. Reference frames are the classic pipeline's output for the same frames. It runs the real pipeline and the real worker. It uses women's material only. No pretrained model may be required by its tests: fakes stand in for them; any pretrained model must have its licence checked and stated in the report and be stored outside tracked paths.

Exit code: 0 `pass`, 1 `fail`, 2 `incomplete`, 3 could not run.

### 9.2 Metrics and thresholds

Worst clip counts, not only the mean: the actors differ in skin tone and framing, and a look that breaks on one of them fails. The table below is the first proposal; **the values in force, with what each is based on, are `evaluate.THRESHOLDS` and `docs/dev/evaluation.md`.** Differences so far: `flicker_ratio` 1.08 / 1.15, `mouth_band_edge_corr` 0.90, `landmark_shift_p99` 0.06, and:

- `identity_cosine_p01` is the output against the bare frame, ≥ 0.70, with or without a model. "At most 0.02 below classic" rejected any added makeup: a stand-in that imitates `natural` from `off` lost 0.028 to 0.059, as much as a nose 30 % wider. The drop against classic is still reported.
- New, because the temporal measures cannot see what holds still: `warp_flow_px` (content that moved, on calm skin), `seam_edge` (steps and grids in the change where the picture has no edge), `side_cast`, `skin_cast`, `rim_change` (paint on a mask's soft rim), `frame_locked_ratio` (a change that holds still on the frame, not on the moving face), `paste_drift_p99` (§7.4).
- `fps_loss` and the other numbers from the clock are read on the clip whose classic effects take longest and on a middling one, at the preset under evaluation and at `natural` (§7.3), with classic runs before and after each of 3 model runs (`--fps-rounds`); the worst run counts, and a stage that paused to keep the frame rate does not pass. The clips for `natural` come from a short timing of all clips at `natural` when the preset under evaluation is another one: at `off` all clips cost the same. After each frame the output thread spends 1.5 ms (`DELIVERY_MS`), an estimate of what the app's virtual camera and preview cost and this tool does not open. Near the limit of the frame time the frame rate does not fall gradually, so without it the same model on the same clip read -0.02 to 1.14 fps.
- `breaks_per_30min`: a spike counts only beyond a quarter of what the reference itself did at that place, as this table says: "while classic shows none". A student's change follows a blink: 3 levels where the picture jumped by 26.
- `deepfake_margin` has no threshold and never takes part in the verdict.
- The quality pass also runs clips with a sway (8) and a nod and a bow (4).

| Key | What | Threshold |
|---|---|---|
| `flicker_ratio` | Mean `|f(t+1) - 2 f(t) + f(t-1)|` over `allowed` minus `keep_out`, in crop space at the frame's own transform, divided by the same for the classic output. Median over clips / worst clip. | ≤ 1.10 / ≤ 1.25 *provisional* |
| `mouth_interior_max` | Largest change against classic inside the `keep_out` mouth polygon, all frames, uint8 levels. | ≤ 0 (exact) |
| `mouth_band_edge_corr` | Gradient-magnitude correlation against classic in a ring 0.04 x face scale around the inner lip polygon, on open-mouth frames. Catches a change to the teeth's shape at the edge. | ≥ 0.98 *provisional* |
| `eye_interior_max` | The same for the eye openings, unless `may_edit.eyes`. | ≤ 0 (exact) |
| `outside_max` | Largest change where `weight == 0` outside the polygons. | ≤ 0 (exact) |
| `landmark_shift_p99` | Landmarks re-detected on the output against the classic output, x face scale. Always available geometric proxy for "still the same face". | ≤ 0.015 *provisional* |
| `identity_cosine_p01` | Embedding cosine, output against bare frame, 1st percentile. Needs `--identity-model`. The classic result is the reference. | ≥ (same value for classic) - 0.02 *provisional* |
| `delta_abs_p99` | 99th percentile of the applied change on skin. A guard against a look that is merely heavy. | ≤ 0.15 *provisional* |
| `breaks_per_30min` | Events in the soak run: a one-frame spike in crop-space change over 6 x the running median while classic shows none; a rejected result; a fall of `health`; a worker restart. Must be real time, ≥ 30 min. | ≤ 1 |
| `deepfake_margin` | From `--deepfake-report`: detector's fake score for the output minus the same for classic. | ≤ 0.05 *provisional* |
| `all_kernels_dml` | From ORT's profile: no node on the CPU. | true |
| `rtt_p95_ms`, `rtt_p99_ms` | Worker round trip next to the running app. | ≤ 12, ≤ 20 |
| `stage_p95_ms` | Stage cost on the camera thread. | ≤ 5 |
| `fps_loss` | Delivered frame rate against classic alone, same clip. | ≤ 0.5 |
| `miss_pct` | Frames drawn with age ≥ 2. | ≤ 0.5 |
| `vram_mb` | Worker's `GPU Process Memory\Dedicated Usage`. | ≤ 1500 |
| `strength_zero_identical` | Strength 0 (and `health` 0): output equals classic, bit for bit. | true |
| `strength_monotonic` | Mean applied change grows with 0.25, 0.5, 1.0. | true |
| `fade_drill` | Kill the worker mid-clip, and force a run error and a bad output: no frame missing, frame step during the fade no larger than 1.5 x the frame step without it, classic reached, recovery works. | true |

### 9.3 Verdict and report

- `fail`: any measured metric misses its threshold.
- `incomplete`: nothing failed, but a metric is `not_run` (identity, deepfake, or a soak under 30 minutes).
- `pass`: everything measured and within thresholds. A `stand_in` model gets its verdict recorded, with `"eligible_for_calls": false` in all cases.

`evaluation.json`:

```json
{
  "schema": 1,
  "model": {"name": "...", "version": "...", "sha256": "...", "stand_in": true},
  "verdict": "incomplete",
  "eligible_for_calls": false,
  "date": "2026-11-06T12:00:00",
  "conditions": {"preset": "natural", "clips": ["1002_IEO_HAP_HI", "..."], "frames": 0, "frame_size": [1280, 720], "soak_minutes": 0,
                 "gpu": "...", "onnxruntime": "1.24.4", "code": "<git hash or null>", "pretrained": [{"name": "...", "licence": "..."}]},
  "metrics": {"flicker_ratio": {"value": 1.04, "worst": 1.12, "worst_clip": "1076_ITS_HAP_XX", "threshold": 1.10, "worst_threshold": 1.25, "passed": true, "status": "measured"}},
  "notes": []
}
```

- `status` is `measured` or `not_run`. A metric with no data never counts as passed.
- Clip names, not paths. No user names anywhere in the file.

---

## 10. Training code

- Own folder `training/`, own environment `.venv-train` (`training/requirements.txt`). Imports from `makeup_cam` are limited to `makeup_cam.neural.geometry` and `makeup_cam.regions`, and these need only numpy and cv2. **`mediapipe` is not in `.venv-train`**, so landmarks are computed once in `.venv-neural` by a dataset tool and stored.
- The same code runs on the local CPU (small smoke runs and tests), on rented NVIDIA GPUs (`--device cuda`), and in an optional `torch-directml` environment (`--device dml`, §1.7, `torch` 2.4.1: stay compatible, and do not use FFT or complex numbers in anything that can run there). No cloud SDKs, host names, keys, or anything specific to a rental in the code.
- Data stays outside tracked paths. Dataset root is an argument. Every data path comes from an argument or an env var.

Dataset layout (one folder per sequence, consecutive frames, so temporal losses can work):

```
<root>/<sequence>/manifest.json     {"kind": "3d_render|recording|classic_pipeline", "input_preset": "...", "fps": 30, "frames": N}
<root>/<sequence>/input/000000.png  the input side, already through the classic pipeline with input_preset
<root>/<sequence>/target/000000.png the made-up side, aligned pixel for pixel to input
<root>/<sequence>/landmarks.npy     (N, 478, 2) float32, of the input side
```

Rules for what is learned:

- Crops by `extract_crop` from `landmarks.npy`, with small jitter of the landmarks as augmentation.
- Target delta `= target_crop - input_crop`. The model outputs a delta.
- **Regions**: the loss is masked with the §5 `weight`, rasterized in crop space. Inside `keep_out` and outside `allowed` the target delta is 0 with a strong penalty, so that a model does not lean on the runtime's mask.
- Identity is protected by limiting the delta (a magnitude penalty), by a low-frequency bias, and by never warping geometry: a delta only changes colour.
- Temporal: a loss on the difference of the deltas of consecutive frames (after alignment by the crop), so the output is stable without any temporal input.
- Teacher: any size and architecture, any input it likes, PyTorch only, never exported. Student: must meet §3 and §8. Distillation: student delta against teacher delta, plus the masked loss against the target.
- `export`: writes `model.onnx` and `model.json` (§6.2) to a folder, under `.test-media/` by default, running the check in §8. `training_data.note` is free text; do not put personal details in it.
- Stand-in for plumbing tests: pairs from CREMA-D through the classic pipeline: input = `off`, target = `natural`, `training_data.kind = "classic_pipeline"`, `stand_in: true`. This proves the code path and says nothing about any look.

Local training facts: §1.

---

## 11. Settings and GUI

```python
@dataclass
class NeuralSettings:
    enabled: bool = flag(False, "学習モデルの仕上げを使う", help="合格したモデルがあるときだけ働きます")
    strength: float = num(0.5, 0.0, 1.0, "仕上げの強さ", help="0 で従来のメイクと同じになります")
```

- Section name `neural`, label `"学習モデルの仕上げ"`, default **off**. With it off, or with no optional package or model, the output MUST NOT change, and no process starts.
- Built-in presets do not enable it. `from_dict` already gives a missing section its defaults, so preset files need no edit.
- `pipeline.OPTIONAL_MASK_USERS` gains `"neural"`.
- The status goes in `RunnerStatus`, with these strings in the GUI's section: 「モデルがありません」「評価に合格したモデルがありません」「GPU が使えないため、従来のメイクのまま動いています」「処理が間に合わないため、一時的に弱めています」「エラーのため止めました。もう一度オンにすると再開します」「準備しています…」「動作中」.

---

## 12. File ownership

No agent edits a file owned by another. What one part needs from another goes in its report under open issues.

| Agent | Owns | May also edit, minimally |
|---|---|---|
| Runtime | `makeup_cam/neural/` (`geometry.py` first), `makeup_cam/effects/neural.py` (a thin registering shim), `tests/test_neural_*.py` | `config.py` (the section), `effects/__init__.py` (the module name), `pipeline.py` (`OPTIONAL_MASK_USERS`, closing effects), `runner.py` (status), `gui.py` (status line), `presets.py` (only if needed), `README.md` (a section), `__main__.py` (offline flags) |
| Evaluation | `makeup_cam/evaluate.py`, `makeup_cam/evaluate_validate.py`, `docs/dev/evaluation.md`, `tests/test_evaluate*.py` | none |
| Training | `training/` (`requirements.txt` already exists and may be changed), `tests/test_training*.py` (`pyproject.toml` collects `tests/` only; use `pytest.importorskip("torch")`) | none |

- `docs/dev/contract-plan-b.md`, `requirements-neural.txt`, `.gitignore`: set by the contract step. Propose changes in reports.
- Evaluation and training read `makeup_cam/neural/geometry.py` and the worker's protocol, and do not edit them.
- Nobody changes git state. The main session commits.

---

## 13. Rules for tests and environments

- The whole suite under `.venv` (no `onnxruntime`, no `onnx`) stays green, and Plan B tests there use fakes or skip with `pytest.importorskip`. Existing tests are not weakened; a changed test is named in the report with the reason.
- Under `.venv-neural`, tests build tiny models with `onnx.helper` in `tmp_path`. A test that needs the real GPU is marked and skipped when `DmlExecutionProvider` is not listed.
- Test media: `.test-media/` only. Women's material only. No test needs a pretrained model or a stand-in on disk.
- Run: `export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen AI_MAKEUP_CAM_MODELS_DIR="$LOCALAPPDATA/ai-makeup-cam/models"; .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --tb=short`.
- The virtual camera is not opened by any Plan B test or tool.
