# FPS contract: every source frame reaches the virtual camera

Binding rules for the build that makes the app keep 30 fps on every preset and every face.
Everything here is technical. Do not add personal details, absolute paths with a user name, or reasons for wanting
the feature to this file or to anything else that is tracked.

Keywords: MUST and MUST NOT are binding. Numbers come from the prototype in `.test-media/qa/fps/proto/` (not tracked)
unless marked *calculated* or *judgement*. Change a number only through this file, and say what measurement justified it.

---

## 1. Result

A pipeline of processes with shared memory, plus a separate process that writes to OBS Virtual Camera on an exact grid,
passes the bar on all three presets on all three faces, with the output bit-identical to the in-process chain:

    main process      camera thread, tracker, segmentation, plan of each frame (as today) + to_float into shared memory
    stage 0 process   reshape, hairshape
    stage 1 process   hair, skin, contour
    stage 2 process   brows, eyeshadow, tearbag, eyeliner, blush, lips, teeth, neural; to uint8
    sender process    mirror, BGR to NV12, one `pyvirtualcam.Camera.send` per place on a 33.3333 ms grid

The bar passes only under the condition of §2.2: the reader's OBS filter must keep the timer resolution it asks for, which a
visible or audible call app has and the bar's hidden reader does not always have. §14 lists what remains unproved. Read it.

---

## 2. Measured facts

Machine: Windows 11, Ryzen 7 9800X3D (8 cores, 16 threads), Python 3.12, 1280x720, 30.00 fps MJPEG clips with the frame number on
the bottom edge, headless. Apps of the desktop session were open during the measurements (a browser, a chat app, a game launcher):
the machine was not idle, which is the condition a real call has. Verdicts are from `bar_run.py` and `judge.py`, unchanged.

### 2.1 Before and after

`today` is the bar before this work, with the reader as it was. `now` is 60 s per run through the same tool, final settings.

| Face | Preset | today fps | now fps | now lost | Stage p50 (ms) | Stage p99 (ms) | Stage max (ms) | CPU today | CPU now (logical) |
|---|---|---|---|---|---|---|---|---|---|
| 1072 | natural | 24.9 | 30.0 | 0 | 6.8 / 12.9 / 15.2 | 9.0 / 14.2 / 17.6 | 10.4 / 15.3 / 18.6 | 2.23 | 2.84 |
| 1072 | polished | 24.5 | 30.0 | 0 | 6.8 / 13.0 / 16.2 | 9.0 / 14.5 / 18.8 | 10.2 / 15.9 / 20.1 | 2.24 | 2.90 |
| 1072 | strong | 20.8 | 30.0 | 0 | 12.9 / 12.9 / 16.6 | 15.9 / 14.5 / 19.1 | 17.6 / 15.6 / 22.8 | 2.40 | 3.11 |
| 1060 | natural | 28.2 | 30.0 | 0 | 4.1 / 11.3 / 16.2 | 5.0 / 12.9 / 18.2 | 5.7 / 14.2 / 21.8 | 2.22 | 2.50 |
| 1060 | polished | 27.2 | 30.0 | 0 | 4.0 / 11.4 / 17.5 | 4.9 / 13.1 / 19.5 | 5.5 / 14.0 / 24.8 | 2.19 | 2.56 |
| 1060 | strong | 23.0 | 30.0 | 0 | 10.8 / 11.3 / 18.1 | 12.9 / 13.0 / 20.4 | 14.6 / 14.2 / 22.3 | 2.37 | 3.19 |
| 1012 | natural | 25.9 | 30.0 | 0 | 3.6 / 9.4 / 11.2 | 4.4 / 10.3 / 13.1 | 5.1 / 11.0 / 14.7 | 1.86 | 2.15 |
| 1012 | polished | 29.9 | 30.0 | 0 | 3.6 / 9.4 / 11.9 | 4.3 / 10.3 / 13.6 | 4.9 / 11.3 / 14.5 | 1.93 | 2.15 |
| 1012 | strong | 29.5 | 30.0 | 0 | 7.6 / 9.3 / 11.8 | 9.3 / 10.2 / 13.8 | 10.1 / 11.5 / 14.5 | 2.22 | 2.52 |

Worst 1 s window 30 in all nine, no slip and no drop (§6). The worst stage p99 is 20.4 ms and the worst stage time 24.8 ms against a
budget of 33.3 ms. Stage time includes the boundary of §3.

28 minutes on 1072 strong through the same tool, 51000 frames in the judged part: `pass`, lost 0, mean 30.00 fps, worst 1 s window 29, biggest gap 66.2 ms, 2 slips, 0 drops, stage p99 17.3 / 16.2 / 23.0, stage max 34.7 / 32.5 / 55.2, CPU 3.27. Each slip added 33 ms of delay for the rest of the run: to write 84 ms at the start, 150 ms at the end of it (§6). So over a long call the delay can only grow, up to the bound of `SPARE_FRAMES`.

### 2.2 The reader's timer: a condition of the bar

The OBS filter runs in the process that reads the camera. It ticks with `timeBeginPeriod(1)` and `Sleep`. Windows 11 ignores that request for
a process that is invisible and inaudible:

| Process | `Sleep(1)` after `timeBeginPeriod(1)`, every second for 12 s |
|---|---|
| Window visible | 2.0 ms |
| Window minimized | 15.5 to 16 ms |
| No window (`CREATE_NO_WINDOW`) | 2 ms for 3 s, then 16 ms |
| Window minimized, sound playing | 2.0 ms |
| No window, `SetProcessInformation(ProcessPowerThrottling, IGNORE_TIMER_RESOLUTION, state 0)` | 2.0 ms |

With the 16 ms timer the filter ticks in steps of 15.625 ms (read gaps of 31.25, 46.9, 62.5 ms), and at some phases two writes fall in one gap: about 14 % of the
frames are lost, at about 26 fps. No writer can prevent it. Control without the app, an exact grid and test patterns, 20 s each:

| Reader | Regime of the read gaps | Runs passed |
|---|---|---|
| `read_numbers`-style reader, no window (as `bar_run.py` starts it) | 15.6 ms steps | 2 of 3 (the third lost 14.2 %) |
| The same with the opt-out above | even 33.33 ms | 3 of 3 |
| Chrome `getUserMedia`, window visible | even 33.3 ms | 4 of 4 at 30.000 and 1 of 1 at 30.009. Two more runs went partly to the 15.6 ms steps (one of them lost frames), with other windows opening in front: *judgement*, an occluded window counts as minimized |
| Chrome, window minimized | 15.6 ms steps | 2 of 2 passed (a lucky phase) |
| Chrome headless, 12 s reads | 15.6 ms steps | 4 of 6 and 1 of 3; the others lost 14 % |

Before 2026-11-07 13:04 the hidden reader was not throttled on this machine (even read gaps in all saved runs); after a restart at that time it was. The cause of
the change is not known. Through the same `bar_run.py` with the hidden reader, on the final prototype, 1072 strong gave 26.14, 30.01 and 30.02 fps.

Therefore: the numbers of sections 2 to 15 come from the then unchanged `bar_run.py`, `read_numbers.py` and `judge.py`, with one addition: `PROTO_READER_VISIBLE=1` and the
`sitecustomize.py` of the prototype make the reader process opt out as in the table, so that it behaves as a visible call app does. **A bar run MUST be judged invalid
unless the reader's gaps are on the even clock** (`regime.py`: at least 0.95 of the gaps within 1 ms of 33.33 ms, or the run is reported as a timer regime, not as a result of the app). Since section 16 `read_numbers.py` opts out by itself and `judge.py` does not pass a run below 0.95.

### 2.3 What the OBS boundary needs

The filter's tick is exact on QPC at the declared interval: 33.3332 to 33.3333 ms fitted over 20 s, with `interval` = 333333 (units of 100 ns) in the queue. A later declared interval is
followed too (30.01 and 30.03 fps measured). A tick shows the newest write that has ended. So a frame is lost when two writes end between two ticks. Three ways of writing, at the same grid, on the real app:

| Write | Time per write | Width of the write's end against the grid | Lost per crossing of a tick (camera 150 ppm fast, 240 s, one crossing, floor is 1) | Reader openings passed (soak, 1072 strong) |
|---|---|---|---|---|
| `send` with BGR (today) | 1.8 ms p50, 2.6 p99 | 1.4 ms | 7, over 2.6 s | not soaked (estimate from the width: 4 % of phases are bad) |
| `send` with NV12, converted before (public API) | 0.10 to 0.13 ms p50, 0.18 to 0.30 p99 | 0.15 to 0.28 ms | 2, over 0.9 s | 110 of 110 (4 s reads); 59 of 60 before the priority of §3.5 (6 s reads) |
| Own store into the queue (`obsq.py`) | under 0.01 ms | 0.02 ms | 1 | 110 of 110; 60 of 60 |

The public NV12 path gives the same bytes in the queue as pyvirtualcam's BGR path: 20 of 20 frames (random, black, white, frames of a clip), compared in the queue itself. Conversion with the formula below takes 5.8 to
8.6 ms p50 to p99 on one thread:

    Y = (66 R + 130 G + 26 B + 0x1000) >> 8
    2x2 mean per channel: (((a + c + 1) >> 1) + ((b + d + 1) >> 1) + 1) >> 1
    U = (112 B - 74 G - 38 R + 0x8000) >> 8, V = (112 R - 94 G - 18 B + 0x8000) >> 8

**Direct writing into the queue is not needed for the bar**: the soak does not tell it from the public path. It is used only as the rescue of §8.

A bad phase cannot be seen from the writer (the filter writes nothing to the queue). It remains a risk per opening of the camera; see §14.

### 2.4 Where to cut

1072, 20 s per run, all with the final sender. p99 per stage in ms; the budget is 33.3:

| Cuts after | natural | polished | strong | Verdict |
|---|---|---|---|---|
| `hair` (2 stages) | 14.3 / 28.2 | 16.6 / 34.3 | 36.5 / 46.5 | strong fails (lost 6); over budget |
| `skin` (2) | 26.2 / 32.5 | 19.9 / 26.4 | 28.6 / 27.8 | passes; natural had a spike to 92.9 |
| `contour` (2) | 21.7 / 17.7 | 22.1 / 19.3 | 30.8 / 20.2 | passes; 2.5 ms of p99 margin on strong |
| `hair`, `brows` (3) | 14.1 / 12.3 / 16.1 | 21.7 / 17.8 / 27.7 | 25.3 / 15.8 / 22.0 | passes |
| `hairshape`, `contour` (3) | 11.4 / 18.8 / 24.9 | 9.6 / 14.9 / 21.7 | 19.6 / 18.4 / 25.3 | passes; chosen |

These runs were a 20 s sample under desktop activity; the 60 s table in §2.1 is the same cut when the machine was quieter. `hairshape,contour` wins because the slowest stage is the
lowest on `strong`, and because reshape, the only effect that strong adds, has a stage with room for it. A fourth stage is not needed.

Per effect, 1072 strong, p50 / p99 (ms): reshape 6.57 / 9.41, hairshape 6.72 / 9.83, hair 5.04 / 7.02, skin 4.48 / 6.51, contour 3.50 / 5.44, brows 2.86 / 4.80, eyeshadow 2.64 / 4.37,
tearbag 1.00 / 1.93, eyeliner 1.59 / 2.88, blush 2.75 / 4.08, lips 3.58 / 5.36, teeth 2.68 / 4.31.

### 2.5 What the boundary costs

| Item | Measured |
|---|---|
| Way in, on the output thread: LUT to float32 in shared memory, masks when new, pickle, signal | 0.78 ms p50, 1.70 p99 |
| Way out of a stage: copy a changed picture or mask, pickle, signal | 0.33 ms p50, 0.84 p99 (1072 strong) |
| Metadata of one frame: landmarks, z, 12 settings sections, references | 8.2 kB; pickle 17 us, unpickle 16 us |
| Wake-up by a kernel semaphore through `ctypes` (chosen) | 16.8 us p50, 40.4 p99, 158 max, no CPU while waiting |
| Others: Win32 event | 15.0 us p50, 75.8 p99 |
| `multiprocessing.Semaphore` / `Event` / `Queue`, pipe | 14.5 / 51.7 / 67.0 / 26.6 us p50; 22.7 / 157 / 237 / 47.2 p99 |
| Spin | 2.0 us p50, one whole logical CPU per waiter: rejected |
| 1 ms sleep poll | 402 us p50, 1541 p99: rejected |
| Copy of 11 MB | 0.38 ms (`cv2.copyTo`) |
| 88 MB segment: create / first touch of every page | 0.08 ms / 17.5 ms: touch at start, never on a frame |
| Shared block, 8 slots and 10 mask sets at 1280x720 | 322 MB |
| Start of 3 stages and the sender until ready (warm disk) | 0.36 s |
| Memory per process (working set, shared block included; private) | main 506 / 894 MB, stages 192 to 205 / 117 to 148 MB (reshape's tables up to 350 MB on 1072), sender 82 / 105 MB. Peaks vary by run |

### 2.6 Room for a call

Stand-in: `load.py`, one ffmpeg libx264 `veryfast` 720p30 encode in real time and two busy processes. No network, no audio, no decoding of others: it is a stand-in, not a call app.

| Run, 60 s | fps | Lost | Worst stage p99 (ms) | Worst stage (ms) | App CPU | x264 |
|---|---|---|---|---|---|---|
| 1072 natural | 30.0 | 0 | 19.1 | 22.0 | 2.90 | 30 fps, 1.01x |
| 1072 polished | 30.0 | 0 | 20.2 | 24.7 | 2.91 | 30 fps, 1.01x |
| 1072 strong | 30.0 | 0 | 20.5 | 23.6 | 3.14 | 30 fps, 1.01x |
| 1060 strong | 30.0 | 0 | 21.9 | 25.7 | 3.15 | 30 fps, 1.01x |
| 1012 strong | 30.0 | 0 | 15.1 | 17.7 | 2.55 | 30 fps, 1.01x |

The app takes up to 3.2 of 16 logical CPUs, which is 0.3 to 0.9 more than today.

### 2.7 Delay

From the frame leaving the file camera, by `delay.py` (QPC at both ends). "To write" does not depend on the reader's phase; "to reader" adds the wait for the tick, 0 to 33 ms and fixed per opening. 30 s per run:

| Face | Preset | To write p50, today | To write p50, now | To reader p50, today (range over runs) | To reader p50, now (one run) |
|---|---|---|---|---|---|
| 1072 | natural | 58.5 | 74.4 | 84.4 | 94.8 |
| 1072 | polished | 59.5 | 74.2 | 83.7 | 81.4 |
| 1072 | strong | 67.5 | 82.4 | 87.0 | 101.9 |
| 1060 | natural | 52.5 | 68.9 | 76.8 | 74.6 |
| 1060 | polished | 55.3 | 71.0 | 67.4 | 99.7 |
| 1060 | strong | 60.9 | 78.6 | 82.5 | 103.0 |
| 1012 | natural | 31.1 | 60.8 | 40 to 59 | 85.5 |
| 1012 | polished | 31.5 | 95.9 (one slip, §6) | 38 to 55 | 101.7 |
| 1012 | strong | 39.1 | 66.2 | 59.8 | 94.5 |

So the design adds 15 to 18 ms to the write on the two slow faces and about 30 ms on the fast face (conversion in place of the send: +5; the grid must hold the worst frame of the settling window rather than the typical one; margin 15), plus 33 ms after a slip. With a camera clock free of the file camera's 15.6 ms steps (`PROTO_CAM_EXACT=1`) it is 2 ms less: 1072 natural 72.4, strong 80.3, 1012 natural 58.2, strong 64.2.

What people notice: ITU-R BT.1359-1 puts the detectability of audio ahead of video at about 45 ms and acceptability at about 90 ms, from subjective tests on television material. A later picture in a call is audio ahead. The added 15 to 30 ms is below that threshold on its own. The total from the file camera to the reader, 75 to 103 ms, is above it, as it is today on the slow faces (67 to 87 ms), and the camera's own delay comes on top. *Judgement*: acceptable for a call, and 30 fps is worth 15 to 30 ms; it must still be judged by eye and in a real call. If it is too much, lower `MARGIN_MS` first (§6), at the price of more slips.

### 2.8 The camera's clock against the filter's

`PROTO_CAM_PPM` runs the file camera on an exact clock, fast or slow. No webcam on this machine: the real error of a webcam is unknown.

| Camera error | Grid | Run | Reader lost | Slips (repeats) | Drops | Note |
|---|---|---|---|---|---|---|
| +1 ppm | slots (§6) | 60 s | 1 | 1 | 0 | one stall of the machine: reader gap 98.6 ms and one slip at the same moment |
| -300 ppm | slots | 120 s | 0 | 1 | 0 | a repeat, as predicted |
| -1000 ppm | slots | 60 s | 0 | 2 | 0 | repeats every 33 s (*calculated* 33.3) |
| +300 ppm | slots | 120 s | 0 | 0 | 0 | delay grows until a drop is due; none due yet |
| +1000 ppm | slots | 180 s x 3 | 4 / 5 / 72 | 0 | 4 / 3 / 3 | first drop at 77 to 95 s, then every 31 to 34 s, as *calculated*. Lost equals drops in one run; 2 more in another; the third fell on a partly bad phase (§14) |
| +150 ppm | grid locked to the camera, NV12 | 240 s | 2 | - | - | around the one crossing of a tick |
| -150 ppm | grid locked to the camera, NV12 | 240 s | 8 | - | - | a slow camera should lose none; the crossing flaps |
| +150 ppm | grid locked to the camera, BGR | 240 s | 7 | - | - | |

A camera faster than the filter cannot pass `lost == 0` with any design: the reader takes one frame per tick, at most 30.00003 per second. The floor is one frame per `1 / (30 * error)` seconds (100 ppm: one in 5.5 minutes). The slots grid reaches that floor and adds nothing to it. A declared interval below 33.33 ms removes the floor only for a reader that takes every tick: Chrome, asked for 30 fps, then lost 1 frame per 15 s at a declared 30.03 fps and no frame in 15 s at 30.009. It is rejected (§11).

### 2.9 Failure drills

Real app, bar's reader attached, failure at frame 480 of 1072 strong, 18 s read. "Biggest gap" is what the reader saw.

| Failure | Biggest gap | Drawing done again | After | Status line | Left over |
|---|---|---|---|---|---|
| Stage 0 dies | 133 ms | 162 ms, 2 frames | 22.4 fps in process | yes | nothing |
| Stage 1 dies | 200 ms | 220 ms, 3 frames | 21.2 fps | yes | nothing |
| Stage 2 dies | 133 ms | 163 ms, 2 frames | 22.6 fps | yes | nothing |
| Sender dies | 333 ms | 387 ms, 3 frames | 20.4 fps | yes | nothing |
| Stage 1 hangs | 1367 ms | 607 ms, 8 frames | 22.6 fps | yes | nothing |
| Sender hangs | 1234 ms | 470 ms, 8 frames | 21.8 fps | yes | nothing |
| Stage 1 slow, +40 ms for 150 frames | 67 ms | none | 30.2 fps, no fallback, 59 frames lost while slow | no | nothing |
| Ctrl+C | - | - | exit 0, main gone in 0.47 s, all in 0.75 s, camera opens again | - | nothing |
| Main process killed | - | - | all gone in 0.24 s, camera opens again | - | nothing |

A hang limit of 0.25 s was tried first and gave one false alarm in about 30 runs: a stall of 0.27 s moved the whole run to one process. So the limit is 1.0 s.

After a sender's death `pyvirtualcam.Camera()` fails while a reader holds the old queue (`virtual camera output could not be started`, seen 3 of 3 tries; the same error from a browser that kept the camera). That is why the rescue of §8 exists. Through it the reader kept getting frames. After a killed hung sender the second try of `Camera()` succeeded in the one run of that drill; the reason for the difference is not known, and §8 covers both.

### 2.10 Bit identity

297 frames of 1072 at each of natural, polished and strong, with a script of preset glide, slider, sections off, bypass, face lost, lost camera frames and a plate; tracker on every frame, segmentation inline: 0 frames differ from the in-process chain, with 3 stages and with 2 stages. Two in-process runs agree with each other.

---

## 3. Processes

### 3.1 Main process

It keeps everything it has today up to `_plan` and `_segment`: camera thread, track-ahead thread, segmentation thread, settings, glide, fade, face state, plate, GUI. Instead of drawing, `_render` converts to float32 in a free slot (`_TO_FLOAT` into shared memory), writes the metadata and signals stage 0. `process()` then returns a result that says "the frame is on its way". The output thread therefore never waits for drawing, and `_FrameSlot` never overwrites a frame unless the output thread itself is held up.

A collector thread (priority above normal) takes finished slots from the last edge in order, builds the preview, calls `on_frame`, frees the slot, and watches the processes (§8).

### 3.2 Stages

| Stage | Effects, in order | Ends with |
|---|---|---|
| 0 | reshape, hairshape | changed picture, masks and landmarks in place (§4) |
| 1 | hair, skin, contour | |
| 2 | brows, eyeshadow, tearbag, eyeliner, blush, lips, teeth, neural | `cv2.max(image, 0)`, `convertScaleAbs(alpha=255)` into the slot's uint8 picture |

The order MUST be the order of `Effect.order`. An effect that is added later joins the stage of the effect it follows. A stage MUST NOT change the order within or across stages.

Each stage process runs `_limit_cv2_threads` and `_use_one_thread_per_core`, as the main process does today. It holds its own effect objects. It does not load the tracker or the segmentation model.

### 3.3 Sender

One process owns `pyvirtualcam.Camera` for the whole session: one hot thread and one control thread that reads commands from stdin (open, close, size). See §6.

### 3.4 Why not fewer

A single worker for all effects leaves its median above the budget on the worst face (§11). A thread cannot be the sender: the send must end within a fraction of a millisecond of its place, and a thread of the main process shares the GIL with the tracker's hand-over, the collector and the GUI. A sender thread with the BGR send varied by 1.3 ms (-0.1 to +1.2) in a 30 s run; a process, 0.25 ms.

### 3.5 Priorities and power

Every process of this design, including the main process, MUST call `SetProcessInformation(ProcessPowerThrottling)` with `EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION` in the control mask and 0 in the state mask. Without it, processes without a window took about 20 % more CPU for the same work on the same machine (total 4.0 logical CPUs against 3.3, stage p99 19.5 / 18.3 / 25.2 against 18.4 / 16.4 / 22.8 on 1072 strong). The stages and the sender MUST run at `ABOVE_NORMAL_PRIORITY_CLASS`: with it and `MARGIN_MS` 15 there were 1 slip in 3 minutes on 1072 strong, against 4 in 3 minutes at normal priority and margin 6. The sender's hot thread MUST run at `THREAD_PRIORITY_TIME_CRITICAL`; it runs for under 10 ms of every 33 and sleeps until 2 ms before its place.

Stages call `gc.collect(); gc.freeze()` after start; the sender also `gc.disable()`, as it makes no cycles (a gen 2 collection took 10 to 12 ms in the investigation).

### 3.6 Modules

New package `makeup_cam/parallel/`: `shm.py` (layout, rings), `engine.py` (main process side), `stage_main.py`, `sender_main.py`, `nv12.py`, `obs_queue.py` (§8 only). Started as `sys.executable -m makeup_cam.parallel.stage_main`, with the arguments on the command line as JSON. A stage MUST NOT import anything that starts a model; the sender MUST NOT import the effects.

---

## 4. The boundary

### 4.1 Shared block

One `multiprocessing.shared_memory` block per frame size, made by the main process, touched completely before the first frame, unlinked at close.

    control      64 kB: rings, flags (stop, heartbeat per process, counters)
    slots x 8    metadata (128 kB) | float32 picture | uint8 picture | 3 masks that a stage changed
    mask sets x 10    hair, face skin, body skin
    plates x 2   float32

A slot belongs to exactly one process at a time; ownership passes with the ring. No locks.

### 4.2 Rings and signals

One ring per edge (main to 0, 0 to 1, 1 to 2, 2 to sender, sender to main): one writer, one reader, slot numbers. A named kernel semaphore per edge wakes the reader; it waits with a 100 ms timeout, so that it can check the stop flag and write its heartbeat. Names MUST contain the main process's id. Nothing spins, except the sender's last 2 ms before a place.

### 4.3 What crosses, for each frame

| Field | How | Note |
|---|---|---|
| Picture | in the slot, float32, in place | stage 2 writes uint8 |
| `landmarks`, `landmarks_z` | pickled in the metadata, read-only on arrival | a stage that changes `ctx.landmarks` puts the new array in the metadata |
| `hair_mask`, `face_skin_mask`, `body_skin_mask` | reference to a mask set, or to the slot's own masks once a stage has changed them | see §4.4 |
| `background` | generation number of the plate | see §4.5 |
| `timestamp_ms`, `masks_timestamp_ms`, `skipped_frames` | metadata | |
| Steps: effect name, settings section, intensity, `needs_hair_mask` | metadata, per frame | this is how a glide or a slider reaches the stages; no other channel for settings exists |
| Reset | flag on the next frame | every stage resets its effects when that frame arrives, before drawing it, in order with the frames |
| `seq`, camera time, mirror, "pass through" | metadata | |

Effects that write to the context: reshape (`landmarks`, three masks) and hairshape (`hair_mask`). `ctx.cache` is not used by any effect today; if one starts to use it across a cut, the test of §12.1 fails, and it MUST then cross explicitly. `ctx.intensity` is set per step in each stage.

The masks a stage must pass on come from a table of what each effect reads:

    reshape, hairshape: hair, face skin, body skin
    hair, skin, contour: hair, face skin
    blush: face skin

It MUST be checked by a test against the code (§12.1), as a mask missing there changes the picture silently.

### 4.4 Identity of arrays

The hair effect recognizes a mask it has searched by its identity, and reshape recognizes the plate by its identity. So a stage MUST keep one array object per mask set and generation, and MUST make a new array object for a mask set that was filled again. A mask that a stage changed arrives as a new view for each frame, as reshape's masks do in-process. A set is used again only when no frame in flight refers to it (reference count in the main process).

### 4.5 Plate

Two plate slots. A new plate goes into the one not in use and gets a new generation; frames carry the generation; a stage makes a new array object when the generation changes. No plate: generation `None`. A slot is reused only when no frame in flight refers to its generation. The prototype had one slot and copied in place, which is a race on a second capture; the build MUST NOT do that.

---

## 5. The output is bit-identical

Rule: for the same frames, the same landmarks, the same masks, the same settings and the same order of resets, the bytes of `ProcessResult.output` of the chain in the stages MUST equal those of the in-process chain, on every frame, and the bytes in the OBS queue MUST equal those of pyvirtualcam's BGR path.

The segmentation thread's timing varies, so masks differ between two real runs. The test takes that out, not the comparison: it uses a mask source that gives the same masks for the same frame number (`Segmenter(background=False)` on clips, a fixed function in pytest) and the tracker on every frame. Nothing is rounded: hashes of the bytes are compared.

Exceptions, all outside the rule: frames after a fallback (§8; the effects restart), and the first frames after a handover from in-process drawing (§9), which includes the return to the stages after a fallback.

---

## 6. The output clock

The frame's place on the output is what decides whether it is lost.

    P = interval * 100 ns = 33.3333 ms, the period declared in the queue; never 1/30 s. Over an hour the difference is 3.6 ms.
    BURST_S = 0.6       frames go out as they come: the first frames of new stages are slow and arrive in a heap
    SETTLE_S = 1.5      from the first frame to the grid, when nothing disturbs the window from BURST_S on (was 6; section 16)
    SETTLE_LATEST_S = 6 the grid starts by then, however disturbed the window
    DISTURBED_MS = 20   a frame this far from the median of the window starts the window over, so that a stall stays out of the delay
    GUARD_MS = 9        time reserved for the conversion (p99 8.6 on the real app)
    MARGIN_MS = 15      spare, for spikes of the stages
    SPARE_FRAMES = 3    frames of delay allowed above the settled one
    WINDOW = 30 frames

In the window, before the grid, each frame goes at `max(the last place + P, now + GUARD)`, so that no two writes fall within one period, and a frame that waited for the sender more than `P` goes as it comes and starts the window over, as holding a heap at one per period would leave it in the delay. At the end of the window, from its frames:

    camera_phase = median(t_camera - seq * 1/30)
    d = max(t_finished - (seq * 1/30 + camera_phase)) + GUARD + MARGIN
    grid[0] = max(camera time of this frame + d, now + GUARD, the last place + P)

New stages (§7, §8) and a camera opened again start the settling once more. The grid keeps its phase: the first place is the first `grid[0] + k * P` at or after that time, so the new delay can be smaller than the old one.

Then for each camera frame, in order:

    k = k + 1; due = grid[0] + k * P
    if now > due - GUARD:              # slip: the reader sees the last frame once more; nothing is lost
        k = ceil((now + GUARD - grid[0]) / P)
    elif over the last WINDOW frames min(due - t_finished) > GUARD + MARGIN + SPARE_FRAMES * P:
        drop this frame; k = k - 1     # the camera outruns the filter, or a long stall; bounds the delay
    convert to NV12 in a private buffer
    sleep until 2 ms before due, then spin; send

Properties that a build MUST keep:

- The phase of the grid against the filter's ticks never moves. A slip moves by whole P. That is what makes a clock error cost one repeat or one drop per crossing and nothing more; a grid that follows the camera flaps at each crossing (§2.8).
- At most one write per place. Never two in one P.
- A late frame is never sent "as soon as possible" between places.
- Placeholders (no camera) are written at once, off the grid, and `k` catches up when the camera returns through the slip rule.
- Slips and drops are counted and logged with their time. A drop is a lost frame and MUST count as one in the status statistics.

Delay after slips does not shrink by itself: the only way is to lose a frame, and `lost == 0` comes first. It is bounded by `SPARE_FRAMES`. The status line shows it (`遅延`), from the camera to the end of the write, so that it does not stay unseen.

`pyvirtualcam.Camera(width, height, fps=30, fmt=PixelFormat.NV12)` declares 333333. If the output size differs from the frame size the sender MUST run `fit_frame` before the conversion. If the camera cannot open with NV12 it MUST open with BGR (same bytes, wider write; §2.3), log it once, and continue.

The conversion MUST be the formula of §2.3 and MUST give the bytes of pyvirtualcam's BGR path; a test compares them (§12.3). A pyvirtualcam update that changes its own conversion fails that test, and only then is this formula changed.

---

## 7. Behaviour of the app

| Situation | Rule |
|---|---|
| Slider (`set_settings`), preset (`glide_to_settings`) | No new path. Each frame's plan carries its settings. Frames in flight keep theirs. A slower preset may cause one slip. |
| Bypass (メイクOFF), no effect planned, closed pipeline | The frame takes its turn behind the frames in flight as "pass through": copied into the slot's uint8 picture, no stage draws, a reset flag still travels. Delay does not change, so the order is kept and no frame jumps ahead. |
| Face lost, fade | As today in the main process; the plan and the landmarks say what the stages do. Reset goes with the frame. |
| Lost camera frames | `skipped_frames` as today, in the metadata. |
| Camera lost | Placeholder through the sender (§6), as today at the target rate. |
| Frame size changes | `_on_frame_shape` as today. In-process drawing takes over at once, a new block and new stages start in the background, handover as in §9. The old stages stop after their last frame. The clock settles again (§6). |
| Output size changes, virtual camera on or off | Commands to the sender on the output thread. It reopens the camera. With the virtual camera off, stages run, the sender only returns slots. |
| Background plate | §4.5. `capture_background`, `clear_background` as today on their threads; `_plate` is read once per frame by the main process. Raw frames for the capture come from the output thread as today. |
| Neural stage (order 90) | Runs in stage 2, in its own process tree (children stay in the job, §9). `start_ahead` runs when the frame arrives in stage 2, since all geometry is final by then. Its status message comes to the main process through the control channel. **Not measured with this design**; a test and a measurement are required (§12.4, §13). |
| Preview (`on_frame`) | Built by the collector thread when the slot comes back: `_compose` on the finished picture, with the frame's own landmarks and the raw frame kept in the entry. On the collector thread, not the output thread; GUIs already hand it over with a queued signal. Measured with a flip per frame: main 1.11 logical CPUs, pass. The Qt painting itself was not in the measurement. |
| Status: fps, 処理 ms | fps comes from the sender's writes, from the shared counters. 処理 ms is the median from camera to finished frame, so that it means what it says; today's number would drop to 7 ms and mislead. |
| Photo analysis (`analyze.py`), evaluation (`evaluate.py`), recorder (`record.py`) | Never use the stages. `MakeupPipeline(...)` stays in-process unless its owner passes the engine; only `gui.py` and `run_headless` in `__main__.py` do. The recorder does not use the pipeline. `analyze.py` renders with its own effect objects. |
| Exception in an effect | In the stage, logged once per effect and message, the effect skipped for that frame, as `_run_effect` does. Logged through stderr to the main process's log. |

---

## 8. When something goes wrong

The camera MUST never freeze for good. The status line MUST say what happened, in Japanese.

Detection, on the collector thread, every 25 ms or every finished frame:

- A process ended: at once.
- No frame came out for `HANG_LIMIT_S = 1.0` while frames are in flight (2.0 for the first 30 frames, whose tables are built lazily).

Slow is not hung. A stage that is slow for a while costs frames while it lasts (§2.9), and the output thread never waits for a free slot longer than 100 ms before it drops the frame and counts it. Slow frames do not start a fallback: in one process it would be slower still. The one exception is below.

**Too few cores.** On at most `FEW_CORES = 5` cores that the process may use, which takes the switch forced on (§10), the app's 3 to 3.5 logical CPUs starve a reader at normal priority. Once per second the collector counts the frames lost, repeated or dropped:

- In 7 of the last 10 seconds some: all processes go to normal priority, and the sender's hot thread to `THREAD_PRIORITY_HIGHEST`. `SPARE_FRAMES` becomes 1, as frames are being lost anyway. After 30 s without any, the priority returns.
- In each of the last 10 seconds at least a third of the frames: a fallback with `CPU が足りないため、1 つのプロセスで処理を続けています（フレームが落ちることがあります）。ほかのアプリを閉じてアプリを再起動すると元に戻ります。`, and no try again.

With more cores nothing changes: what the frames lack goes to other programs, and giving way to those only costs frames (§16).

Fallback:

1. Stop accepting frames. Kill the stages, and the sender if it is dead or its heartbeat has stopped.
2. Reset the effects of the main process. Draw the frames in flight again in order, with what they were planned with (the entry keeps it), then every later frame, in process.
3. Send through the sender if it lives (it writes at once, off the grid). If not, open `pyvirtualcam.Camera` in the main process, 3 tries 50 ms apart. If that fails because a reader holds the old queue: rescue (below).
4. `RunnerStatus.processing_error` = `並列処理が止まったため、1 つのプロセスで処理を続けています（フレームが落ちることがあります）。`.
5. Try again `RECOVER_AFTER_S = (2, 20, 60)` seconds after the failure and after each try that fails. A try starts new stages, and a new sender if the old one is gone, in the background; frames are drawn in process meanwhile, and the first frame in the new stages carries a reset, as in §9. It succeeds silently: the status line clears. After three failed tries it says `並列処理を再開できませんでした。1 つのプロセスで処理を続けています（フレームが落ちることがあります）。アプリを再起動すると元に戻ります。`. After 60 s of running, a failure counts as the first again. (This section said "no automatic restart" before: a window has only one `start()` in its life, so "the next start" meant restarting the app, and a stall of one second cost the bar for the rest of the call; §16.)

A new sender can open the camera only when no reader holds the dead sender's queue (§2.9). If one does, it writes into that queue by the rescue below, with the same checks, on its grid as ever. If that fails too, the rescue in the main process stays, and the next try sees whether the reader has let go.

The bar does not hold during a fallback: in-process drawing gave 20 to 23 fps on 1072 strong.

### Rescue: the only direct write into the OBS queue

Only when `Camera()` fails after the sender's death: at first from the main process, then from the new sender. `obs_queue.py` opens the existing mapping `OBSVirtualCamVideo` and writes buffer `(write_idx + 1) % 3` (timestamp at +0, the frame at +32, on the offsets of the header), then stores `write_idx`, `read_idx` and `state` in that order.

Why it is a risk: nothing public promises this layout, and an OBS or pyvirtualcam update can change it. How it is detected, and what follows:

| Check | When | On failure |
|---|---|---|
| Header: frame type NV12, `cx`, `cy` as ours, three offsets equally spaced and at least frame size + 32 apart, `interval` 333333, `state` ready | at every rescue, before the first write | no rescue |
| After the first write: `write_idx` and `read_idx` both moved to ours | at every rescue | no rescue |
| `test_obs_queue_layout`: the bytes and the header after `Queue.commit` equal those after `pyvirtualcam.Camera.send` of the same frame, on the installed OBS and pyvirtualcam | every test run on a machine with OBS; MUST be run after any update of either | the test fails; pin the version or fix the module |
| Pinned versions: `pyvirtualcam==0.15.0` in the requirements | always | |

No rescue: `Camera()` is tried again every second, frames are drawn for the preview, and the status line says `仮想カメラを開き直せません。通話アプリのカメラを一度オフにして、オンに戻してください。`.

A silent break that passes those checks can only show as a wrong picture or no new frames at a reader, and only after the sender has died. That is accepted: it is the second of two rare events, and the normal path uses the public API alone.

---

## 9. Start and stop

**Start.** `CameraRunner.start()` starts the engine on a thread of its own, in the background. The sender starts first and is the only owner of the camera. Until the stages report ready the output thread draws in process and hands its frames to the sender as frames to write at once; until the sender is ready the virtual camera is simply not open yet, as during today's retries. The first frame in the stages carries a reset. Spawning MUST NOT block the output thread: 0.36 s warm, and more on a cold disk. Failure to start any process is a fallback at once, with the status line.

**Orphans.** Every child goes into a Windows job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Each child also watches the main process's handle on its own thread (`WaitForSingleObject`) and exits when it ends. MUST NOT use a blocking read of stdin for that in a thread of a stage: it blocked later DLL loads of the process. Children ignore Ctrl+C: `SIGINT` is set to ignore, and the main process stops them in order.

**Stop.** `MakeupPipeline.close` and the engine: set the stop flag, post all semaphores, wait up to 3 s, kill what remains, then unlink the block. The sender closes the camera last, so that readers see the camera stop. Ctrl+C: as today, by `KeyboardInterrupt` in `run_headless`, then the same path; 0.75 s until nothing is left. Closing the window: `closeEvent` and `stop_runner` as today, then the same path; it MUST hold within `RUNNER_STOP_DEADLINE_S`. The console close watch (`ConsoleCloseWatch`) goes the same way. A killed main process: the job object ends everything, 0.24 s. **Closing the GUI has not been run on the prototype**; the test is required (§12.4).

---

## 10. Switch and default

`AI_MAKEUP_CAM_PARALLEL`: `1` on, `0` off. Default on, since the design passes. `0` is exactly today's path: in-process drawing, pyvirtualcam with BGR, no extra process. Also `--no-parallel` for the command line. Below 6 physical cores that the process may use (its affinity counts) the default is off and the status line says nothing; this limit is *judgement*, as only this machine was measured.

Settings for measurements only, not documented for users: the constants of §6 as module constants.

---

## 11. Alternatives, and why not

| Alternative | Measured | Decision |
|---|---|---|
| One worker process for all effects, the main process captures and tracks | 1072: 26.5 / 27.0 / 20.8 fps (natural, polished, strong); worker p50 35.0 / 36.1 / 42.1, p99 39.9 / 41.0 / 47.6. 1060 natural passes (31.7, p99 35.1) | Rejected. The whole chain is over 33.3 ms on the worst face |
| Two stages | §2.4. `contour` passes, with p99 30.8 on strong | Rejected for lack of margin; three cost 0.1 to 0.2 logical CPU more |
| Frame-level parallel workers | Not run. Hair, hairshape, reshape, contour and teeth carry state from frame to frame (statistics, strand memory, poses, plate trust), and frame n needs it from n - 1. A worker for frame n would wait for that state effect by effect, which is this pipeline with more memory and copies. Without that wait, pixels change | Rejected: impossible with identical output |
| Effects with separate regions in parallel processes | *Calculated* from §2.4: only the steps after skin and contour could run side by side, and not all (blush lies on contour, eyeliner on eyeshadow, teeth next to lips). At best {brows, eyeshadow, tearbag, eyeliner}, {blush}, {lips, teeth}: 43.4 ms becomes 34.4 ms on 1072 strong at p50, before any boundary. Earlier result with threads: 36.83 to 33.92 ms | Rejected: still over budget |
| Speeding up hairshape and hair | Not attempted. *Calculated*: with both at zero, 1072 strong has p50 33 ms and p99 41 ms in process | Rejected as the answer; still a good thing later for delay and CPU |
| Hairshape analysis less often | Every 2nd / 3rd frame, 1072: hairshape p50 8.3 to 7.4 / 5.8 ms (natural), whole effects 35.0 to 32.9 / 31.3 ms; strong 43.2 to 40.5 / 38.3 ms. Every frame differs from the reference: the worst pixel 34 / 43 of 255 (natural), 46 / 56 (strong), up to 2.4 % of the pixels by more than 1 level | Rejected: changes the look and does not reach the budget |
| Short queue in front of the drawing | From the investigation: 1060 natural slot loss 4.50 to 2.63 %, others noise; each step adds about 33 ms when frames pile up | Rejected: it holds frames, it does not draw faster |
| Effects on GPU shaders | From an earlier study: round trips eat the gain | Rejected |
| Threads | From an earlier study: GIL, 36.83 to 33.92 ms | Rejected |
| Sender in the main process | §3.4 | Rejected |
| Grid locked to the camera | §2.8 | Rejected |
| Direct writes as the normal path | §2.3: no difference in the soak; a private layout | Rejected, rescue only |
| Declaring 30.009 or 30.03 fps so that a fast camera has room | The filter follows it. A reader that takes every tick loses nothing. Chrome limited to 30 fps then discards real frames (1 in 15 s at 30.03) | Rejected |
| Declaring 60 fps, each frame shown on two ticks | Bar's reader: no phase and no timer regime can make it miss a frame (0 lost, even with the hidden reader). Chrome limits to 30 fps by taking every second tick: headless 4 of 6 passed, the same as 30 fps, so no gain there. Native call apps not tested, and they would see a 60 fps device | Rejected: it would make the bar look better than a browser call is, for an unknown risk in native apps. Worth a test with Zoom and Teams if bad phases show up in practice |

---

## 12. Tests required

pytest. Those that need OBS skip without it and say so. Those that need the clips run from `.test-media` and skip without them; each has a synthetic counterpart with a fake tracker and fixed masks, so that the rule is always tested. No test MAY be weakened to pass; say so if one changes.

### 12.1 Bit identity

- Stages against in-process on synthetic frames with landmarks and masks, per preset: bytes equal on every frame, for the cut of §3.2.
- The same with a script: preset glide, slider during a glide, sections off and on, intensity 0, bypass on and off, face lost and found, lost frames, plate set, replaced and cleared, mirror on and off.
- Clips, 300 frames per preset on 1072 (`identity.py` of the prototype): bytes equal.
- Reads table: running each effect with a context that records attribute access gives exactly the masks of §4.3, for every registered effect.
- Writes: a test fails if an effect writes a context field other than those listed in §4.3, or uses `ctx.cache` across a cut.
- Array identity: `hair`'s search runs once per mask set, in a stage as in process. A filled-again set is seen as new. A new plate is seen as new.
- Resets arrive in order: a reset on frame n reaches a stage before frame n and after frame n - 1.

### 12.2 Boundary

- Rings: order, wrap-around, one writer and one reader across processes, for over 2^16 frames.
- Slot ownership: no slot is written by two processes; all return after a run (count equal to 8 at the end).
- Mask set and plate reference counts return to the base; no reuse while a frame in flight refers to one.
- Metadata larger than the space: fails loudly rather than overwrite.
- Block layout for other frame sizes, including odd ones.

### 12.3 Output clock (with a fake camera and a fake clock where possible)

- Locked camera: `k` increases by 1 per frame; no slip, no drop.
- Camera -1000 ppm: one slip per 33 s, no drop, phase unchanged (grid values are `grid[0] + k P` exactly).
- Camera +1000 ppm: drops at the expected rate, delay bounded by `SPARE_FRAMES`.
- Spike: one late frame gives one slip and no drop; a second gives one more; delay within the bound.
- Long stall: delay returns within the bound through drops, one per `WINDOW`.
- Placeholder and back; bypass; preset to slower preset.
- `P` is the declared interval (333333 x 100 ns), not 1/30.
- `to_nv12` equals a reference in Python, on random frames, black, white, odd values.
- With OBS: `test_nv12_same_as_pyvirtualcam`: bytes in the queue equal between the BGR path and the NV12 path.
- With OBS: `test_obs_queue_layout` of §8.

### 12.4 Life cycle and failure

- Each process killed at a random frame: fallback within 250 ms of detection, frames in flight drawn in order, later frames go on, status line in Japanese is the exact string.
- Each process hung: the same after `HANG_LIMIT_S`. A stage slower than the budget for 5 s: no fallback.
- Sender killed while a reader holds the queue (with OBS): rescue works; with a fake queue with a wrong header: no rescue, the other string, retries.
- `Camera()` refused at start; NV12 refused; a failed spawn; a stage that never reports ready.
- Ctrl+C in `run_headless`, stop and start 10 times in a row, closing the GUI (offscreen), console close, `set_camera` with another size, virtual camera off and on: no process left, no semaphore or block left, camera opens again.
- Main process killed: no process left within 2 s.
- `AI_MAKEUP_CAM_PARALLEL=0` and `--no-parallel`: no child process, and output bytes as today.
- Analysis, evaluation and the recorder start no process.
- Neural on (`.venv-neural`, with a stand-in model): runs in stage 2, its status reaches the status line, its worker dies with the job. In-process and in-stage outputs equal when the same deltas are given.
- Failure and try again: a killed stage, a killed sender, a stage that stood still for over the limit: the stages draw again, the status line clears, frames stay in order. Tries used up: the other string, frames go on. Too few cores: one stall changes nothing, trouble second after second lowers every process's priority and it returns, a third of the frames for 10 s ends in one process for good; with more cores nothing gives way.
- A start right after a stop, with a spawn still under way, is not disturbed. A child that cannot join the job does not stay. A hung sender during a change of frame size still ends the old stages.
- What the sender writes, in OBS's queue: the output mirrored and fitted, compared with the output of the collector.
- The full existing suite passes, with the switch on and with the switch off.

---

## 13. Numbers a build must reach

All with the bar's tool, reader on the even clock (§2.2), final settings, on this machine or one like it. Nothing rounded up.

| What | Must |
|---|---|
| 3 faces x 3 presets, 60 s each | `pass` in `judge.py` on all nine, `lost == 0`, in one series. A run that fails MUST be reported, not repeated away. A repeat is allowed only with evidence of a stall or a bad phase, and the count of such runs is reported (see §14) |
| The same with `load.py` on 1072 x 3 presets, 1060 strong, 1012 strong | pass; x264 at 30 fps |
| Worst stage p99 | at most 25 ms |
| Way in (`submit`) p99 | at most 3 ms |
| Worst CPU, whole app | at most 3.6 logical CPUs on 1072 strong |
| To write (delay tool), p50, no slip | at most 90 ms on 1072 strong, 72 ms on 1012 natural |
| Sender: end of write against its place, p0.1 to p99.9 | within 0.4 ms |
| Reader openings, 110 at 4 s, 1072 strong | 110 passed |
| 28 minutes on 1072 strong | lost 0, at most 3 slips, drops 0 |
| Identity | 0 frames differ, §12.1 |
| Drills | as §2.9: gap under 500 ms for a death, under 1.5 s for a hang; nothing left over |
| Start until ready, warm | under 1 s, and the output thread not blocked |

---

## 14. Not proved, and limits

- **No real webcam, no real call app was used.** The clips are 30.00 fps with a 15.6 ms hand-over jitter; an exact camera clock was simulated. Zoom and Teams were not read. Chrome `getUserMedia` was (§2.2), without an encoder and without a real call. Checks with a real webcam and real apps remain to be done.
- **A clock error of a real webcam is unknown.** A camera faster than 30.00003 fps loses one frame per `1 / (30 * error)` seconds, by physics (§2.8). Slower: repeats only. `lost == 0` then cannot hold on a fast camera for any design; the build reports drops so that the rate can be read.
- **Bad phase.** At 1 opening in several hundred at most by the width of the write (*calculated*: 0.25 ms of 33.3 ms, under 1 %), the grid can fall on the filter's tick, and frames are lost for as long as that reader stays open, up to about 13 %. Observed: 0 of 110, 1 of 60 before the priority of §3.5, and 2 runs in about 45 bar runs across this work (one with 50 lost in 20 s, one with 68 lost in 180 s). The writer cannot see it. Opening the camera again in the call app gives a new phase. It has no cure in this design.
- **A throttled reader** (§2.2): a call app that is minimized and silent, or whose capture process Windows treats as invisible, gets a 16 ms filter timer. Then about half of the phases lose about 14 %, with any writer, including today's. A minimized Chrome window went to that timer (2 of 2 runs happened to pass). Real calls have sound, in another process; that was not tested.
- **Stalls of the whole machine.** A stall of 50 to 300 ms costs frames in the reader itself or in the slots, and adds a slip. One such run in 60 s (+1 ppm row of §2.8).
- After slips the delay stays higher until a drop or a restart (§6).
- The neural stage with this design is not measured. Qt painting is not in the numbers. Closing the GUI has not been run on the prototype. A frame size change is specified but not prototyped.
- Sources for §2.7: ITU-R BT.1359-1 (1998); the judgement is mine.

---

## 15. As built: where the build differs from the text above

Each of these came up while building, and the text above stays as it was measured. The build follows this section where they differ.

- **The table of §4.3 was incomplete.** `neural` reads the hair and face-skin masks too (`MASK_READS`). `skin` reads the hair mask only when there is no face-skin mask. Both are checked against the code by `test_parallel_identity`.
- **§6, other camera rates.** `camera_phase` and `d` assume 1/30 s per camera frame, which at a 15 fps camera would count half a settling window of drift as delay. If the period measured in the settling window is more than 1 % from 1/30 s, the measured one is used. At 30 fps nothing changes.
- **§6, the fitted time.** `grid[0]` uses the fitted camera time of the frame (`camera_phase + seq / 30 + d`), as the prototype did, and not its raw time, which has the file camera's 15.6 ms steps.
- **§3.3 and §7, frames to write at once** do not go through a slot. Placeholders have the virtual camera's size, which may differ from the frame size, and exist before any block does. So there is a second, small block for the sender's whole life (`SessionBlock`: flags and 3 frames). The sender's heartbeat and counters live there too, as it outlives the blocks of §4.1.
- **§4.2, the edge to the sender** uses one semaphore shared with those frames and with the commands, so that its one hot thread waits on one object.
- **§8, frames the sender has written.** A failure asks a living sender to forget the line and reads the id of the last frame it dealt with. Only later frames are drawn again; earlier ones go to the preview only. So no frame goes out twice.
- **§8, a stall of everything.** If the collector itself was not scheduled for 0.5 s, the hang limit starts again: a machine that stood still (sleep, for one) is not a hung stage.
- **§8, slow.** §2.9 has no status line for a slow stage. It now says `処理が遅れているため、一部のフレームが落ちています。` for 3 s after a frame is lost.
- **§3.6.** One more module, `clock.py`: the output clock as pure arithmetic, for the tests of §12.3. The children are started on the real interpreter, as `multiprocessing` does: in a venv `sys.executable` is a launcher, and a kill would hit only that.
- **§10.** Off outside Windows: kernel semaphores and job objects, and nothing was measured there.
- **§3.5, the main process's priority.** The prototype's main process also ran at `ABOVE_NORMAL_PRIORITY_CLASS`, so all numbers above were measured that way, though §3.5 asks it only of the stages and the sender. So the main process MUST do the same between `start` and `stop`. Measured next to a game that took 2 to 3 logical CPUs, 1072 natural, polished and strong in turns, 40 s reads, the same settings otherwise: at normal priority 7 of 12 runs passed, 36 frames lost, biggest gap 347.9 ms; above normal 9 of 12 passed, 10 lost, biggest gap 68.9 ms. It is a small sample, and that load is not a call.
- **§2.2, the bar's reader** in the build's runs: `.test-media/qa/fps/impl/sitecustomize.py` makes the same opt-out as the prototype's, under `BAR_READER_VISIBLE=1`.

### 15.1 What the build measured against §13

Same machine and tool, reader with the opt-out, except the row that says otherwise. `Quiet`: no other process above 0.3 logical CPUs and all of them under 1.0 for 15 s before the run. Runs next to a game that took 2 to 5 logical CPUs are marked; they are not results of the app alone, and are reported because a busy desktop is what they show.

| What | Must | Measured |
|---|---|---|
| 1072 x 3 presets, 60 s, final code, quiet | `pass` | 3 of 3: 30.0 fps, worst 1 s 30, lost 0, biggest gap 34.7 ms |
| The same, next to the game | | 0 of 3: lost 17, 16, 18; worst 1 s 8, 10, 10; biggest gap 768.8 ms. Stalls of 100 to 170 ms in all processes |
| 1060, 1012 x 3 presets, final code, quiet | `pass` | 6 of 6: lost 0; two runs with one slip each, worst 1 s 29 |
| The same 9 runs before the main process's priority changed | `pass` | 9 of 9 in one series: 30.0 fps, worst 1 s 30, no slip |
| With `load.py`, 5 runs | pass, x264 at 30 fps | Final code: 4 of 5; 1012 strong lost 1 frame with even writes and no slip, so the reader's tick moved over a write. Before the priority change: 5 of 5. x264 30 fps, 1.01x |
| Worst stage p99 | at most 25 ms | Quiet 24.0 (1072 natural) to 24.3 (1060 polished); with the load 29.2 |
| Way in p99 | at most 3 ms | 1.2 to 1.5; next to the game 33.0 |
| Worst CPU, whole app | at most 3.6 on 1072 strong | 3.19 quiet, 3.68 with the load, 3.71 next to the game |
| To write p50, no slip | at most 90 ms on 1072 strong, 72 on 1012 natural | 85.8 and 89.8 (60.9 in the series before), so **the second is missed in one run**: the settling holds the worst frame of its window, and one spike there stays in the delay for the whole run |
| Sender: end of write against its place | within 0.4 ms | 0.14 to 0.28 in the 9 final quiet runs; **0.51 and 0.44 in two runs of the series before** |
| Reader openings, 110 at 4 s | 110 | **109**: one opening lost 2 frames, with no slip or drop at that time. A first series gave 87, with a game starting in the middle of it and another program before that |
| 28 minutes on 1072 strong | lost 0, at most 3 slips, no drops | Quiet: `pass`, 50251 frames, lost 0, 1 slip, 0 drops, worst 1 s 29. A first run had 0 lost for 20 minutes and then lost 373 from the moment a game started |
| Identity | 0 frames differ | 0 of 294 at each of 3 presets on a real clip with the script and a plate; 0 of 300 at each without; the synthetic scripts |
| Drills | gap under 500 ms for a death, under 1.5 s for a hang; nothing left | Worst 233 ms and 1166.8 ms; nothing left in any of 12 |
| Learned finish on, stage 2 | a measurement | `pass` on 1072 strong with a zero-delta model, status 動作中. A floor: a real model costs more |
| Hidden reader, as `bar_run.py` leaves it, no opt-out | invalid by §2.2 | 1 of 3 passed; the others lost 12.6 % at 26 fps, on the 15.6 ms steps |

---

## 16. Second round: after the review

An independent review and an independent run of the bar found the faults below. Each change was measured on the same machine, with the reader of §2.2. `Quiet`: as in §15.1. The stand-ins of §2.6 (`load.py`) and a second reader on the same camera were used where marked. A row marked "harness" comes from a script around the real app objects with a reader, not from `bar_run.py`: it is a guide, not a verdict.

### 16.1 The first seconds (was: frames as they come for 6 s)

The time from a camera frame to a finished frame, minus its place on the camera's clock, by time since the first frame of the stages (1072 strong, quiet, from the logs of a bar run; 1012 natural and 1072 strong next to `load.py` the same):

| Since the first frame | 0 to 0.5 s | 0.5 to 1 s | 1 to 6 s | after 12 s |
|---|---|---|---|---|
| p50 / max (ms) | 101 / 241 | 54 / 57 | 52 to 54 / 61 | 52 / 68 |

So the load has settled half a second after the first frame, and 6 s of frames as they come only cost frames: a reader opened 1.5 s after the start lost 22 (harness). Now `BURST_S` 0.6 s as they come, then paced (never two writes in one period), the grid at 1.5 s, and a window that a stall disturbs starts over (§6). Harness, 1072, reader opened 1.0 or 1.5 s after the start, 20 s: lost 0 in 6 of 6 over two series (was 22 in 1 of 1). A stall of 0.3 s of stage 1 in the window: lost 0, settled 77 to 80 ms, to reader 85.8 and 86.7 ms (was 166.8 ms and 5 drops). Two changes of the frame size (§7): the clock settles again for each new line; lost 11 around the changes (in-process drawing meanwhile), 0 slips, 0 drops, to reader 95.7 ms (was lost 13, 49 slips, 6 drops, 161.9 ms). A reader opened before the end of `BURST_S` can still lose frames there, and the 0.5 s of in-process drawing before the stages are ready costs frames as before.

### 16.2 After a failure (was: one process until the app restarts)

A stage that stood still for 1.2 s ended the stages for the rest of the call (harness: 24.47 fps, lost 249 in 45 s). Now §8 step 5. Harness, the same: lost 67 and 52 in two runs of 45 s, all within 6 s of the stall, then 30 fps; the new grid keeps the old phase, so the delay after it can be up to one period more (to reader p50 88 and 129 ms). Drills of §2.9 on the final code, reader attached (`impl/drill.py`, 22 s reads, failure at 10 s): every stage killed or hung, and the sender killed, back to 30.0 fps in the last 5 s; worst gap 266.5 ms for a death, 1133.3 ms for a hang; nothing left after any. A hung sender while a reader holds its queue: the first run of that drill ended at 20.7 fps, because the new sender tried the camera only once, just after the main process had closed its own; it now tries 3 times 50 ms apart, as the main process does, and then the old queue (§8). That last change is only on this path; the drill after it is in 16.7.

### 16.3 Too few cores (was: nothing gives way)

All processes kept to the same 2, 3 or 4 cores as the reader, 1072 strong, 40 s reads (harness):

| Cores | Neither step | Normal priority only | Both steps (final) | `--no-parallel` (review) |
|---|---|---|---|---|
| 2 | 7.06 fps, lost 736, to reader 485 ms | 17.71 fps, 317 ms | 17.14 and 17.45 fps, lost 476 and 465, 106 ms (one process after 9 s) | 15.92 fps, 99.5 ms |
| 3 | 17.42 fps, lost 485, 173 ms | | 28.44 and 24.55 fps, lost 38 and 194, 143 and 153 ms | 18.43 fps, 88.1 ms |
| 4 | 29.91 fps, lost 4, 117 ms | | 30.01 and 29.98 fps, lost 0 and 1, 99 and 117 ms (no step taken) | 18.93 fps |

The reader at normal priority starved behind the app: at 3 cores the sender wrote 28.5 frames per second and the reader read 17. With all 16 logical CPUs and other programs taking them (x264 and 12 busy processes), normal priority made it worse (23.0 to 24.3 fps against 19.1 to 28.5 at above normal, 3 runs each): so the steps apply only on `FEW_CORES` or fewer (§8). The default is off below 6 cores anyway (§10), now counting the cores the process may use.

### 16.4 The learned finish alone (was: no masks in the stage)

With every section but the finish off, nothing in the main process asked for masks, as its copy of the finish never runs. Now the stage says whether it will use them. `.venv-neural`, a model with a real delta, 330 frames: masks made in the stages, and the changed area of the last frame within 25 % of the in-process run (`test_a_finish_that_alone_uses_the_masks_gets_them_in_the_stage`; the review measured 0 mask updates and 17940 against 59966 changed pixels before).

### 16.5 The bar's tools

`judge.py` passed a read that stopped getting frames, lost frames across the end of the clip without counting them, and compared a rounded mean. It now also asks for at least `(seconds - 5) * 29.9` frames, a last frame within 1 s of the end, the even clock of §2.2, and counts a loss across the end of the clip. `read_numbers.py` opts out of the timer throttling by itself (`BAR_READER_HIDDEN=1` to leave it). `bar_run.py` fails a run whose app died or would not stop, and reports the CPU of the whole process tree (`logical_cpus_all`; `logical_cpus` is the main process only).

### 16.6 Final code: the bar

`bar_run.py` and `judge.py` as in 16.5, 60 s each, logs on. Quiet unless marked.

| Face | natural | polished | strong |
|---|---|---|---|
| 1072 | pass, lost 0, w 30 | pass, lost 0, w 30 | pass, lost 0, w 30 |
| 1012 | pass, lost 0, w 30 | pass, lost 0, w 30 | pass, lost 0, w 30 |
| 1060 | pass, lost 0, w 30 | pass, lost 0, w 30 | pass, lost 0, w 30 |
| 1073 | pass, lost 0, w 30 | pass, lost 0, w 30 | pass, lost 0, w 30 |

No slip and no drop in these 12. Worst stage p99 22.93 ms (1073 polished), way in p99 at most 1.92 ms, CPU of the whole app 2.20 to 3.10, to write p50 82.74 ms on 1072 strong and 62.76 ms on 1012 natural, write against its place within 0.17 to 0.49 ms.

Next to `load.py`, x264 at 30 fps in every run: 1072 natural, polished, strong, 1060 strong, 1012 strong all pass, lost 0 (1072 natural one slip, worst 1 s 29). With a second reader as well: 1072 natural and polished pass on both readers; 1072 strong 10 of 10 pass on both readers. Worst stage p99 with the load 25.02 ms (1060 strong), CPU at most 3.53.

28 minutes, 1072 strong, quiet: pass, 50252 frames, lost 0, 0 slips, 0 drops, worst 1 s 30, biggest gap 35.9 ms, stage p99 16.2 / 15.0 / 20.4, to write p50 84.2 ms, CPU 3.12.

Identity (`impl/identity.py`, script and plate, 294 frames each): 0 differ on natural, polished and strong.

### 16.7 What still fails

**The phase of an opening, and nothing else.** Before the final code, the same 1072 strong with `load.py` and a second reader ran 15 times: 2 failed, on the first reader only, lost 1 and lost 114. In both the second reader on the same camera at the same time lost nothing, and the sender's log has no slip, no drop and every write within 0.39 ms of its place. That is the bad phase of §14: the reader's tick falls on the end of a write, and its jitter, larger next to a load, sends some ticks to either side. The independent run's failure of that combination (both readers lost frame 1980) had gaps of 58.8 and 41.1 ms and of 62.6 and 37.4 ms around it, off the 33.3 ms clock in both readers at once: both readers were late by 25 to 29 ms at the same moment, which no write can cause. Reader openings (`impl/soak.py`, 4 s each, final code): quiet 108 of 110, 2 with a loss of 1 and 3 frames; with `load.py` 99 of 100, 1 with a loss of 2; the sender had no slip and no drop through all of them. The same openings on an exact writer with no app at all (`fix/exact_openings.py`, the numbered pattern on a fixed grid, end of write within 0.44 ms): quiet 110 of 110, with `load.py` 98 of 100 (2 with a loss of 1). So next to a load the lottery is the same with or without the app; quiet, 2 of 110 against 0 of 110 is too few to tell apart. §13 asks 110 of 110: **not reached.** Neither the processes' CPU nor segmentation on the GPU would change a reader's tick, so the second was not done here.

**After a slip the delay stays.** Next to `load.py`, 3 of the 17 runs of 16.6 had one slip (each +33 ms for the rest of the run: to write 110 to 122 ms instead of 77 to 97). Quiet: 0 slips in the 12 runs and the 28 minutes. It can only shrink by losing a frame (§6); the status line now shows it.

**A sender that dies while a reader holds its queue** is replaced only through the rescue's private layout (§8), on its grid. Drills on the final code: the sender killed, 30.0 fps in the last 5 s; the sender hung, 29.8 fps.
