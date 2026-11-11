# Simulator

A real-time simulator of the v1.1 benchmark course: the Foothill College northern lot
(37.364, −122.132) as it was laid out on 27 October 2026, driven by a Tesla Model 3. Every client
drives it exactly as it drives the car.

- **The real control stack.** `device/service.py` runs unchanged; only the comma's native process
  and camerad are replaced. `SimNative` (`sim/car.py`) runs DrivingBench's `Bridge` and
  `Controller`, openpilot's `LongControl` and `LatControlAngle`, and the Tesla car controller's
  angle limits (50 Hz) and acceleration clip (25 Hz), tick for tick as `controlsd` and `card` run
  them. openpilot comes from the `vendor/openpilot` submodule with DrivingBench's patched
  `drive_helpers.py` (`native/openpilot`) loaded in its place, as on the comma (`sim/logs.py`).
- **A car fitted from its logs** (`sim/plant.py`, `sim/params.json`). Speed: the applied request
  through a delay and a first-order lag; from standstill a positive request must persist for the
  launch wait, then the car departs at the measured lurch. Wheel: the EPAS follows the applied
  angle through a delay, a lag and a rate cap, only while openpilot steers (the Tesla steers at
  standstill). Pose: yaw rate = scale × speed × openpilot's `VehicleModel` curvature with the
  learned steer ratio, stiffness, offset and roll; the rear axle advances at true speed
  (0.976 × `vEgo`, from GNSS Doppler) and the comma rides 2.28 m ahead of it. The plant is fitted
  to the requests the car received, not to tool calls, so it is independent of the controller
  above.
- **A camera.** The road camera renders a Gaussian splat of the lot, trained from the car's own
  footage, with the real cones removed and the course's cones inserted as objects
  (`sim/render.py`), at the measured mount (1.24 m up, 4.6° pitch, 1.8° right of travel, 36.3°
  vertical field of view, 1344×760) and auto-exposed to the road camera's mean luma. Each frame
  shows the pose at its capture time, stamped on the camera clock, so capture-anchored plans and
  image ages behave as on the car. The wide camera is a 60° pinhole stand-in.
- **The course** (`sim/course.py`): the course line (the reference run's smoothed GNSS path) and
  finish zone, the reconstructed ground, 144 cones measured from footage and completed into two
  rows, and the drivable region, the lane between the rows widened 1.2 m beyond the cones. The
  car's whole body (the Model 3's footprint around the comma) stays inside it: a step that would
  leave it does not happen, and the car stands at the edge until a reset.

## Setup

```sh
export GIT_LFS_SKIP_SMUDGE=1   # skip openpilot's model and asset blobs; the simulator needs none
git submodule update --init --depth 1 vendor/openpilot
git -C vendor/openpilot submodule update --init --depth 1 opendbc_repo
uv sync --locked
```

The course, ground, cones, drivable region and fitted car are committed under `sim/`. The splat
is published on Hugging Face ([drivingbench/simulator](https://huggingface.co/datasets/drivingbench/simulator),
CC BY 4.0); download it where the site file expects it:

```sh
uvx --from huggingface_hub hf download drivingbench/simulator --repo-type dataset --include "*.splat" --local-dir runs/sim/explorer
```

The drives it is built from are not published: build a splat from drives of your own with the
pipeline (below), or run with `--placeholder` (plain frames). The splat camera renders in headless
Google Chrome at its macOS location.

## Run

```sh
uv run python -m drivingbench.sim.runtime --placeholder   # plain sky-and-asphalt frames
uv run python -m drivingbench.sim.runtime                 # the splat camera
```

The producer starts on `127.0.0.1:8877` (`--port`; state in `runs/sim/state`). 8877 is the
gateway's `tunnel_port`, its forward to the comma's producer, so with the comma not brought
online the gateway, the UI's Drive and Runner tabs and every chat's `drivingbench_sandbox` drive
the simulator unchanged. Clients that talk to a producer directly use the same address.

Two operator-only routes score an episode; the model never sees them, the course line or the
drivable edge:

- `GET /sim/episode`: `progress` (v1.1's rule: the course line resampled every metre to its first
  sample in the finish zone; the furthest sample within 4 m of the comma, searched from 10 m
  behind to 20 m ahead of the furthest reached, at 5 Hz), `distance_m`, `cone_contacts` (each
  cone, by its index in the course's cones, whose base the body overlapped, with the time of first
  contact), `off_course_s` (more than 4 m from the line), `finished_at` (full progress, the comma
  in the finish zone, stopped) and `edge_at` (when the drivable region's edge first stopped the
  car). Times are the simulator's clock, this machine's monotonic seconds.
- `POST /sim/reset`: stop, back to the start, a new episode.

Driving straight from the start, the car touches a cone of the first turn and the edge stops it
about 20 m on.

## How faithful

**The car.** Fitted on 26 October (1,561 speed and 13,939 steering 3 s windows): speed delay
0.1 s, lag 0.1 s, gain 1.0; wheel delay 0.04 s, lag 0.1 s, 400 °/s cap; launch wait 0.19 s at a
0.15 m/s² lurch; yaw scale 1.02. Held out on 27 October, the plant replays each command window's
logged requests from the logged state (`sim/validate.py`); median absolute error against the car
(gyro heading, speed dead-reckoned position):

| 2,402 windows | 5 s | 10 s | 30 s (34) |
| --- | --- | --- | --- |
| Fitted: speed, wheel, heading, position | 0.029 m/s, 0.32°, 0.28°, 0.11 m | 0.032 m/s, 0.33°, 0.53°, 0.29 m | 0.057 m/s, 0.45°, 1.4°, 0.75 m |
| Ideal car | 0.031 m/s, 1.45°, 0.35°, 0.13 m | 0.035 m/s, 1.72°, 0.51°, 0.30 m | 0.052 m/s, 1.77°, 2.1°, 1.15 m |

The wheel is where the model matters; the Tesla's speed response is close to ideal. This
validates the car below openpilot, not DrivingBench's controller.

**The scene.** One reconstruction holds 10,182 frames from both road cameras over the day: 0.60 px
reprojection, 1.9 m median from GPS it never saw, per-step heading 0.05° from the gyro and travel
3.3 cm from GNSS Doppler (median). The splat, trained on 3,354 road-camera frames from
15:30–17:00, scores 22.6 dB PSNR and 0.80 SSIM on 168 held-out views. Of the course's 144 cones,
111 stand as measured, 8 were moved onto their row and 25 fill gaps few frames saw; against the
cones the reference run itself detected within 12 m, its recall is 83% and its precision 70%.
Every reconstructed pose of the five attempts that completed this layout keeps the whole body
inside the drivable region.

**Real attempts replayed** (`sim/faithfulness.py`, the two attempts that scored 100%). From each
reconstructed pose, the fitted car runs the logged requests: median position error 0.06 m at
0–2 s, 0.21–0.25 m at 2–5 s, 0.48–0.62 m at 5–10 s, against 0.06, 0.20–0.24 and 0.40–0.51 m for
the logged speed and gyro integrated, so the car model adds about 0.1 m at 10 s. Rendered at the
real camera poses, the simulated camera scores 21.6 and 18.7 dB PSNR (0.74 and 0.76 SSIM) against
the real frames. What remains is the splat: floaters where the sky is clear, flatter asphalt,
close foliage, and one splat for the afternoon's changing light. Lens distortion (8 px at the
corners) is not modelled.

The splat camera is ready about 5 s after start and renders all 3.0 M splats in 107 ms median per
frame (194 ms p95).

## Explorer

```sh
uv run python -m drivingbench.sim.explorer.serve   # http://127.0.0.1:8890
```

The splat and inserted cones in the browser (three.js and Spark from jsDelivr), in the course
frame: a car view at the road camera's height, a raised view (`2`), the course line, finish and
drivable edge (`C`), the cones on and off (`K`). Walking stays inside the drivable region.
`?file=full` loads all 3.0 M splats (the default keeps the 1.5 M most opaque); `?file=raw`, the
splat with its real cones, is not published.

## Build

Everything the simulator loads but the course (its line and finish zone) is built from the car's
drives at the site by one pipeline. The site file (`sim/foothill-oct27.json`;
`DRIVINGBENCH_SIM_SITE` names another) names the course and outputs, the inputs, the simulated
`day` and its `daylight` window, the `fit_day`, the splat's lighting window, the cone-measurement
windows and the reference run; the build and the simulator both read it.

```sh
uv run --group sim-build python -m drivingbench.sim.pipeline                # each stage, done or not, and its commands
uv run --group sim-build python -m drivingbench.sim.pipeline --run          # the missing stages, in order
uv run --group sim-build python -m drivingbench.sim.pipeline --run layout   # named stages again
```

| Stage | Makes |
| --- | --- |
| `series`, `fit` | the car's logs and commands as 100 Hz series; its fitted dynamics (`sim/params.json`) |
| `frames` | both road cameras' frames chosen for coverage over the day, with GPS priors |
| `reconstruct` | GLOMAP's model, aligned to GPS, degenerate frames dropped |
| `level` | one asphalt where passes registered at different heights; every road-camera pose |
| `check` | the model against the gyro and GNSS Doppler it never used |
| `ground` | the asphalt's height (`sim/foothill-oct27-ground.json`) |
| `training-set`, `train`, `score` | the splat (Brush) and its held-out score |
| `clean-splat`, `explorer` | the splat without real cones or the haze the car drove through; the three splats |
| `cones`, `verify`, `layout` | cone candidates per window; those the reference run's frames allow; the cones and drivable region |

Inputs: the openpilot routes driven at the site on its `day` and `fit_day`, as the comma stores
them (`<route>--<n>/` with `rlog.zst`, `fcamera.hevc`, `ecamera.hevc`) under the site's `routes`;
the producer's recordings of DrivingBench's commands (`runs/recordings`, mirrored by `drivingbench
sync`); and the course. The `sim-build` group adds pycolmap, SciPy, contourpy and matplotlib;
frames need ffmpeg; training needs Brush 0.3.0 (the official release binary; check its SHA-256),
as `brush_app` on PATH or `BRUSH`. Slow stages: `reconstruct` (hours of CPU, about 34 GB of memory
for a day), `level` (about 15 min), `train` (about 40 min on a GPU Mac). Every stage but `train`
is a module that also runs on its own with the arguments the listing shows.

A person still chooses the site file's windows (one lighting window for the splat, cone windows
after the course was last set out, the run the course line came from) and reads each stage's
report: `check` for seams, the held-out score and sheet, the cone figures and the layout's counts.
Then drive it and replay real attempts:

```sh
uv run python -m drivingbench.sim.validate                 # held-out windows → runs/sim/validate
uv run python -m drivingbench.sim.faithfulness SESSION     # path and video → runs/sim/faithfulness
```
