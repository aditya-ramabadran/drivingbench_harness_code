# DrivingBench Harness

A small harness that lets an AI model drive a real car at parking-lot speed while a human
sits in the driver's seat with a foot over the brake. It runs on a [comma](https://comma.ai)
device with [openpilot](https://github.com/commaai/openpilot) in a supported Toyota or Tesla.
Any chat app that speaks MCP (Codex, Claude Code, Cursor) gets one server,
`drivingbench_sandbox`, with three tools: `observe`, `set_motion`, `stop_now`. Models without
a chat app, decision models and trained driving policies drive through the same interface from
the laptop's runner loop, and a simulator of the benchmark course runs underneath all of
them. Everything the model says and does, and everything the car did, is recorded as a trace
you can replay.

This is research software for closed courses. Read the next section before anything else.

## Disclaimer and safety

**Use at your own risk.** This software commands the steering, acceleration and braking of a
motor vehicle based on the output of an AI model. Models make mistakes; so does this code, the
device, the car, and the person supervising. The authors provide this software as is, without
warranty of any kind, and accept no liability for any damage, injury, loss, fine, or other
consequence arising from its use or from anything derived from it. If you use it, you are the
driver and the operator, and you are responsible for the vehicle at every moment.

The authors are not affiliated with, endorsed by, or supported by comma.ai, openpilot, Toyota,
Tesla, or the makers of any model or application named here. openpilot's own terms and safety
model apply to the device; this project modifies openpilot in ways its authors never intended
or reviewed.

Do not trust the software limits. The harness has a speed ceiling, an emergency speed cut-off,
command expiry, and cancels motion on brake, pedal, disengagement, or invalid sensors. These
are ordinary code paths on an unreviewed system, not safety functions; any of them can fail.
The safety measures are the human ones:

- A licensed, sober, attentive driver in the driver's seat, foot over the brake, hands ready,
  at all times. The brake or the steering wheel always wins: pressing either cancels the
  model's command and disengages openpilot.
- Private property you have permission to use, empty of people and other vehicles, with soft
  obstacles (cones) only. Never on public roads, never near pedestrians, never with
  passengers, never at night or in rain until you have far more experience with the system
  than this README can give you.
- Walking pace. The default ceiling is 3 m/s (about 7 mph); keep it there.
- One operator supervises the car and the UI; treat every command in the history as something
  you personally authorized.
- Do a supervised check of every release before letting a model drive with it
  ([docs/deployment.md](docs/deployment.md#supervised-check)), and calibrate the steering scale
  on your car ([docs/steering-calibration.md](docs/steering-calibration.md)).
- Know your local law. Operating a modified driver-assistance system, even on private ground,
  may be regulated where you live. That is your problem to solve before you drive.

## What's new in v1.1

- **Tesla Model 3** (HW3, angle steering) alongside the Toyota, on openpilot `59fcd340`, the
  source of the comma four's `release-mici-staging` build that supports Tesla's current
  steering firmware ([docs/tesla.md](docs/tesla.md)).
- **The runner loop** ([docs/runner.md](docs/runner.md)): a chat model on OpenRouter or
  Google's Gemini API drives with exactly the public tools and no chat app, or supervises a fast
  driver: a decision model, or a trained policy (VLA) served over HTTP that sends timed
  curvature and speed plans anchored to its frame's capture.
- **The simulator** ([docs/sim.md](docs/sim.md)): the v1.1 course as a Gaussian splat, driven
  by the real control stack over a Tesla fitted from its logs, with progress, cone contacts and
  the finish scored as on the car. Every client drives it unchanged.
- A simpler, retuned speed loop (proportional, bounded and jerk-limited), and the producer's
  status reporting the car's measured curvature alongside its steering.

## How it works

```text
Codex / Claude / Cursor chat         runner loop (chat model · decision model · trained policy)
            │                                   │
   drivingbench_sandbox (MCP)                   │      observe · set_motion · stop_now · /chunk
            └──────────────┬────────────────────┘
                  laptop gateway ───── Setup / Drive / Runner / Traces UI
                           │
                       SSH tunnel                      (or the simulator, on the same port)
                           │
                    comma producer                     cameras · shared settings · recorder · HTTP
                           │
          native control loop, 100 Hz                  timed command · steering target · speed loop
                           │
                  openpilot → car
```

The comma runs openpilot with two files changed ([native/](native/), recorded in
[docs/deployment.md](docs/deployment.md#native-changes)). In `DRIVINGBENCH_MODE=experimental`,
openpilot's control loop hands each 100 Hz tick to the harness controller instead of its own
planner: the controller holds at most one timed command, turns the requested steering percent
into a wheel-angle target and then a curvature through openpilot's own vehicle model and
limiter, runs a small speed loop for the requested speed, and gives the acceleration, steering
and stop decision back to openpilot, whose car interface and Panda safety model still enforce
the usual actuator limits. Without the environment variable, openpilot is stock.

A producer process on the comma serves cameras, shared settings, `observe` and commands over
HTTP on localhost; the laptop gateway reaches it through an SSH tunnel and serves the operator
UI. Each chat app spawns a tiny MCP process that talks to the gateway. No client "owns" the car:
any connected chat, or the runner, can replace the active command or stop.

### The three tools

```python
# Road image(s) and a small state summary: speed, measured steering_percent, whether a
# command is executing or the car is held, timestamp, image age.
observe()

set_motion(
    direction="left",  # "left" | "right" | "straight"
    steering_percent=50,  # 0–100 of the shared steering scale; 0 is straight
    speed_mps=1.0,  # 0.5–3.5 m/s; rejected above the operator's speed ceiling
    duration_s=10,  # 5–60 s; the command expires and the car brakes
    reason="Cone gap opens to the left; ease in.",  # optional, recorded, never executed
)

stop_now(reason="Pedestrian ahead.")  # cancel motion and begin braking
```

Each accepted `set_motion` replaces the previous one and starts immediately; expiry or
`stop_now` starts braking, and `observe` tells the model whether the car actually stopped.
`steering_percent` is a percent of one shared setting, the **steering scale**, in
steering-wheel degrees at 100% (default 340°, the Tesla's; calibrate it per car).
`observe` reports measured steering on the same scale, positive left; the model never sees
degrees. `reason` is evidence for the operator: it appears in the Drive-tab history and in the
trace and never reaches the controller.

### What the software enforces

- Motion needs openpilot engaged, Drive, released pedals, valid sensors, and openpilot's own
  longitudinal control active. Brake, gas, steering override or disengagement cancels the
  command and rejects new ones until the car is ready again; nothing resumes on its own.
- Speed above the operator's shared ceiling (0.5–3.5 m/s, default 3) is rejected. Measured speed
  above the fixed 6 m/s emergency limit cancels motion and refuses commands.
- Commands expire after their duration (5–60 s for `set_motion`, at most 6 s for a chunk); the
  driver must keep observing and issuing.
- Steering goes through openpilot's lateral limiter and the car's own torque or angle and rate
  limits; the harness raises openpilot's lateral-acceleration cap for parking-lot turning and
  lets the configured wheel-angle range replace the generic curvature cap. Everything else in
  openpilot's control and safety path is unchanged.
- Every tool call, every 100 ms of telemetry, every settings change and every image the model
  saw is recorded on the device and published as a trace.

## What you need

- A **comma four** running openpilot built from upstream `59fcd340` (the comma four's
  `release-mici-staging` build of 14 October 2026). Activation refuses native files it does not
  recognize, so another build needs the two files in [`native/`](native/) re-derived first.
- A **car openpilot drives with its own longitudinal control**: a Toyota with torque steering
  (recent TSS2 models; developed on a Corolla) or a **Tesla Model 3 with HW3** and angle
  steering ([docs/tesla.md](docs/tesla.md)), with the matching comma harness.
- A **Mac** with Python 3.12 and [`uv`](https://docs.astral.sh/uv/), on the same network as the
  comma. Linux runs the gateway in the foreground; the launchd service, client-config paths and
  Keychain are macOS-only.
- An empty, private lot and a set of traffic cones. Or only a Mac, for the simulator.
- A chat app with MCP support and a model that handles images and tool calls, and/or an
  [OpenRouter](https://openrouter.ai) or Google Gemini API key for the runner.

## Set up

```sh
git clone https://github.com/aditya-ramabadran/drivingbench_harness_code
cd drivingbench_harness_code
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 vendor/openpilot
GIT_LFS_SKIP_SMUDGE=1 git -C vendor/openpilot submodule update --init --depth 1 opendbc_repo
uv sync --locked
```

1. **Laptop:** [docs/install.md](docs/install.md). `uv run drivingbench install --client codex
   --client claude --client cursor --host <comma>` registers the MCP server in each chat app,
   copies the checkout into a runtime venv, and starts the gateway service.
2. **Comma:** [docs/deployment.md](docs/deployment.md). `uv run drivingbench bundle` builds a
   self-contained release; copy it over and run `activate.py` on the parked device. It checks
   every precondition before writing anything and keeps a rollback.
3. **Calibrate** the steering scale on your car:
   [docs/steering-calibration.md](docs/steering-calibration.md).
4. **Supervised check** of the release with a human on the brake:
   [docs/deployment.md](docs/deployment.md#supervised-check).

## Drive

Open <http://127.0.0.1:8766>, **Bring online** (SSH tunnel and the installed producer), set the
shared settings (steering scale, speed ceiling, cameras, objective), engage openpilot in the car,
and either copy the generated prompt into a chat or start a run on the
[runner](docs/runner.md) tab. The Drive tab shows the cameras, the executing command, measured
steering and speed, and every call with its reason. To benchmark, start a **labeled session**
(model, harness, notes) before the attempt and end it with an outcome.

To drive the simulator instead, start it where the tunnel would be and leave the comma offline:

```sh
uvx --from huggingface_hub hf download drivingbench/simulator --repo-type dataset --include "*.splat" --local-dir runs/sim/explorer
uv run python -m drivingbench.sim.runtime   # --placeholder for plain frames, without the splat
```

Afterwards, on the laptop that drove:

```sh
uv run drivingbench sync    # mirror the comma, publish segments and sessions under traces/,
                            # attach the chat transcripts and runner logs that drove them
git add traces && git commit -m "Trace: <what was driven>"
uv run drivingbench upload  # optional: labeled sessions' bulk artifacts to your HF dataset
```

Replay any segment at <http://127.0.0.1:8766/traces>: a scrubber over the frames the model saw,
synchronized with its tool calls and the car's telemetry. [traces/README.md](traces/README.md)
describes the layout.

## Layout

| Path | Owns |
| --- | --- |
| `controller/core.py` | Latest timed command or chunk, expiry, steering target, stop; the guards above |
| `controller/speed.py` | The speed loop |
| `device/native.py` | The `controlsd` hook: vehicle gate, samples in, angle → curvature out |
| `device/service.py` | Producer HTTP: settings, cameras, observe, commands, sessions |
| `device/ipc.py` `cameras.py` `recording.py` `settings.py` | Socket RPC, VisionIPC frames, JSONL recorder, atomic settings |
| `device/deploy.py` | Versioned bundle build and on-device activation |
| `native/` | The two modified openpilot files; the stock ones are in `vendor/openpilot` |
| `vendor/openpilot` | openpilot `59fcd340` (submodule) |
| `gateway/app.py` | Laptop proxy, static UI, runner and trace viewer routes |
| `gateway/setup.py` `install.py` `cli.py` `diagnostics.py` | Config, SSH tunnel, installers, the `drivingbench` CLI, `doctor` |
| `gateway/traces.py` `artifacts.py` `evaluations.py` | Trace publishing, bulk artifacts, attempt metadata |
| `harness/` | The runner: chat model, decision driver, trained policies |
| `mcp/tools.py` | The three tools over plain HTTP |
| `shared/contracts.py` | `Motion`, `Chunk`, `Settings`, limits, protocol number |
| `sim/` | The simulator, its course and fitted car, and the pipeline that builds them |
| `ui/` | Setup, Drive, Runner and trace viewer pages |
| `tests/` | Behavior tests, replayed real telemetry, two-process MCP end-to-end |
| `traces/` | Published segments and labeled sessions (empty until you drive) |

Motion authority lives in the native loop. The producer owns persistent settings and publishes
native state. The gateway and UI keep no state of their own.

## Develop

```sh
uv run pytest -q                                  # whole suite
uv run ruff check . && uv run ruff format --check .
node tests/ui_behavior.cjs && node tests/trace_viewer_behavior.cjs && node tests/runner_ui_behavior.cjs
```

`tests/test_e2e.py` starts two real stdio MCP processes against the real gateway, producer,
socket RPC and controller; only cameras and the car are fixtures. `tests/test_traces.py` replays
measured Corolla telemetry. `tests/test_native.py` and the simulator's tests run the patched
openpilot limiter and the real vehicle model from the submodule. Offline tests verify the command
pipeline; only the supervised check in the car qualifies physical steering. Contributor
conventions are in [AGENTS.md](AGENTS.md).

## License and citation

MIT, see [LICENSE](LICENSE). The two openpilot files under `native/` are derived from openpilot
and MIT-licensed by comma.ai ([native/LICENSE](native/LICENSE)); `vendor/openpilot` is
openpilot itself under its own license. To cite this work use [CITATION.cff](CITATION.cff) or:

```text
Ramabadran, A., Mahns, S., Gessler, T. (2026). DrivingBench Harness: a steering-command harness
for benchmarking AI models in a real car. https://github.com/aditya-ramabadran/drivingbench_harness_code
```
