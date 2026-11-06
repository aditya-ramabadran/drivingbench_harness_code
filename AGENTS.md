# Contributing to DrivingBench Harness

Conventions for people and coding agents working in this repository. Read
[README.md](README.md) first, including its disclaimer: this code moves a car.

## Ethos

- Work like a staff engineer: intelligent, elegant system design; clean, reliable
  code. Avoid unnecessary complexity. Elegance is the goal, not a nicety.
- No redundancy. Reuse a utility or create one when two places do the same thing.
- When unsure, test and let real results decide. For important design decisions
  you are unsure about, ask.
- No shims or compatibility layers. Do what needs doing directly.
- Verify end to end and dig in until the work demonstrably works. Be adversarial in
  tests: cover the edge cases, not the happy path.
- Code must be as clean, simple, and readable as possible. Docs must be
  information-dense and concise.

## Rules

- Keep it small. The primary path is `mcp/tools.py` → `gateway/app.py` →
  `device/service.py` → `device/native.py` → `controller/core.py`. Controller
  decisions stay in `controller/core.py`; helpers are for concrete plumbing only.
- Do not add: route executors, path conditioning, turning profiles, writer ownership,
  per-chat registration or attach, session lifetimes, camera-age admission gates, a fake
  MCP, compatibility aliases, or limits that the car and openpilot do not require.
- The public MCP is exactly `drivingbench_sandbox` with `observe`, `set_motion`,
  `stop_now`. Tool inputs and outputs stay tiny; images are separate content
  blocks on `observe` only. Speed above the shared ceiling is rejected; there is no
  fixed ceiling in schemas. The optional `reason` on `set_motion`/`stop_now` is recorded
  evidence: never required, never echoed, never read by the controller.
- Steering is a percent of one shared scale in steering-wheel degrees, calibrated
  in the car ([docs/steering-calibration.md](docs/steering-calibration.md)); tire
  angle never appears outside `VehicleModel`. `observe` reports the same percent.
- The runner ([docs/runner.md](docs/runner.md), `harness/`) is a client of the producer API,
  hosted by the gateway. A chat model with no driver gets exactly the public tools. Trained
  policies send `/chunk` (timed curvature and speed setpoints through the same controller and
  limits); chunks are a producer command, never an MCP tool. Keys come only from the macOS
  Keychain items `drivingbench-openrouter` and `drivingbench-gemini`, never from the
  environment or a file; the policy registry holds no credentials.
- The simulator ([docs/sim.md](docs/sim.md), `sim/`) sits below the unchanged producer: only
  the native process, the car and the cameras are replaced. Keep every client driving it
  exactly as it drives the car.
- `traces/` holds published **segments** (automatic, per engagement) and labeled
  **sessions** (operator-declared benchmark attempts). Before pushing from a laptop
  that has driven, run `uv run drivingbench sync` and commit what it publishes; it
  attaches the chat transcripts and runner logs that drove each segment. Never delete,
  rewrite, or hand-edit a published segment or session; never commit a running one
  (no `segment_end` / `ended_at`); no Git LFS, no commit hooks. The model never sees
  sessions; do not add them to the MCP tools.
- Laptop install, update and the client checks are in [docs/install.md](docs/install.md).
  Pulling Git changes nothing that runs; re-run `install` to refresh the runtime copy.
  Never redeploy the comma to add a laptop.
- Deploying to the comma, rollback, the supervised car check, and the record of the
  openpilot changes are in [docs/deployment.md](docs/deployment.md). Native changes
  need one coordinated device deployment; shared-setting changes need none. Do not
  touch the device unless the operator explicitly asked for device work.
- `vendor/openpilot` is a submodule pinned to the openpilot the comma runs; never edit it.
  `native/` holds DrivingBench's two modified files (`device/deploy.py::NATIVE_FILES`).
  Moving to another openpilot means re-deriving both, checking the diff, and a supervised
  check; preserve `native/LICENSE`.
- Tests: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`,
  `git diff --check`, and the `node tests/*.cjs` UI checks. `tests/test_e2e.py` runs
  two real MCP processes; `tests/test_traces.py` replays measured telemetry. Offline
  results never count as physical steering verification.
- Small, focused commits on feature branches; pull requests into `main`.
