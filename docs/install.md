# Install and update a laptop

One workflow covers Codex, Claude Code, and Cursor. It edits only the
`drivingbench_sandbox` entry of each client config (with a timestamped backup
beside the file), installs one launchd gateway per laptop, and never deploys
anything to the comma. Device releases are a separate, explicit step:
[deployment.md](deployment.md).

## First install

Requirements: macOS, Python 3.12, `uv`, and SSH access to the comma as `comma`
with its host key already in `~/.ssh/known_hosts` (`ssh comma true` once; comma's
[SSH setup](https://github.com/commaai/openpilot/wiki/SSH) puts your GitHub keys on
the device).

```sh
git clone <this repository> && cd <checkout>
uv sync --locked
uv run drivingbench install \
  --client codex --client claude --client cursor \
  --host comma            # hostname, IP, or ~/.ssh/config alias
# optional: --user comma --key ~/.ssh/id_ed25519 --codex-config PATH ...
#           --dataset <user>/<repo> for Hugging Face artifact uploads
```

What that does:

- Writes `~/.config/drivingbench-v01/config.json` (SSH host/user/key, ports
  8766 web · 8877 tunnel · 8876 producer, device root `/data/drivingbench-v01`).
- Installs this checkout into a runtime venv at `~/.local/share/drivingbench-v01/venv`
  from `uv.lock`, non-editable (macOS refuses launchd agents and GUI apps access to
  `~/Documents` and similar folders, so nothing runs from the checkout), and records
  what it copied in `install.json` beside it. Re-run `install` after pulling to refresh
  the copy.
- Registers `drivingbench_sandbox` in `~/.codex/config.toml`, `~/.claude.json` and
  `~/.cursor/mcp.json` as `~/.local/share/drivingbench-v01/venv/bin/drivingbench-sandbox
  --gateway-url http://127.0.0.1:8766 --client <name>`. Unrelated entries and Codex
  comments are preserved. A project-local Claude `.mcp.json` or Cursor
  `.cursor/mcp.json` in the folder you open can shadow the global entry; check those
  first if a chat sees other tools.
- Installs and starts `com.drivingbench.v01.gateway` (launchd, `KeepAlive`),
  logging to `~/.config/drivingbench-v01/logs/gateway.log`. Pass `--no-service`
  to run `uv run drivingbench serve` yourself instead (the only option on Linux).

Then fully restart the chat apps you registered and open <http://127.0.0.1:8766>:
**Setup**, **Drive via chat app** and **Drive via runner loop** ([runner.md](runner.md)).

Once per client, before trusting it with a moving car: restart the app; confirm the
MCP lists exactly `observe`, `set_motion`, `stop_now`; call `observe` and check it
returns images; call `stop_now` while the car is stopped and confirm it appears in the
tool-call history on **Drive via chat app**.

Optional, per laptop:

- **Runner keys.** Runner chat models and decision drivers read their keys only from
  the macOS Keychain items `drivingbench-openrouter` (OpenRouter) and
  `drivingbench-gemini` (Google Gemini direct); see [runner.md](runner.md#keys).
- **Hugging Face.** With `--dataset`, log this laptop in once (`uv tool install
  huggingface_hub && hf auth login`, a token with write access to the dataset). Without
  a dataset, `sync` still publishes everything to Git and stages nothing.

`uv run drivingbench doctor` checks this laptop read-only: installed code against the
checkout, chat registrations, Keychain items (presence only), transcript locations and
the SSH key. `--network` also checks Hugging Face access to the dataset; `--device`, near
the car, checks SSH and an already running producer. The Setup tab's **Check laptop
setup** runs the same local checks.

## Every drive

```text
Bring online → choose shared settings → engage (Tesla: stalk down; Toyota: SET, RES if asked)
             → use any connected chat → observe / set_motion / stop_now
             → stop_now → confirm held → disengage
```

**Bring online** opens (or reuses) one SSH master with a local forward
`127.0.0.1:8877 → comma:8876`, runs the installed release's `start-producer`
(a no-op if it is already running), and checks the producer reports protocol 2.
It never installs or changes device code. The same is available as
`uv run drivingbench online`; `uv run drivingbench status` shows which clients
are installed.

Shared settings (steering scale, speed ceiling, cameras, image adjustments,
objective) live on the comma. Every laptop reads the same values; saving applies
only the fields you changed. Steering and speed changes affect the next accepted
command, with no redeploy. The prompt is generated from the saved settings each time
it is shown (`ui/prompt.js`); **Start trial** on **Drive via chat app** copies it.

Command limits live in `shared/contracts.py` and are enforced in the MCP schema
and again in the controller: speed 0.5–3.5 m/s (typically 1), duration 5–60 s
(aim for ~10). Below 0.5 m/s a Toyota's launch does not roll; a command shorter
than an observe round trip would brake before the model saw anything. The one
editable speed number is the **speed ceiling** (`speed_limit_mps`, within 0.5–3.5,
default 3); commands above it are rejected. The **emergency limit** is fixed at
6 m/s (`EMERGENCY_SPEED_MPS`): measured speed above it by more than 0.5 m/s, or by
more than 0.05 m/s for 0.5 s, cancels the command and refuses new ones until speed
drops. Lowering the ceiling never turns ordinary launch overshoot into an emergency
stop. Every tool response carries a UTC `timestamp`, so the model can measure its
own latency and size durations.

Client names in tool history are diagnostics only. Nothing about a chat, laptop,
or MCP process grants or blocks motion authority. Engagement, Drive, released pedals
and valid sensors are required for a new command. Stop and intervention cancel
motion; recovery never replays the previous command.

## Update

Pulling Git changes nothing that is running. After `git pull`:

```sh
uv sync --locked
uv run drivingbench install --client codex --client claude --client cursor
```

then restart the chat apps. The gateway is restarted by the installer. If the release
changed `controller/`, `device/`, `shared/`, or `native/`, the comma also needs one
coordinated deployment ([deployment.md](deployment.md)); the gateway refuses to start
motion against a producer with a different protocol number, but stays readable and
stoppable.

Do this per laptop. Do not redeploy the comma to add another laptop.

## Traces after a drive

The comma records every **segment** (engagement → disengagement) under
`/data/drivingbench-v01/state/recordings/segments/<segment>/`: every tool call with
the model's `reason`, settings revisions, ~10 Hz native telemetry while engaged, and
the segment start/end. Observed images are kept full-size on the comma with small
thumbnails beside the segment. Recording is best-effort: failures show in the UI and
history and never block motion.

A **labeled session** groups the segments of one benchmark trial under a model,
harness, and notes: start it on **Drive via chat app** → **Trial** (or `drivingbench
session start --model … --harness …`) before driving, end it with an outcome afterwards. The
Runner labels its own runs. Unlabeled driving needs nothing. **Human demonstration** on the same
form labels a pass you drive by hand (training data for a policy): lot, layout, direction and a
train or val split go into its notes, and the form shows live speed against the commands' range. Layout, rules, and
post-hoc labeling are in [traces/README.md](../traces/README.md).

On the laptop that drove, even if a different chat does the push:

```sh
uv run drivingbench sync    # mirror → publish segments and ended sessions → attach
                            # the driving transcripts → stage labeled sessions' artifacts
git add traces && git commit -m "Trace: <what was driven>"
uv run drivingbench upload  # optional, comma not needed: labeled sessions to the dataset
```

Transcripts are searched in the standard Codex, Claude Code, and Cursor locations plus
`$CODEX_HOME` and the Runner's logs; a second Codex home or any other location goes in
the laptop config as
`"extra_transcripts": {"codex-second": "~/.codex-profiles/second/CODEX_HOME/sessions/*/*/*/rollout-*.jsonl"}`.
