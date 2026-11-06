# Deploy a release to the comma

One operator does this, once per release, with the car in Park and openpilot
disengaged. Laptops never redeploy the comma by connecting; **Bring online**
only starts the release that is already installed. Steering-scale,
speed-ceiling, camera and objective changes are shared settings and need none of
this.

DrivingBench drives a Toyota with torque steering or a Tesla with angle steering, each
with openpilot controlling speed (on a Tesla, `AlphaLongitudinalEnabled`; see
[tesla.md](tesla.md)). Any other car latches `unsupported_vehicle: …` as the producer's
`native_error` and never moves.

## 0. The matching openpilot on the comma

The native changes are written against one openpilot build: the comma four build
`release-mici-staging` `fc6b67f9c8`, made from upstream `59fcd340`, which is the commit
of the `vendor/openpilot` submodule. To put it on the comma, uninstall and enter the
custom installer URL `installer.comma.ai/commaai/release-mici-staging` (prebuilt; no
on-device compile). Activation checks the two native files, not the commit: a later
staging build is accepted while its `controlsd.py` and `drive_helpers.py` are
byte-identical to the submodule's, and refused (`Unknown native edits`) otherwise, rather
than running mixed code. Reinstalling openpilot may clear `/data`: run
`drivingbench sync` first so no unsynced traces are lost. Activation sets
`DisableUpdates` so the build stays put; without it openpilot would also refuse to go
onroad after days without an update check. A fresh install has no DrivingBench exports,
so stock openpilot runs (and reports Park) until activation.

## 1. Build the bundle on a laptop

From a clean, committed checkout with the submodule initialized ([Set up](../README.md#set-up)):

```sh
uv run drivingbench bundle \
  --settings-file device/initial-settings.json \
  --output ~/drivingbench-releases/$(git rev-parse --short HEAD).tgz
```

The bundle contains the complete aarch64 Python tree for the comma (every
dependency resolved from `uv.lock` with hash checks, plus this package), the
modified openpilot files from `native/`, a `manifest.json` with the release, the
submodule commit and each file's stock (`baseline`, read from `vendor/openpilot`) and
`installed` digests, the reviewed initial settings, and `activate.py`. Nothing is
downloaded on the device. Bundling refuses a dirty checkout (a moved submodule counts)
and a missing submodule.

`device/initial-settings.json` carries the reviewed longitudinal coefficients (gain
1/s, acceleration cap 1 m/s², deceleration cap 2 m/s², slew 1 m/s³; the slew is the
jerk limit that keeps speed changes smooth). The bundle refuses a settings file that
omits any of them, so a code default can never silently retune the launch. Settings
are seeded from this file only on the very first producer start under
`/data/drivingbench-v01`; later restarts keep the operator's saved values. The
producer refuses saved settings with fields it does not know: after upgrading from
v1 (whose settings carry `prompt` and two retired speed-loop fields), remove
`/data/drivingbench-v01/state/settings.json` once so this file seeds it again.

## 2. Copy and activate on the comma

```sh
scp ~/drivingbench-releases/<release>.tgz comma:/data/
ssh comma
mkdir -p /data/drivingbench-v01/incoming/<release>
tar -xzf /data/<release>.tgz -C /data/drivingbench-v01/incoming/<release>
/usr/local/venv/bin/python /data/drivingbench-v01/incoming/<release>/activate.py
# add --fingerprint TESLA_MODEL_3 if a car software update leaves the car unrecognized
```

`activate.py` checks every precondition before it writes anything, so a failed
attempt is fixed and simply re-run: every native file on the device matches the
bundle's baseline, the bundle's own content, or the previously installed
release; fresh `carState`/`selfdriveState` show Park, zero speed, and
disengaged; the previous producer (if any) has exited; and port 8876 is free.
It then copies the bundle to `releases/<release>.partial`, re-reads Park and the
native files, and renames it atomically to `/data/drivingbench-v01/releases/<release>`
(an identical release already there is reused; a different build of the same commit
is refused). Next it imports the staged runtime with the native interpreter
(`drivingbench.*`, `openpilot.cereal`, `msgq.visionipc`, `PIL`, `numpy`); an import
error aborts with the message and native files untouched. Then it sets openpilot's
`DisableUpdates`, backs up the native files and `launch_env.sh` into
`<release>/native-backup/`, writes the native files atomically, and replaces every
`DRIVINGBENCH_*` (and `FINGERPRINT`) export in `/data/openpilot/launch_env.sh` with
this release's (`DRIVINGBENCH_MODE=experimental`, `DRIVINGBENCH_PYTHON_ROOT`,
`DRIVINGBENCH_NATIVE_SOCKET`, and `FINGERPRINT` only when `--fingerprint` is given:
opendbc's own override, which skips car recognition). Driver monitoring is stock.
Activation points `current` at the release and `previous` at the one before. A failed
write restores the backups before raising.

Then restart openpilot while still parked (reboot, or the usual
`source /etc/profile; /data/continue.sh` after stopping the manager). Back on a
laptop, **Bring online** starts the producer via `current/start-producer` and the
UI should report the new release and protocol 2. Update the native bundle, producer
and laptop gateway together; the gateway refuses to start motion against a producer
with a different protocol. The launcher restarts a crashing producer up to five
times, two seconds apart, then gives up and says so in `state/producer.log`; Bring
online again after fixing the cause.

## Rollback

Each release directory keeps the native files and `launch_env.sh` exactly as they
were before it was activated. Rollback only works between releases built for the
same openpilot build; across builds, reinstall that build first (step 0). To return to
the previous release, parked and disengaged:

```sh
cd /data/drivingbench-v01
cp current/native-backup/openpilot/selfdrive/controls/controlsd.py           /data/openpilot/openpilot/selfdrive/controls/controlsd.py
cp current/native-backup/openpilot/selfdrive/controls/lib/drive_helpers.py   /data/openpilot/openpilot/selfdrive/controls/lib/drive_helpers.py
cp current/native-backup/launch_env.sh                                       /data/openpilot/launch_env.sh
ln -sfn "$(readlink previous)" current
```

Stop the running producer (`kill $(cat state/producer.lock)` after confirming
the PID), restart openpilot, then Bring online. Saved shared settings are kept.
To remove the harness entirely, restore the first release's `native-backup/` (the
stock files) and delete the `DRIVINGBENCH_*` exports from `launch_env.sh`.

## Supervised check

Offline tests verify the command pipeline. Before calling physical steering
verified, one short supervised session in the car should cover, with the
operator ready on the brake:

0. Calibrate the steering scale first: [steering-calibration.md](steering-calibration.md)
   (a Tesla's first drive: [tesla.md](tesla.md)).
1. Engage (Toyota: SET, release the brake, then RES if requested): a `straight 0%`
   command at 0.6 m/s launches and expires into a hold; `observe` shows `held` afterwards.
2. `left 50%` then `right 50%`, each replaced before expiry: native limits govern
   wheel response, and measured steering follows the target sign (positive left).
3. Expiry mid-turn: braking starts and the wheel stays where it was; no
   automatic centering.
4. `stop_now` from a second chat/MCP process, then a new command from the first
   without an authorization step.
5. Press the brake or disengage native control: motion is cancelled and new
   commands are rejected until vehicle readiness returns. After recovery, verify
   a fresh command works and the cancelled command does not resume.

Record the release, the settings revision, and what was seen in a short note in the
trace commit.

## Native changes

`vendor/openpilot` is stock upstream openpilot `59fcd340` (with its `opendbc_repo`
submodule), the source of the step 0 build. DrivingBench modifies two of its files,
listed in `device/deploy.py::NATIVE_FILES` and deployed together; the modified copies
live in `native/` at the same relative paths, so
`diff -u vendor/openpilot/<file> native/<file>` is the complete record:

| File | Change |
| --- | --- |
| `openpilot/selfdrive/controls/controlsd.py` | Opt-in `DRIVINGBENCH_MODE=experimental` seam: imports `drivingbench.device.native.Bridge` from `DRIVINGBENCH_PYTHON_ROOT`; per tick hands `carState`/`vehicleParameters`/`selfdriveState`, `latActive`, `longActive` and the `VehicleModel` to the bridge; takes acceleration, `should_stop`, and `resume` from it; converts the target wheel angle (or a policy chunk's curvature) through `apply_curvature` instead of the stock planner path; records the requested torque and wheel angle, the car-controller output, Panda blocked-transmit counters, EPS/driver torque, steering rate and fault flags as evidence; a forced-PID launch and a standstill PI reset prevent integrator wind-up while the car is held; soft-disable deceleration remains. Stock mode never imports, subscribes to the additional `pandaStates` evidence, or opens anything. |
| `openpilot/selfdrive/controls/lib/drive_helpers.py` | Lateral acceleration cap raised from 3 to 12 m/s² before roll compensation (parking-lot turning at walking pace is far below either), including stock callers in this build; lateral jerk remains 5 m/s³. `clip_curvature(..., max_curvature=MAX_CURVATURE)`: DrivingBench commands pass the curvature of the configured wheel-angle range instead of the generic ±0.2/m cap. Stock callers keep the default. |

`longcontrol.py`, `selfdrived.py` and everything else are unmodified: stopping, hold
behavior, driver monitoring, the car interfaces and the Panda safety model are stock.
To move to another openpilot version, point the submodule at the new commit, re-apply
the diff to `native/`, run the tests, bundle, and do the supervised check again.
Preserve `native/LICENSE` (openpilot's MIT license for the modified files).
