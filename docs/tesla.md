# Tesla Model 3 (2021, HW3)

Upstream openpilot supports the Model 3 with HW3 (2019–23) using angle steering; no
fork is needed. Sources: comma's supported-vehicles list, the opendbc Tesla code and the
comma community's Tesla channel.

## Before install day

- Car: Software → Additional Vehicle Information → **Autopilot computer = HW3**, so
  the **Tesla A** harness (behind the glovebox, right side; no glovebox removal on
  HW3, but tight). HW4 cars use Tesla B. Note the Tesla software version.
- Tesla steering protocol: firmware from 2026.8.6 uses a 3-bit
  `DAS_steeringControlType`. openpilot `59fcd340` (the `vendor/openpilot` submodule,
  step 0 of [deployment.md](deployment.md)) detects it and puts old 2-bit firmware in
  dashcam-only mode ("the car's software needs to be updated"). Older openpilot,
  including 0.11.2, engages without steering on new firmware.
- Car settings: Traffic-Aware Cruise Control, Autosteer and FSD off (enabled
  Autosteer is an invalid-LKAS fault). Do not accept Tesla updates before the run.
- The Vehicle CAN addon, the A-pillar connector, radar, MADS and cooperative
  steering are not needed.

## First drive (about 10 minutes)

1. Install, boot fully (TACC can engage before openpilot finishes booting), and
   confirm the comma shows `TESLA_MODEL_3`. The staging build may first flash a newer
   AGNOS: tap Update on its screen, then allow about 10 minutes online. Car firmware that
   openpilot does not recognize yet fingerprints as `MOCK` (VIN all zeros, no firmware
   match) despite live CAN: add `export FINGERPRINT=TESLA_MODEL_3` to
   `/data/openpilot/launch_env.sh` and restart, since activation needs Park telemetry an
   unrecognized car never sends; then activate with `--fingerprint TESLA_MODEL_3`, which
   owns that line from then on.
2. Enable openpilot longitudinal control. A release channel (`release-mici-staging`
   among them) hides the toggle and clears every development-only param at each manager
   start, so the param alone never survives; the channel is the local git branch name, so
   rename it, then set the param and restart, parked:
   `git -C /data/openpilot branch -m release-mici-staging drivingbench`,
   `echo -n 1 > /data/params/d/AlphaLongitudinalEnabled`. Its only visible cost is the
   untested-branch startup alert. Without it the producer reports
   `unsupported_vehicle: … enable AlphaLongitudinalEnabled`.
3. Activate the release ([deployment.md](deployment.md)), parked.
4. Settings → Device → **Reset Calibration**, then drive straight above 15 mph for
   2–3 minutes until calibration completes; openpilot will not engage before.
5. In the lot: `straight 0%` at 1 m/s for 10 s. If speed overshoots or lags, tune
   `speed_control` in Setup (no redeploy).
6. Steering scale **340°** (EPAS faults above 360°), the default, but a comma with saved settings
   from another car keeps its old scale: check Setup shows 340 before any command, then one
   100% turn each way. opendbc's
   Model 3 geometry (12:1, 2.875 m wheelbase) predicts a 5.8 m radius, a 90° turn in
   about 9 s at 1 m/s.

## Each evaluation session

After Bring online, press the gear shift stalk down once to engage. Sometimes this needs a
brief slow manual drive first. Once engaged, stay stopped in Drive, release the brake when
ready, and press Start attempt (or start from the chat app). The UI's vehicle guide defaults
to this Tesla reminder; selecting Toyota changes the guidance only. Connecting the laptop never
engages vehicle control by itself.

## Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| Engages but does not steer, or invalid LKAS | Old openpilot on 3-bit firmware, or Autosteer on | The [step 0](deployment.md#0-the-matching-openpilot-on-the-comma) build; TACC with Autosteer off |
| Dashcam only, "software needs to be updated" | 2-bit Tesla firmware | Update the car |
| Unrecognized or dashcam only after a Tesla update | New firmware not yet fingerprinted | Re-activate with `--fingerprint TESLA_MODEL_3` |
| `unsupported_vehicle` in the producer status | Alpha longitudinal off, or another car | First drive step 2 |
| Tesla warnings right after plugging in | Connector not fully seated | Re-seat; let the car sleep once |
| Install at 93% for a long time | Source builds compile on the device | Wait (up to an hour); keep power and network |
| Boot loop after a reset | Device install state | comma's recovery procedure for comma four |
| Disengages when the wheel is touched | Tesla safety disengages on driver override | Hands off the wheel while engaged |
