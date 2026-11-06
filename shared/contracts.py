"""The complete command and shared-settings contract. Angles are left-positive."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL = 2
SERVER_NAME = "drivingbench_sandbox"
NATIVE_WATCHDOG_S = 2.0
# A command's reason is asked for in about 25 words (under 200 characters); only a far longer one
# is refused, so a wordy reason never costs a command.
REASON_ASK = "about 25 words (under 200 characters)"
REASON_MAX_CHARS = 600
NOTE_MAX_CHARS = 200  # a session's notes and closing note

# Hard command limits, enforced in the MCP schema and again in the controller.
# Below ~0.5 m/s the Corolla's launch never rolls; a command shorter than an
# observe round trip would brake before the model could see anything.
SPEED_RANGE_MPS = (0.5, 3.5)
DURATION_RANGE_S = (5.0, 60.0)
TYPICAL_SPEED_MPS = 1.0
TYPICAL_DURATION_S = 10.0
STEERING_SCALE_DEG = 340  # default steering-wheel degrees at 100% (see Settings)


def curvature(angle_deg, steer_ratio, wheelbase):
    """Curvature (1/m, left positive) of a steering-wheel angle, as the producer reports it: the
    kinematic bicycle model, since slip is negligible at these speeds. Unlike the wheel angle it
    means the same on every car."""
    import numpy as np

    return np.tan(np.radians(angle_deg) / steer_ratio) / wheelbase


def wheel_angle(curvature_inv_m, steer_ratio, wheelbase):
    """The steering-wheel angle (degrees, left positive) of a curvature: curvature's inverse.
    openpilot's linearized VehicleModel is not, and overshoots by 7% at 300 degrees."""
    import numpy as np

    return np.degrees(np.arctan(curvature_inv_m * wheelbase) * steer_ratio)


# A trained policy's views and the comma road cameras they come from; the road views are cut
# from the narrow frame (harness/vla.py).
POLICY_CAMERAS = {"front": "narrow", "wide": "wide", "road_left": "narrow", "road_right": "narrow"}

# A trained policy's chunk: one setpoint per step, up to 6 s.
CHUNK_STEP_S = 0.1
CHUNK_MAX_STEPS = 60
# An anchored plan runs at least this long after its acceptance, however late it arrived.
ANCHORED_LEASE_S = 0.8
# Fixed native emergency stop, well above any commandable speed plus launch overshoot.
# Never the editable ceiling, so lowering the ceiling cannot make overshoot an emergency.
EMERGENCY_SPEED_MPS = 6.0


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Motion(Model):
    direction: Literal["left", "right", "straight"]
    steering_percent: float = Field(ge=0, le=100)
    speed_mps: float = Field(ge=SPEED_RANGE_MPS[0], le=SPEED_RANGE_MPS[1])
    duration_s: float = Field(ge=DURATION_RANGE_S[0], le=DURATION_RANGE_S[1])
    reason: str = Field(default="", max_length=REASON_MAX_CHARS)  # evidence only; never executed


class Chunk(Model):
    """Timed setpoints from a trained policy; the last one's end brakes, like a motion's expiry.
    Curvature is left-positive and still capped by the shared steering scale."""

    curvature_inv_m: list[Annotated[float, Field(ge=-1, le=1)]] = Field(
        min_length=1, max_length=CHUNK_MAX_STEPS
    )
    speed_mps: list[Annotated[float, Field(ge=0, le=SPEED_RANGE_MPS[1])]] = Field(
        min_length=1, max_length=CHUNK_MAX_STEPS
    )
    reason: str = Field(default="", max_length=REASON_MAX_CHARS)  # evidence only; never executed
    # An anchored plan: row 0 due at `anchor_s`, its input's capture less the actuator lead on the
    # camera clock (the producer converts it to native monotonic time). Each step runs the row due
    # then, holding the last, until the plan's own end, at least ANCHORED_LEASE_S and at most
    # CHUNK_MAX_STEPS steps after acceptance. Without an anchor, rows start at acceptance and the
    # last step's end brakes.
    anchor_s: float | None = None

    @model_validator(mode="after")
    def setpoints(self):
        if len(self.curvature_inv_m) != len(self.speed_mps):
            raise ValueError("curvature and speed need one value per step")
        if any(0 < v < SPEED_RANGE_MPS[0] for v in self.speed_mps):
            raise ValueError("each speed is 0 (brake) or within the motion speed range")
        return self


class ImageSettings(Model):
    exposure_ev: float = Field(default=0, ge=-2, le=3)
    contrast: float = Field(default=1, ge=0.5, le=2)
    saturation: float = Field(default=1, ge=0, le=2)


class SpeedSettings(Model):
    """Longitudinal feedback; the slew is the jerk limit that keeps speed changes smooth."""

    gain: float = Field(default=1.0, gt=0)
    max_accel_mps2: float = Field(default=1.0, gt=0)
    max_decel_mps2: float = Field(default=2.0, gt=0)
    accel_slew_mps3: float = Field(default=1.0, gt=0)


class Settings(Model):
    # Steering-wheel degrees at 100%. 340 is the Tesla Model 3's (its EPAS faults above 360);
    # the Corolla used 180. Recalibrate per docs/steering-calibration.md.
    max_steering_angle_deg: float = Field(default=STEERING_SCALE_DEG, gt=0)
    # Per-session operator ceiling within the hard range; commands above it are rejected.
    speed_limit_mps: float = Field(default=3.0, ge=SPEED_RANGE_MPS[0], le=SPEED_RANGE_MPS[1])
    road_camera_mode: Literal["narrow_only", "wide_only", "narrow_and_wide"] = "narrow_only"
    image_adjustments: ImageSettings = Field(default_factory=ImageSettings)
    speed_control: SpeedSettings = Field(default_factory=SpeedSettings)
    objective: str = "Follow the operator-designated course and stop at the destination."


class SessionLabel(Model):
    """Operator-declared benchmark session; the model never sees it."""

    model: str = Field(min_length=1, max_length=80)
    harness: Literal["codex", "claude", "cursor", "other"]
    notes: str = Field(default="", max_length=NOTE_MAX_CHARS)


class SessionEnd(Model):
    outcome: Literal["completed", "collision", "aborted"]
    note: str = Field(default="", max_length=NOTE_MAX_CHARS)
    # One letter per attempt in the trial's chat, F (failed) or S (succeeded), e.g. "FFS".
    attempts: str = Field(default="", pattern=r"^[FS]{0,10}$")


class Request(Model):
    """Internal transport envelope. No client identity grants control authority."""

    protocol: int = PROTOCOL
    request_id: str = Field(min_length=1, max_length=128)
    operation: Literal["motion", "chunk", "stop"]
    boot_id: str
    command_epoch: int
    motion: Motion | None = None
    chunk: Chunk | None = None
    settings: Settings | None = None


INSTRUCTIONS = (
    "Use only the drivingbench_sandbox MCP. Observe the road, then use "
    "set_motion(direction, steering_percent, speed_mps, duration_s, reason). "
    f"Speed is {SPEED_RANGE_MPS[0]}–{SPEED_RANGE_MPS[1]} m/s (typically {TYPICAL_SPEED_MPS}); "
    "speeds above the operator's shared ceiling are rejected. "
    f"Duration is {DURATION_RANGE_S[0]:g}–{DURATION_RANGE_S[1]:g} s; aim for about "
    f"{TYPICAL_DURATION_S:g} s and adjust to the latency you measure from the timestamps. "
    "Direction is left, right or straight. Percent is 0–100 of the operator's shared "
    f"wheel-angle setting; zero means straight. State in reason, in {REASON_ASK}, "
    "what the command is for; it is recorded for the operator and never affects the car. "
    "Commands replace globally and start on acceptance. Motion continues while you think; "
    "expiry starts braking. "
    "Observe measured steering_percent (same scale, positive left) and movement; steering "
    "may be unavailable at standstill. "
    "Use stop_now then observe to confirm standstill. Any connected chat can continue. "
    "DrivingBench is always available; native engagement and ready vehicle inputs are required. "
    "A rejected command leaves the previous "
    "one unchanged; after an uncertain result, observe rather than blindly retrying. "
    "timestamp is UTC response time; image_age_s describes the image's age."
)
