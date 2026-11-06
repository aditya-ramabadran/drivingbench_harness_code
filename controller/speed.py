"""Speed feedback: acceleration proportional to the speed error, bounded and jerk-limited."""

from drivingbench.shared.contracts import SpeedSettings


def clip(value, low, high):
    return max(low, min(high, value))


class SpeedControl:
    def __init__(self, settings: SpeedSettings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.acceleration = 0.0

    def step(self, target, measured, dt):
        if target == 0:
            self.reset()
            return 0.0
        p = self.settings
        desired = clip(p.gain * (target - measured), -p.max_decel_mps2, p.max_accel_mps2)
        delta = p.accel_slew_mps3 * dt
        self.acceleration = clip(desired, self.acceleration - delta, self.acceleration + delta)
        return self.acceleration
