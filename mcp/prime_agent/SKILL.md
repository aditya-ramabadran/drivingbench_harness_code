---
name: drivingbench-sandbox
description: Drive the DrivingBench car from your Python kernel. The drivingbench_sandbox tools observe(), set_motion(...) and stop_now(...) are functions you can call directly or from your own code; read() gives code the camera frames as images.
---

# drivingbench_sandbox

You drive the car from a persistent async Python kernel. The drivingbench_sandbox tools are
functions here, so besides calling them one at a time you can write and test code that uses them,
before or while driving: compute on camera frames, call the tools in loops, or start asyncio tasks
that keep running between your turns. `frames` holds the images you last observed, so you can
crop, enlarge or measure what you just saw (`attach_image` shows you a saved image), and
`history()` replays every call this session made, frames included. A stopped car stays put, so
preparing while stopped is safe; the objective says how much total time matters. The kernel and
its history persist across attempts in this session. Whatever your code sends is your command,
with the same limits and recording.

```python
import asyncio
import numpy
import drivingbench_sandbox as car

print(await car.observe())  # state as JSON; the camera images are attached to your context
print(await car.set_motion("left", 30, 1.0, 10, reason="ease left around the cone"))
print(await car.stop_now(reason="cone ahead"))

# The same observation for code: a dict and PIL images, nothing attached.
state, frames = await car.read()
# Height × width × RGB, for OpenCV, NumPy or SciPy (all installed).
pixels = numpy.asarray(frames["narrow"])
# Every call so far, oldest first: timestamp, tool, arguments, result (and frames).
for call in car.history()[-10:]:
    print(call["timestamp"], call["tool"], call["arguments"], call["result"])


async def every_second():  # any coroutine you write
    while True:
        state, frames = await car.read()
        ...
        await asyncio.sleep(1)


# Keeps running between your turns; task.cancel() stops it.
task = asyncio.create_task(every_second())
```

`help(car.set_motion)` shows each tool's full description.

Saved observations have stable identities. Keep the `state, seen = await car.read()` pair
when measuring pixels: later reads replace `car.frames`. `seen.observation_id` identifies that
snapshot, `seen.metadata("narrow")` gives raw and displayed image dimensions, and
`seen.attach("narrow")` attaches that exact saved frame. `car.observation(id)` retrieves it again.
Coordinates from an attached image use its displayed size; code sees the raw size. Timestamps
are response times, not capture times; `image_age_s` is measured at the producer response.

## Instructions

Use only the drivingbench_sandbox MCP. Observe the road, then use set_motion(direction, steering_percent, speed_mps, duration_s, reason). Speed is 0.5–3.5 m/s (typically 1.0); speeds above the operator's shared ceiling are rejected. Duration is 5–60 s: the command's lifetime, not a minimum interval before its replacement. Direction is left, right or straight. Percent is 0–100 of the operator's shared wheel-angle setting; zero means straight. reason is an optional brief annotation, recorded for the operator; it never affects the car. Commands replace globally and start on acceptance. Motion continues while you think; expiry starts braking. Observe measured steering_percent (same scale, positive left) and movement. Use stop_now then observe to confirm standstill. Any connected chat can continue. DrivingBench is always available; native engagement and ready vehicle inputs are required. A rejected command leaves the previous one unchanged; after an uncertain result, observe rather than blindly retrying. timestamp is UTC response time; image_age_s describes the image's age.
