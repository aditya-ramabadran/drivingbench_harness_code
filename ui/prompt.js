"use strict";

// The generated prompt: fixed wording, with the objective and shared car settings filled in. The
// supervisor variant is the same prompt for a frontier model supervising a runner driver. The
// reflection and continuation prompts follow a failed attempt in the same conversation.
window.DrivingBenchPrompt = (() => {
  const IMAGE_PRESETS = {
    neutral: { exposure_ev: 0, contrast: 1, saturation: 1 },
    night: { exposure_ev: 1.2, contrast: 1.18, saturation: 1.18 },
  };
  // role: "app" (a chat app with the MCP), "solo" (a Runner chat model calling the same tools
  // directly) or "supervisor" (a Runner chat model beside a driver).
  function buildPrompt(saved, role = "app") {
    const supervisor = role === "supervisor";
    return [
      role === "app"
        ? "Use only the `drivingbench_sandbox` MCP. Complete the stated objective using its tools."
        : "Complete the stated objective using your tools.",
      "",
      "OBJECTIVE",
      saved.objective || "No objective entered.",
      "",
      "ROLE",
      ...(supervisor
        ? ["You supervise a fast driver model that controls the car. Use set_brief to give it instructions; observe shows the road, the car, and the driver's brief and latest decisions. Use set_motion or stop_now only when you need to take over directly; the driver resumes when your command ends or you write a new brief."]
        : []),
      "Use set_motion(direction, steering_percent, speed_mps, duration_s, reason). Direction is left, right or straight; steering_percent is 0–100. Zero is straight for either turn direction.",
      `100% requests ${saved.max_steering_angle_deg} steering-wheel degrees; observe reports steering_percent on the same scale, positive left. The speed ceiling is ${saved.speed_limit_mps} m/s. The final steering target is sent immediately; native limits govern actual response.`,
      "In every set_motion and stop_now, give reason: about 25 words (under 200 characters) on what you see and what the command is for. It is recorded for the operator and never changes what the car does.",
      "",
      "",
      "DRIVING",
      "- Choose durations that allow observation and reasoning latency; the car continues moving while you think.",
      "- Observe, send a motion command, then observe actual movement and adapt.",
      "- Commands replace globally rather than queue. Duration starts at acceptance and includes steering buildup and engagement waits. Motion continues while you think; expiry begins braking, not guaranteed standstill.",
      "- A rejected replacement leaves the previous command unchanged. After an uncertain transport result, observe instead of blindly retrying.",
      "- Any connected chat can replace the active command or call stop_now. DrivingBench is always active. Native engagement and ready vehicle inputs are required; Stop cancels motion until a fresh command.",
      "- timestamp is UTC response time, not image capture or command acceptance time. image_age_s is image age at producer response.",
      "- Carefully look for objects and obstacles (parked cars, buildings, islands, trees, etc); they’re all 3D and the vehicle you are controlling is also 3D and is also wider than it might seem from the camera images. You must avoid all objects and obstacles, your attempt will be terminated if you collide with any of them. ",
      "- If possible, try to always be in the center of your lane/road and maximize distance from obstacles.",
      "- In successive observed images, pay attention to what changes, i.e. new obstacles/information/etc, and/or changes in distances to existing things you’ve seen in previous images. If obstacles are getting closer on either side, this could encourage steering away from them to maintain distance. ",
      "- Over time, understand the behavior that occurs when you choose different actions or steering inputs and adapt to this. ",
      "- Think and plan carefully where you want to end up and plan your route accordingly and advance toward subgoals. ",
      "- Your vehicle turns tightly: 100% is close to a U-turn, very roughly a 5.5 m radius, so a 90 degree turn at 1 m/s takes about 9 seconds once moving. Roughly, 60% is a 10 m radius (about 15 seconds for 90 degrees) and 30% a 20 m radius (about 30 seconds). Treat these as rough estimates and judge the actual response from what you observe. The steering wheel reaches its target within about 1.5 seconds, also at a standstill, so the turn starts as soon as the car moves.",
      "- Use the least steering that makes the turn. Most turns on a cone course need well under 100%; 100% swings the car around almost on the spot, so turning too sharply is as easy a mistake as turning too little. Start moderate, observe how the car responds, and add steering only if it is not turning enough.",
    ].join("\n");
  }
  const REFLECTION = "Your previous attempt has now ended (likely due to a mistake or collision with an object or the environment). Reflect on how you did in your previous attempt, why it might have ended, and what you can do better in a future attempt. Prepare however you like before your next attempt, but do not move the car until it begins.";
  const CONTINUATION = "You now have been granted another attempt to complete the same objective as before. Begin now.";
  return { buildPrompt, IMAGE_PRESETS, REFLECTION, CONTINUATION };
})();
