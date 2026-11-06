"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Load the page scripts without a DOM: each returns its pure helpers on window.
function load(...names) {
  const context = { window: {} };
  for (const name of names) vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../ui", name), "utf8"), context);
  return context.window;
}

test("runner prompt adds the supervisor role and keeps the shared driving guidance", () => {
  const { DrivingBenchPrompt } = load("prompt.js");
  const saved = { objective: "Park in the blue box.", max_steering_angle_deg: 180, speed_limit_mps: 3.5 };
  const plain = DrivingBenchPrompt.buildPrompt(saved);
  const runner = DrivingBenchPrompt.buildPrompt(saved, "supervisor");
  const solo = DrivingBenchPrompt.buildPrompt(saved, "solo");
  assert.match(plain, /^Use only the `drivingbench_sandbox` MCP\./);
  assert.match(solo, /^Complete the stated objective using your tools\./);
  assert.doesNotMatch(solo, /drivingbench_sandbox|ROLE\nYou supervise/);
  assert.ok(solo.endsWith(plain.slice(plain.indexOf("OBJECTIVE"))));
  assert.match(plain, /about 25 words \(under 200 characters\)/);
  assert.match(runner, /^Complete the stated objective using your tools\./);
  assert.doesNotMatch(runner, /drivingbench_sandbox|drivingbench_driver/);
  assert.match(runner, /ROLE\nYou supervise a fast driver model that controls the car\. Use set_brief/);
  assert.ok(runner.endsWith(plain.slice(plain.indexOf("Use set_motion("))));
  assert.match(DrivingBenchPrompt.REFLECTION, /^Your previous attempt has now ended/);
  assert.match(DrivingBenchPrompt.CONTINUATION, /^You now have been granted another attempt/);
});

test("runner page enables only the controls that fit each state", () => {
  const { DrivingBenchRunner: page } = load("runner.js");
  const plain = (value) => JSON.parse(JSON.stringify(value)); // objects from another VM context
  const driver = (state, reason = null, model = "openai/gpt-6-astra") => ({ state, reason, model });
  const stopped = { running: false };
  const driving = { running: true, run: "ab12cd34", attempt: 1, between_attempts: false, driver: driver("driving"), supervisor: driver("thinking") };
  const between = { ...driving, between_attempts: true, driver: driver("stopped"), supervisor: driver("waiting") };
  const paused = { ...driving, driver: driver("paused", "supervisor_command") };
  assert.deepEqual(plain(page.controls(stopped)), { start: true, endAttempt: false, nextAttempt: false, resume: false, message: false, stop: false });
  assert.deepEqual(plain(page.controls(driving)), { start: false, endAttempt: true, nextAttempt: false, resume: false, message: true, stop: true });
  assert.deepEqual(plain(page.controls(between)), { start: false, endAttempt: false, nextAttempt: true, resume: false, message: true, stop: true });
  assert.equal(page.controls(paused).resume, true);
  const driverOnly = { ...driving, supervisor: driver("stopped", null, "") }; // a driver alone: nobody to message
  assert.equal(page.controls(driverOnly).message, false);
  assert.equal(page.controls(driverOnly).endAttempt, true);
  assert.equal(page.stateWords(paused), "run ab12cd34 · attempt 1 · driver paused (supervisor_command) · supervisor thinking");
  assert.equal(page.stateWords(between), "run ab12cd34 · attempt 1 ended · driver stopped · supervisor waiting");
  const decision = { timestamp: "2026-10-11T15:00:01.250+00:00", text: "left_60 71% · slow 80%", latency_ms: 180, result: "accepted" };
  assert.equal(page.describe(decision), "15:00:01 left_60 71% · slow 80% · 180 ms · accepted");
});

test("runner fields follow the driver: a trained one takes a chat model only when steerable", () => {
  const { DrivingBenchRunner: page } = load("runner.js");
  const options = { decision_drivers: ["typesafe/jev-1.14"], trained: { fixed: { steerable: false, objective: "Follow the cones." }, steer: { steerable: true, objective: "" } } };
  const fields = (driver, supervisor) => JSON.parse(JSON.stringify(page.fieldsFor(driver, supervisor, options)));
  assert.deepEqual(fields("fixed", "none"), { mode: "Trained driver alone: it follows the objective, the instruction it was trained on.", trained: true, supervisor: false, tier: false, brief: false, prompt: false });
  assert.equal(fields("steer", "none").supervisor, true);
  assert.match(fields("steer", "anthropic/claude-opus-5.5").mode, /chooses the trained driver's commands/);
  assert.deepEqual(fields("typesafe/jev-1.14", "openai/gpt-6-astra"), { mode: "The chat model writes briefs; the fast decision model steers.", trained: false, supervisor: true, tier: true, brief: true, prompt: true });
  assert.equal(fields("none", "gemini/gemini-4-argon").mode, "The chat model drives alone with the public tools.");
});

test("Google models are offered by exact ID, once each, and a custom ID drops a models/ prefix", () => {
  const { DrivingBenchRunner: page } = load("runner.js");
  const choices = JSON.parse(JSON.stringify(page.googleChoices(["gemini/gemini-4-argon", "gemini/gemini-3.8-flash", "gemini/gemini-4-argon"])));
  assert.deepEqual(choices, [
    { value: "gemini/gemini-4-argon", textContent: "gemini-4-argon · Google direct" },
    { value: "gemini/gemini-3.8-flash", textContent: "gemini-3.8-flash · Google direct" },
  ]);
  assert.equal(page.chatModel("gemini/custom", " models/gemini-5-preview "), "gemini/gemini-5-preview");
  assert.equal(page.chatModel("google/gemini-4-argon", "unused"), "google/gemini-4-argon");
});
