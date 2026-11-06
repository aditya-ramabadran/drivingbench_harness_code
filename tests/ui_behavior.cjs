// Browser-free behavioral checks for shared edits, camera retention and stop independence.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.join(__dirname, "..", "ui");

class Element {
  constructor(id = "") {
    this.id = id; this.value = ""; this.hidden = false; this.disabled = false;
    this.children = []; this.listeners = {}; this.dataset = {}; this.attributes = {}; this.style = {};
    this.classList = { toggle() {} };
  }
  set value(value) { this._value = String(value); }
  get value() { return this._value; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  async fire(name, event = {}) { await Promise.all((this.listeners[name] || []).map((fn) => fn(event))); }
  append(...values) { this.children.push(...values); }
  replaceChildren(...values) { this.children = values; }
  setAttribute(key, value) { this.attributes[key] = value; }
  querySelector() { return new Element(); }
  focus() {}
  checkValidity() { return true; }
}
const settings = {
  max_steering_angle_deg: 180, speed_limit_mps: 3.5,
  road_camera_mode: "narrow_only", image_adjustments: { exposure_ev: 0, contrast: 1, saturation: 1 },
  objective: "Turn left and stop",
};
const flush = async () => { for (let i = 0; i < 5; i++) await new Promise(setImmediate); };

async function browser() {
  const elements = Object.fromEntries([...fs.readFileSync(path.join(root, "index.html"), "utf8").matchAll(/id="([^"]+)"/g)]
    .map((match) => [match[1], new Element(match[1])]));
  for (const name of ["drive-panel", "drive-narrow-image", "drive-wide-image"]) elements[name].hidden = true;
  const intervals = [], timeouts = [], requests = [], events = {}, revoked = [], copied = [];
  let clock = 1000, sequence = 0;
  let status = { producer_id: "producer-one", state: "held", held: true, enabled: true, gear: "drive", brake_pressed: false,
    sensors_valid: true, speed_mps: 0, steering_deg: 17, settings: structuredClone(settings), settings_revision: 1 };
  let cameraFailure = false, decodeFailure = false, cameraWait = null, statusFailure = false;
  let recordingError = null;
  let labels = { course: "lot-a-layout-2", operator: "" };
  let visibility = "visible";
  const documentEvents = {};
  let historyCalls = [{ id: "1", timestamp: "2026-09-16T06:11:03.307900+00:00", tool: "observe", client: "Claude", arguments: {}, outcome: { state: "held", speed_mps: 0, steering_percent: 0, remaining_s: 0, image_age_s: 0.31 }, images: [{ camera: "narrow", url: `/api/recordings/${"a".repeat(32)}.jpg` }, { url: "/api/arm" }] }];
  const response = (body, extra = {}) => ({ ok: true, json: async () => structuredClone(body), ...extra });
  const context = {
    URLSearchParams, location: { search: "" },
    console, CustomEvent: class { constructor(type, options) { this.type = type; Object.assign(this, options); } },
    document: {
      getElementById: (id) => { assert.ok(elements[id], `Unknown UI id ${id}`); return elements[id]; },
      createElement: () => new Element(),
      get visibilityState() { return visibility; },
      addEventListener: (name, fn) => (documentEvents[name] ||= []).push(fn),
    },
    performance: { now: () => clock }, crypto: { randomUUID: () => `request-${++sequence}` },
    navigator: { clipboard: { writeText: async (value) => { copied.push(value); } } },
    localStorage: { items: {}, getItem(key) { return this.items[key] ?? null; }, setItem(key, value) { this.items[key] = String(value); } },
    URL: { createObjectURL: () => `blob:${++sequence}`, revokeObjectURL: (url) => revoked.push(url) },
    Image: class extends Element { async decode() { if (decodeFailure) throw Error("decode failed"); } },
    setInterval: (fn, ms) => intervals.push({ fn, ms }),
    setTimeout: (fn, ms) => (timeouts.push({ fn, ms }), timeouts.length), clearTimeout: () => {},
    fetch: async (url, options = {}) => {
      requests.push({ url, ...options });
      if (url.startsWith("/api/camera")) {
        if (cameraWait) await cameraWait;
        if (cameraFailure) throw Error("stream missing");
        return response({}, { headers: { get: (key) => ({ "X-Frame-ID": "same-frame", "X-Image-Age-S": "0.3" })[key] ?? null }, blob: async () => ({}) });
      }
      if (url === "/api/status") { if (statusFailure) throw Error("disconnected"); return response(status); }
      if (url === "/api/install/status") return response({ clients: { codex: { installed: true } } });
      if (url === "/api/evaluations/context") {
        if (options.method === "PATCH") labels = JSON.parse(options.body);
        return response(labels);
      }
      if (url === "/api/doctor") return response({ checks: [{ name: "uv", status: "ok", detail: "/usr/bin/uv", action: "Install uv." }, { name: "drivingbench-gemini", status: "warning", detail: { present: false }, action: "Add the Keychain item." }], errors: 0 });
      if (url === "/api/history") return response({ recording_error: recordingError, calls: historyCalls });
      if (url === "/api/settings") {
        const patch = JSON.parse(options.body);
        status.settings = { ...status.settings, ...patch, image_adjustments: { ...status.settings.image_adjustments, ...patch.image_adjustments } };
        status.settings_revision++;
        return response({ settings: status.settings, settings_revision: status.settings_revision });
      }
      if (["/api/stop", "/api/online"].includes(url)) return response({ status: "accepted" });
      if (url === "/api/evaluations/attempt") return response({ attempt: JSON.parse(options.body).action === "start" ? 2 : 1 });
      if (url === "/api/session/start") { status.session = { id: "2026-09-16-gpt-5-codex-1a2b3c", ...JSON.parse(options.body), segments: [], started_at: new Date(Date.now() - 42000).toISOString() }; return response(status.session); }
      if (url === "/api/session/end") { const ended = { ...status.session, ...JSON.parse(options.body), ended_at: new Date().toISOString() }; status.session = null; return response(ended); }
      throw Error(`Unexpected URL ${url}`);
    },
    addEventListener: (name, fn) => (events[name] ||= []).push(fn),
    dispatchEvent: (event) => (events[event.type] || []).forEach((fn) => fn(event)),
  };
  Element.prototype.replaceWith = function (next) { elements[this.id] = next; };
  context.window = context;
  vm.createContext(context);
  for (const file of ["prompt.js", "app.js", "workspace.js", "drive.js"]) vm.runInContext(fs.readFileSync(path.join(root, file), "utf8"), context, { filename: file });
  await flush();
  return {
    elements, requests, revoked, context, timeouts, copied,
    status: (next) => { status = { ...status, ...next }; },
    failure: (value) => { cameraFailure = value; },
    decoding: (value) => { decodeFailure = value; },
    disconnect: (value) => { statusFailure = value; },
    cameraWait: (value) => { cameraWait = value; },
    recordingError: (value) => { recordingError = value; },
    history: (calls) => { historyCalls = calls; },
    visibility: async (value) => { visibility = value; (documentEvents.visibilitychange || []).forEach((fn) => fn()); await flush(); },
    advance: (ms) => { clock += ms; },
    tick: async (ms) => { intervals.filter((timer) => timer.ms === ms).forEach((timer) => timer.fn()); await flush(); },
  };
}

(async () => {
  const b = await browser(), e = b.elements;
  assert.equal(e["max_steering_angle_deg"].value, "180");
  assert.match(e["client-status"].children[0].textContent, /installed/);

  // Evaluation labels load once, save trimmed, and never touch the shared car settings.
  assert.equal(e["evaluation-course"].value, "lot-a-layout-2");
  assert.equal(e["evaluation-save"].disabled, false);
  e["evaluation-operator"].value = "  AR  "; await e["evaluation-save"].fire("click");
  assert.deepEqual(JSON.parse(b.requests.findLast((r) => r.url === "/api/evaluations/context").body), { course: "lot-a-layout-2", operator: "AR" });
  assert.match(e["evaluation-result"].textContent, /^Saved/);
  assert.equal(b.requests.filter((r) => r.url === "/api/settings").length, 0);
  await e["doctor-run"].fire("click");
  const checks = e["doctor-results"].children[0].children;
  assert.deepEqual(checks.map((c) => c.className), ["diagnostic-ok", "diagnostic-warning"]);
  assert.equal(checks[0].textContent, "uv: ok · /usr/bin/uv"); // actions only for checks that need one
  assert.equal(checks[1].textContent, 'drivingbench-gemini: warning · {"present":false} · Add the Keychain item.');
  assert.match(e.prompt.value, /^OBJECTIVE\nTurn left and stop$/m);
  e.objective.value = "My unsaved objective"; await e.objective.fire("input");
  b.status({ settings: { ...settings, objective: "Another laptop's objective", road_camera_mode: "wide_only" }, settings_revision: 2 });
  await b.context.DrivingBenchUI.refresh();
  assert.equal(e.objective.value, "My unsaved objective");
  assert.equal(e.road_camera_mode.value, "wide_only");
  // The prompt follows saved settings only: never this laptop's unsaved draft.
  assert.match(e.prompt.value, /^OBJECTIVE\nAnother laptop's objective$/m);
  assert.equal(e["save-settings"].disabled, false);
  await e["save-settings"].fire("click");
  assert.deepEqual(JSON.parse(b.requests.findLast((r) => r.url === "/api/settings").body), { objective: "My unsaved objective" });
  assert.match(e.prompt.value, /^OBJECTIVE\nMy unsaved objective$/m);
  assert.equal(e["save-settings"].disabled, true);
  assert.equal(e.road_camera_mode.value, "wide_only");

  e.contrast.value = "1.4"; await e.contrast.fire("input");
  await e["save-settings"].fire("click");
  assert.deepEqual(JSON.parse(b.requests.findLast((r) => r.url === "/api/settings").body), { image_adjustments: { contrast: 1.4 } });
  e["image-preset"].value = "night"; await e["image-preset"].fire("change");
  await e["save-settings"].fire("click");
  assert.deepEqual(JSON.parse(b.requests.findLast((r) => r.url === "/api/settings").body), { image_adjustments: { exposure_ev: 1.2, contrast: 1.18, saturation: 1.18 } });
  assert.match(e.prompt.value, /^Use only the `drivingbench_sandbox` MCP\./);
  assert.match(e.prompt.value, /180 steering-wheel degrees; observe reports steering_percent/);
  assert.match(e.prompt.value, /steering_percent/);
  assert.match(e.prompt.value, /The speed ceiling is 3.5 m\/s\./);
  assert.match(e.prompt.value, /^OBJECTIVE\nMy unsaved objective$/m);
  assert.match(e.prompt.value, /100% is close to a U-turn, very roughly a 5\.5 m radius, so a 90 degree turn at 1 m\/s takes about 9 seconds once moving\. Roughly, 60% is a 10 m radius/);
  assert.match(e["reflection-text"].textContent, /^Your previous attempt has now ended/);
  assert.equal(e["continuation-text"].textContent, "You now have been granted another attempt to complete the same objective as before. Begin now.");
  assert.match(e.prompt.value, /- Use the least steering that makes the turn\. Most turns on a cone course need well under 100%/);
  assert.match(e.prompt.value, /add steering only if it is not turning enough\.$/);
  assert.doesNotMatch(e.prompt.value, /afraid|aggressive|\{30%, 60%, 100%\}/);
  assert.doesNotMatch(e.prompt.value, /curvature|course map|end_session|Drive:|Model client|Profile:|cadence|Model inputs/);

  // Tesla is the guidance default; changing reminders must not change car settings.
  assert.equal(e["vehicle-guide"].value, "tesla");
  assert.match(e["setup-engagement-reminder"].textContent, /stalk down once/);
  assert.equal(e["drive-engagement-reminder"].textContent, e["setup-engagement-reminder"].textContent);
  assert.match(e["runner-engagement-reminder"].textContent, /stay stopped in Drive/);
  const settingsWrites = b.requests.filter((r) => r.url === "/api/settings").length;
  e["vehicle-guide"].value = "toyota"; await e["vehicle-guide"].fire("change");
  assert.match(e["runner-engagement-reminder"].textContent, /Press RES/);
  assert.match(e["drive-engagement-reminder"].textContent, /Press RES/);
  assert.equal(b.requests.filter((r) => r.url === "/api/settings").length, settingsWrites);
  e["vehicle-guide"].value = "tesla"; await e["vehicle-guide"].fire("change");

  // Setup and chat-app buttons share the same action and pending lock.
  const onlineBefore = b.requests.filter((r) => r.url === "/api/online").length;
  const connecting = e["drive-online"].fire("click");
  assert.equal(e.online.disabled, true);
  assert.equal(e["drive-online"].disabled, true);
  await e.online.fire("click");
  await connecting;
  assert.equal(b.requests.filter((r) => r.url === "/api/online").length, onlineBefore + 1);
  assert.equal(e.online.disabled, false);
  assert.equal(e["drive-online"].disabled, false);
  assert.match(e["operation-result"].textContent, /Bring online: accepted/);

  // Collapsed previews, hidden panels and a background browser tab cost no comma bandwidth.
  const reads = () => b.requests.filter((r) => r.url.startsWith("/api/camera") || r.url === "/api/history").map((r) => r.url);
  await b.tick(500);
  assert.deepEqual(reads(), []);
  await e["runner-tab"].fire("click"); await b.tick(500);
  assert.deepEqual(reads(), []);
  e["live-cameras"].open = true; await e["live-cameras"].fire("toggle"); await b.tick(500);
  assert.deepEqual(reads(), []); // an open panel still downloads nothing until switched on
  assert.match(e["drive-narrow-status"].textContent, /only while switched on/);
  e["cameras-live"].checked = true; await e["cameras-live"].fire("change"); await flush();
  assert.deepEqual(reads(), ["/api/camera/narrow", "/api/camera/wide"]); // switching on refreshes at once
  await b.tick(500);
  assert.equal(reads().filter((url) => url === "/api/history").length, 0); // history lives on the Drive tab
  await b.visibility("hidden"); const hiddenReads = reads().length; await b.tick(500);
  assert.equal(reads().length, hiddenReads);
  await b.visibility("visible");
  assert.ok(reads().length > hiddenReads); // returning to the page refreshes at once

  await e["drive-tab"].fire("click"); await b.tick(500);
  assert.ok(!reads().includes("/api/history")); // history also waits for its own switch
  e["history-live"].checked = true; await e["history-live"].fire("change"); await flush();
  assert.ok(reads().includes("/api/history"));
  // Both switches turn themselves off after 20 s; the operator switches on again to resume.
  const expiries = b.timeouts.filter((t) => t.ms === 20000);
  assert.equal(expiries.length, 2);
  expiries.forEach((t) => t.fn()); await flush();
  assert.equal(e["cameras-live"].checked, false);
  assert.equal(e["history-live"].checked, false);
  const offReads = reads().length; await b.tick(500);
  assert.equal(reads().length, offReads);
  for (const id of ["cameras-live", "history-live"]) { e[id].checked = true; await e[id].fire("change"); }
  await flush();
  assert.equal(e["drive-narrow-image"].hidden, false);
  const retained = e["drive-narrow-image"];
  b.failure(true); b.advance(5000); await b.tick(500);
  assert.equal(e["drive-narrow-image"], retained);
  assert.match(e["drive-narrow-status"].textContent, /5.3 s old.*stream missing/);
  b.failure(false); b.decoding(true); await b.tick(500);
  assert.equal(e["drive-narrow-image"], retained);
  assert.match(e["drive-narrow-status"].textContent, /decode failed/);
  b.decoding(false); await b.tick(500);
  assert.match(e["drive-narrow-status"].textContent, /5.3 s old/); // same frame never becomes younger
  assert.ok(b.revoked.length >= 2);
  await e["setup-tab"].fire("click"); await e["drive-tab"].fire("click"); await flush();
  assert.equal(e["drive-narrow-image"].hidden, false);

  b.disconnect(true); await b.context.DrivingBenchUI.refresh();
  assert.equal(e["drive-speed"].textContent, "Unknown");
  assert.equal(e["drive-narrow-image"].hidden, false);
  b.disconnect(false);
  let release;
  b.cameraWait(new Promise((resolve) => { release = resolve; }));
  await b.tick(500);
  await e.stop.fire("click");
  const stop = b.requests.findLast((r) => r.url === "/api/stop");
  assert.equal(stop.method, "POST");
  assert.ok(stop.headers["X-Request-ID"]);
  release(); await flush();
  assert.equal(e["history-calls"].children.length, 1);
  let card = e["history-calls"].children[0];
  assert.equal(card.className, "call call-observe");
  assert.match(card.children[0].innerHTML, /glyph-eye/); // observe glyph
  assert.equal(card.children[0].children[0].textContent, "observe · Claude · 06:11:03 UTC");
  assert.equal(card.children[1].textContent, "held · 0 m/s · steering 0% · 0 s remaining"); // zero is "0", not blank; no reason paragraph
  assert.equal(card.children[2].textContent, "Images 0.3 s old");
  assert.equal(card.children.length, 5); // header, params, outcome, one valid image, raw toggle
  assert.equal(card.children[3].alt, "Recorded narrow model observation");
  assert.equal(card.children[4].children[0].textContent, "Raw");
  assert.doesNotMatch(card.children[1].textContent + card.children[2].textContent, /[{}"]/);
  b.recordingError("disk full"); await b.tick(500);
  assert.match(e["history-notice"].textContent, /Recording issue: disk full/);
  assert.equal(e["history-calls"].children.length, 1);

  // Commands read as sentences with a glyph, a quoted reason, a direction letter and a steering meter.
  b.history([
    { id: "2", timestamp: "2026-09-16T06:12:00+00:00", tool: "set_motion", client: "Codex", arguments: { direction: "left", steering_percent: 50, speed_mps: 0.6, duration_s: 5, reason: "Cone gap opens to the left; ease in." }, outcome: { status: "accepted" } },
    { id: "3", timestamp: "2026-09-16T06:12:04+00:00", tool: "set_motion", client: "Cursor", arguments: { direction: "straight", steering_percent: 0, speed_mps: 3.5, duration_s: 5, reason: "" }, outcome: { error: "speed_limit_exceeded" } },
    { id: "4", timestamp: "2026-09-16T06:12:06+00:00", tool: "set_motion", client: "Codex", arguments: { direction: "right", steering_percent: 100, speed_mps: 1, duration_s: 2 }, outcome: { error: "operator_brake" } },
    { id: "5", timestamp: "2026-09-16T06:12:07+00:00", tool: "stop_now", client: "Claude", arguments: { reason: "Pedestrian ahead." }, outcome: { status: "stopping" } },
  ]);
  await b.tick(500);
  const cards = e["history-calls"].children;
  const motion = (c) => { const body = c.children.find((x) => x.className === "call-body"); return { figure: body.children[0], params: body.children[1].children[0].textContent, outcome: body.children[1].children[1].textContent }; };
  assert.deepEqual(cards.map((c) => c.className), ["call call-accepted", "call call-error", "call call-error", "call call-accepted"]);
  assert.match(cards[0].children[0].innerHTML, /glyph-stop/);
  assert.equal(cards[0].children[1].textContent, "“Pedestrian ahead.”"); // newest first, quoted reason leads
  assert.equal(cards[0].children[2].textContent, "Brake now; authorization kept");
  assert.match(cards[1].children[0].innerHTML, /glyph-go/);
  assert.equal(cards[1].children[1].className, "call-body"); // no reason: the body comes first
  assert.equal(motion(cards[1]).params, "right 100% · 1 m/s · 2 s");
  assert.equal(motion(cards[1]).figure.children[0].textContent, "R");
  assert.equal(motion(cards[1]).figure.children[1].children[0].style.width, "100%");
  assert.equal(motion(cards[1]).figure.children[1].children[0].className, "meter-fill");
  assert.equal(motion(cards[2]).params, "straight 0% · 3.5 m/s · 5 s"); // empty reason omitted
  assert.equal(motion(cards[2]).outcome, "Rejected · speed limit exceeded");
  assert.equal(cards[3].children[1].textContent, "“Cone gap opens to the left; ease in.”");
  assert.equal(motion(cards[3]).figure.children[0].textContent, "L");
  assert.equal(motion(cards[3]).figure.children[1].children[0].style.width, "50%");
  assert.equal(motion(cards[3]).params, "left 50% · 0.6 m/s · 5 s");
  assert.equal(motion(cards[3]).outcome, "Accepted");
  assert.match(cards[3].children[3].children[1].textContent, /"reason": "Cone gap/); // raw JSON only behind the toggle

  // Switching to the runner tab is visibility-only: no run/session mutation or reload.
  e["session-notes"].value = "unsaved trial label";
  const mutations = () => b.requests.filter((r) => ["POST", "PATCH"].includes(r.method)).length;
  const beforeTabs = mutations();
  await e["runner-tab"].fire("click");
  assert.equal(e["runner-panel"].hidden, false);
  assert.equal(e["drive-panel"].hidden, true);
  await e["drive-tab"].fire("click");
  assert.equal(e["session-notes"].value, "unsaved trial label");
  assert.equal(mutations(), beforeTabs);

  // Trials: start needs a model and copies the prompt; the open trial copies the canned prompts.
  assert.equal(e["session-form"].hidden, false);
  e["session-model"].value = "other"; await e["session-model"].fire("change");
  assert.equal(e["session-model-other-label"].hidden, false);
  await e["session-start"].fire("click");
  assert.match(e["operation-result"].textContent, /Enter the model name/);
  e["session-model-other"].value = " gpt-5-codex "; e["session-tier"].value = "ultrafast"; e["session-harness"].value = "codex"; e["session-notes"].value = "lot A";
  await e["session-start"].fire("click");
  const started = b.requests.findLast((r) => r.url === "/api/session/start");
  assert.deepEqual(JSON.parse(started.body), { model: "gpt-5-codex (ultrafast)", harness: "codex", notes: "lot A" });
  assert.equal(b.copied.at(-1), e.prompt.value);
  assert.match(e["operation-result"].textContent, /started; prompt copied/);
  await e["copy-reflection"].fire("click");
  assert.match(b.copied.at(-1), /^Your previous attempt has now ended/);
  await e["copy-continuation"].fire("click");
  assert.match(b.copied.at(-1), /^You now have been granted another attempt/);
  e["session-model"].value = "Claude Fable 5.1"; await e["session-model"].fire("change");
  assert.equal(e["session-model-other-label"].hidden, true);
  assert.equal(e["session-form"].hidden, true);
  assert.equal(e["session-open"].hidden, false);
  assert.match(e["session-current"].textContent, /^2026-09-16-gpt-5-codex-1a2b3c · gpt-5-codex \(ultrafast\) on codex · lot A · 0 segments$/);
  e["session-outcome"].value = "collision"; e["session-note"].value = "clipped the cone";
  const ends = () => b.requests.filter((r) => r.url === "/api/session/end").length;
  e["session-attempts"].value = "FXS";
  await e["session-end"].fire("click");
  assert.equal(ends(), 0);
  assert.match(e["operation-result"].textContent, /one F or S per attempt/);
  e["session-attempts"].value = " ffs ";
  await e["session-end"].fire("click");
  assert.deepEqual(JSON.parse(b.requests.findLast((r) => r.url === "/api/session/end").body), { outcome: "collision", note: "clipped the cone", attempts: "FFS" });
  assert.equal(e["session-attempts"].value, "");
  assert.equal(e["session-form"].hidden, false);
  assert.match(e["operation-result"].textContent, /ended: collision/);

  // Slider preview appears on input with relative filters, hides on the 10 s timer.
  b.timeouts.findLast((t) => t.ms === 10000).fn(); // earlier slider edits opened it; let it time out
  assert.equal(e["adjust-preview"].hidden, true);
  e.exposure_ev.value = "1"; await e.exposure_ev.fire("input"); await flush();
  assert.equal(e["adjust-preview"].hidden, false);
  assert.equal(e["exposure_ev-value"].textContent, "1");
  // Relative to the saved night preset (1.2 EV, 1.18x, 1.18x): the frame already carries those.
  assert.equal(e["adjust-preview-image"].style.filter, `brightness(${2 ** (1 - 1.2)}) contrast(1) saturate(1)`);
  assert.ok(b.requests.some((r) => r.url === "/api/camera/narrow"));
  const hide = b.timeouts.findLast((t) => t.ms === 10000); hide.fn();
  assert.equal(e["adjust-preview"].hidden, true);
  assert.equal(e["image-preset"].value, "custom"); // shown, not selectable

  // A new producer can have a lower revision, without erasing this browser's draft.
  e.objective.value = "Keep this draft"; await e.objective.fire("input");
  b.cameraWait(new Promise((resolve) => { release = resolve; }));
  await b.tick(500);
  const previousFrame = e["drive-narrow-image"];
  b.status({ producer_id: "producer-two", settings_revision: 0, settings: { ...settings, speed_limit_mps: 2 }, enabled: undefined, gear: undefined });
  await b.context.DrivingBenchUI.refresh();
  assert.equal(e.speed_limit_mps.value, "2");
  assert.equal(e.objective.value, "Keep this draft");
  assert.equal(e.engagement.textContent, "Unknown");
  assert.match(e["drive-narrow-status"].textContent, /previous producer/);
  release(); await flush();
  assert.equal(e["drive-narrow-image"], previousFrame); // late old-source result is discarded
  b.cameraWait(null); await b.tick(500);
  assert.doesNotMatch(e["drive-narrow-status"].textContent, /previous producer/);
  assert.match(e["drive-narrow-status"].textContent, /0.3 s old/);
  b.status({ settings_revision: 0, settings: { ...settings, speed_limit_mps: 2 }, state: "executing", command: { target_angle_deg: -180, speed_mps: 0.6 }, remaining_s: 3 });
  await b.context.DrivingBenchUI.refresh();
  assert.match(e["drive-command"].textContent, /-180.00 ° target.*0.60 m\/s target.*3.00 s remaining/);
  // A demonstration: live speed against the commands' range, the checklist, notes required, no prompt.
  const { speedLevel } = b.context.DrivingBenchUI;
  assert.deepEqual(JSON.parse(JSON.stringify([0, 2.5, 3.2, undefined].map(speedLevel))), [
    { text: "0.0 mph", level: "ok" }, { text: "5.6 mph", level: "warn" },
    { text: "7.2 mph", level: "high" }, { text: "— mph", level: "unknown" },
  ]);
  assert.equal(e["demo-guide"].hidden, true);
  e["session-model"].value = "Human demonstration"; await e["session-model"].fire("change");
  assert.equal(e["demo-guide"].hidden, false);
  assert.equal(e["demo-fields"].hidden, false); assert.equal(e["session-notes-label"].hidden, true);
  assert.equal(e["session-start"].textContent, "Start demonstration");
  b.status({ speed_mps: 2.5, session: null }); await b.context.DrivingBenchUI.refresh();
  assert.equal(e["demo-speed"].textContent, "5.6 mph"); assert.equal(e["demo-speed"].dataset.level, "warn");
  const startsBefore = b.requests.filter((r) => r.url === "/api/session/start").length;
  e["demo-lot"].value = "B"; e["demo-layout"].value = ""; e["demo-direction"].value = "north";
  await e["session-start"].fire("click"); await flush();
  assert.match(e["operation-result"].textContent, /Name the lot, layout and direction/);
  assert.equal(b.requests.filter((r) => r.url === "/api/session/start").length, startsBefore);
  e["demo-layout"].value = "2"; e["demo-split"].value = "train"; await e["demo-layout"].fire("input");
  assert.equal(e["demo-next"].textContent, "Next: lot B · layout 2 · north · pass 1 · train");
  const copiedBefore = b.copied.length;
  await e["session-start"].fire("click"); await flush();
  const demonstrated = JSON.parse(b.requests.findLast((r) => r.url === "/api/session/start").body);
  assert.deepEqual(demonstrated, { model: "Human demonstration", harness: "other", notes: "lot B · layout 2 · north · pass 1 · train" });
  assert.equal(b.copied.length, copiedBefore); // nothing to paste: a person drives
  assert.match(e["operation-result"].textContent, /drive the pass by hand/);
  assert.equal(e["demo-guide"].hidden, false);
  assert.equal(e["session-copy-row"].hidden, true); assert.equal(e["session-attempts-label"].hidden, true);
  assert.equal(e["session-end"].textContent, "End demonstration");
  assert.match(e["demo-elapsed"].textContent, /^recording 0:4[2-3]$/);
  b.status({ speed_mps: 3.3 }); await b.context.DrivingBenchUI.refresh();
  assert.equal(e["demo-speed"].dataset.level, "high");
  e["session-outcome"].value = "completed"; await e["session-end"].fire("click"); await flush();
  assert.deepEqual(JSON.parse(b.requests.findLast((r) => r.url === "/api/session/end").body), { outcome: "completed", note: "", attempts: "" });
  assert.match(e["operation-result"].textContent, /ended: completed, 4[2-3] s/);
  assert.equal(e["demo-next"].textContent, "Next: lot B · layout 2 · north · pass 2 · train"); // the pass counts up
  e["demo-direction"].value = "south"; e["demo-split"].value = "val"; await e["demo-split"].fire("change");
  assert.equal(e["demo-next"].textContent, "Next: lot B · layout 2 · south · pass 1 · val");
  assert.match(e["demo-tally"].textContent, /^Today on this laptop: 1 completed pass, 0\.7 min \(val 0\.0\)$/);
  await e["session-start"].fire("click"); await flush();
  e["session-outcome"].value = "aborted"; await e["session-end"].fire("click"); await flush();
  e["session-model"].value = "Human demonstration"; await e["session-model"].fire("change");
  assert.match(e["demo-tally"].textContent, /1 completed pass, 0\.7 min \(val 0\.0\); 1 not completed$/);
  e["session-model"].value = "GPT-6 Astra"; await e["session-model"].fire("change");
  assert.equal(e["demo-fields"].hidden, true); assert.equal(e["session-notes-label"].hidden, false);
  assert.equal(e["session-start"].textContent, "Start trial and copy prompt");
  console.log("UI behavior checks passed: shared edits, prompts, retained cameras, independent stop, producer restart, demonstrations.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
