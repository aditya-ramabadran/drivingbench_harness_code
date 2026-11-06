"use strict";

// The producer owns state. This page keeps only unsaved field edits and requests.
window.DrivingBenchUI = (() => {
  const node = (id) => document.getElementById(id);
  const text = (value) => String(value ?? "Unknown").replaceAll("_", " ");
  const imageKeys = ["exposure_ev", "contrast", "saturation"];
  const textKeys = ["road_camera_mode", "objective"];
  const settingKeys = ["road_camera_mode", ...imageKeys, "max_steering_angle_deg", "speed_limit_mps", "objective"];
  const dirty = new Set();
  const pending = new Set();
  let snapshot = null;
  let saved = {};
  let revision = null;
  let producerId = null;
  let lastStatusAt = 0;
  let polling = false;
  let checkingClients = false;
  const DEMONSTRATION = "Human demonstration";
  // Ended demonstrations on this laptop ({id, notes, seconds, outcome, ended_at}) and the last fields.
  const DEMO_LOG = "drivingbench.demonstrations";
  const DEMO_FIELDS = ["demo-lot", "demo-layout", "demo-direction", "demo-split"];
  const stored = (key, fallback) => JSON.parse(globalThis.localStorage?.getItem(key) || fallback);
  const store = (key, value) => globalThis.localStorage?.setItem(key, JSON.stringify(value));
  // Demonstrations stay in the commands' range: 3.5 m/s is 7.8 mph.
  const speedLevel = (mps) => {
    if (typeof mps !== "number") return { text: "— mph", level: "unknown" };
    const mph = mps * 2.23694;
    return { text: `${mph.toFixed(1)} mph`, level: mph > 7 ? "high" : mph > 5 ? "warn" : "ok" };
  };

  const engagementGuide = (vehicle) => vehicle === "tesla"
    ? "Tesla: after Bring online, press the gear shift stalk down once to engage. Sometimes engagement needs a brief slow manual drive first. Once engaged, you can stay stopped in Drive, release the brake when ready, and press Start attempt (or begin from your chat app)."
    : "Toyota: press SET while holding the brake, then release it when ready. Press RES if requested when a motion command starts.";
  node("vehicle-guide").value = stored("drivingbench.vehicle-guide", '"tesla"');
  function renderGuide() {
    for (const id of ["setup-engagement-reminder", "drive-engagement-reminder", "runner-engagement-reminder"]) node(id).textContent = engagementGuide(node("vehicle-guide").value);
  }
  node("vehicle-guide").addEventListener("change", () => {
    store("drivingbench.vehicle-guide", node("vehicle-guide").value);
    renderGuide();
    renderStatus(snapshot);
  });
  renderGuide();
  async function request(path, method = "GET", body) {
    const options = { method, cache: "no-store" };
    if (method !== "GET") {
      options.headers = { "Content-Type": "application/json", "X-Request-ID": crypto.randomUUID() };
      options.body = JSON.stringify(body || {});
    }
    const response = await fetch(`/api/${path}`, options);
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(text(data.error || data.reason || data.detail || `HTTP ${response.status}`));
    return data;
  }
  function notice(message, error = false) {
    node("notice").textContent = message;
    node("notice").classList.toggle("error", error);
  }
  function result(message, state) {
    node("operation-result").hidden = false;
    node("operation-result").textContent = message;
    node("operation-result").className = `operation-result ${state}`;
  }
  function preset() {
    node("image-preset").value = Object.entries(window.DrivingBenchPrompt.IMAGE_PRESETS)
      .find(([, values]) => imageKeys.every((key) => Number(node(key).value) === values[key]))?.[0] || "custom";
  }
  function controls() {
    const available = Boolean(snapshot?.settings);
    for (const key of [...settingKeys, "image-preset"]) node(key).disabled = !available;
    const unchanged = !available || pending.has("settings") || !settingKeys.some((key) => dirty.has(key));
    node("save-settings").disabled = unchanged;
    node("reset-settings").disabled = unchanged;
    for (const id of ["online", "drive-online"]) node(id).disabled = pending.has("online");
  }
  function readouts() {
    for (const key of imageKeys) node(`${key}-value`).textContent = node(key).value;
  }
  // Image-adjustment preview: the last narrow frame (already carrying the saved adjustments)
  // with the difference to the slider values applied as CSS filters. Hides after 10 s idle.
  let previewTimer = null;
  let previewUrl = null;
  async function preview() {
    const figure = node("adjust-preview");
    const image = node("adjust-preview-image");
    const current = saved.image_adjustments || { exposure_ev: 0, contrast: 1, saturation: 1 };
    const chosen = Object.fromEntries(imageKeys.map((key) => [key, Number(node(key).value)]));
    image.style.filter = `brightness(${2 ** (chosen.exposure_ev - current.exposure_ev)}) contrast(${chosen.contrast / current.contrast}) saturate(${chosen.saturation / current.saturation})`;
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => { figure.hidden = true; }, 10000);
    if (figure.hidden) {
      figure.hidden = false;
      try {
        const response = await fetch("/api/camera/narrow", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const url = URL.createObjectURL(await response.blob());
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = url;
        image.src = url;
      } catch (_) { node("adjust-preview").querySelector("figcaption").textContent = "No narrow frame available to preview yet."; }
    }
  }
  function renderSettings(settings, nextRevision) {
    if (nextRevision != null && revision != null && nextRevision < revision) return;
    saved = settings;
    revision = nextRevision;
    for (const key of settingKeys) {
      if (!dirty.has(key)) node(key).value = imageKeys.includes(key) ? saved.image_adjustments?.[key] ?? "" : saved[key] ?? "";
    }
    // Always the saved settings, so a copied prompt never carries unsaved or stale values.
    node("prompt").value = window.DrivingBenchPrompt.buildPrompt(saved);
    readouts();
    preset();
    node("settings-revision").textContent = `Saved revision ${revision ?? "unknown"}. Unsaved edits stay on this laptop.`;
  }
  function renderStatus(next) {
    if (next?.producer_id && next.producer_id !== producerId) {
      producerId = next.producer_id;
      revision = null; // Revisions belong to one producer; a restart may restore an earlier one.
    }
    snapshot = next;
    if (next?.settings) renderSettings(next.settings, next.settings_revision);
    node("engagement").textContent = next?.enabled === true ? "Engaged" : next?.enabled === false ? "Disengaged" : "Unknown";
    node("summary").textContent = next ? `Comma connected · ${text(next.state)}` : "Connection unavailable; displayed settings may be old.";
    const checks = [[next?.gear && next.gear !== "unknown" ? next.gear === "drive" : null, "Car in Drive"], [typeof next?.brake_pressed === "boolean" ? !next.brake_pressed : null, "Brake released"], [next?.sensors_valid, "Native motion sensors ready"]];
    const list = node("readiness-checklist");
    list.replaceChildren();
    for (const [ok, label] of checks) {
      const li = document.createElement("li");
      li.textContent = `${ok === true ? "✓" : ok === false ? "○" : "?"} ${label}`;
      list.append(li);
    }
    node("readiness-instruction").textContent = !next ? "Restore the connection to read the car's state."
      : next.enabled === false ? engagementGuide(node("vehicle-guide").value)
      : next.brake_pressed ? "Release the brake when ready, then send a fresh motion command."
      : next.reason ? text(next.reason)
      : node("vehicle-guide").value === "tesla" ? "Engaged. You can start while stopped in Drive once ready." : "Ready for a chat. Press RES if the car requests it when a motion command starts.";
    renderSession(next?.session);
    renderDemonstration();
    window.dispatchEvent(new CustomEvent("drivingbench-status", { detail: next }));
    controls();
  }
  async function refresh() {
    if (polling) return;
    polling = true;
    try {
      const status = await request("status");
      lastStatusAt = performance.now();
      renderStatus(status);
      notice(status.recording_error ? `Connected. Recording issue: ${text(status.recording_error)}.` : "Shared comma status connected. Any connected chat can use the tools.", Boolean(status.recording_error));
    } catch (error) {
      renderStatus(null);
      notice(error.message, true);
    } finally { polling = false; }
  }
  async function clients() {
    if (checkingClients) return;
    checkingClients = true;
    const list = node("client-status");
    try {
      const data = await request("install/status");
      list.replaceChildren();
      for (const [name, value] of Object.entries(data.clients || {})) {
        const li = document.createElement("li");
        li.textContent = `${name}: ${text(value.detail || value.status || value.state || (value.installed ? "installed" : "not installed"))}`;
        list.append(li);
      }
      if (!list.children.length) list.textContent = "No client installation reported. Follow the repository's install guide.";
    } catch (error) { list.textContent = `Client check unavailable: ${error.message}`; }
    finally { checkingClients = false; }
  }
  async function act(operation, label) {
    if (pending.has(operation)) return;
    pending.add(operation);
    controls();
    result(`${label}…`, "pending");
    try {
      const data = await request(operation, "POST");
      const rejected = data.status === "rejected" || data.accepted === false;
      result(`${label}: ${text(data.reason || data.status || "request received")}. Read the live state for confirmation.`, rejected ? "error" : "ok");
    } catch (error) { result(`${label}: ${error.message}. Check the live state; the request is not retried automatically.`, "unknown"); }
    finally { pending.delete(operation); controls(); }
    await refresh();
    if (operation === "online") await clients();
  }
  async function save() {
    if (pending.has("settings")) return;
    const targetProducer = producerId;
    const values = Object.fromEntries(settingKeys.filter((key) => dirty.has(key)).map((key) => [key, node(key).value]));
    const patch = {};
    for (const [key, value] of Object.entries(values)) {
      const number = !textKeys.includes(key);
      if (number && (!value.trim() || !Number.isFinite(Number(value)) || !node(key).checkValidity())) {
        result(`Enter a valid ${text(key)}.`, "error");
        return;
      }
      if (imageKeys.includes(key)) (patch.image_adjustments ||= {})[key] = Number(value);
      else patch[key] = number ? Number(value) : value;
    }
    pending.add("settings");
    controls();
    try {
      const updated = await request("settings", "PATCH", patch);
      if (targetProducer !== producerId) throw new Error("Producer changed during save; check the shared values before saving again");
      for (const [key, value] of Object.entries(values)) if (node(key).value === value) dirty.delete(key);
      if (updated.settings) renderSettings(updated.settings, updated.settings_revision);
      result("Settings saved and shared.", "ok");
    } catch (error) { result(`Save failed: ${error.message}. Your edits are still here.`, "error"); }
    finally { pending.delete("settings"); controls(); }
    await refresh();
  }
  for (const key of settingKeys) node(key).addEventListener("input", () => {
    dirty.add(key);
    if (imageKeys.includes(key)) { readouts(); preset(); preview(); }
    controls();
  });
  node("image-preset").addEventListener("change", () => {
    const values = window.DrivingBenchPrompt.IMAGE_PRESETS[node("image-preset").value];
    for (const [key, value] of Object.entries(values || {})) { node(key).value = value; dirty.add(key); }
    readouts();
    preview();
    controls();
  });
  async function copy(value, label) {
    try { await navigator.clipboard.writeText(value); notice(`${label} copied.`); return true; }
    catch (_) { notice(`Clipboard unavailable. Copy the ${label.toLowerCase()} from the Setup tab.`, true); return false; }
  }
  // A trial is one labeled session (operator-only grouping of segments; the model never sees it).
  function renderSession(session) {
    const available = Boolean(snapshot);
    node("session-form").hidden = Boolean(session);
    node("session-open").hidden = !session;
    node("session-start").disabled = !available || pending.has("session");
    node("session-end").disabled = !available || pending.has("session");
    if (session) node("session-current").textContent = `${session.id} · ${session.model} on ${session.harness}${session.notes ? ` · ${session.notes}` : ""} · ${session.segments.length} segment${session.segments.length === 1 ? "" : "s"}`;
  }
  // A demonstration is driven by hand: live speed and its checklist while it records.
  // Its notes are built from the fields; the pass number counts this laptop's earlier passes.
  function demoNotes() {
    const [lot, layout, direction] = DEMO_FIELDS.slice(0, 3).map((id) => node(id).value.trim());
    if (!lot || !layout || !direction) return "";
    const prefix = `lot ${lot} · layout ${layout} · ${direction}`;
    const pass = stored(DEMO_LOG, "[]").filter((d) => d.notes.startsWith(`${prefix} · pass `)).length + 1;
    return `${prefix} · pass ${pass} · ${node("demo-split").value}`;
  }
  function demoTally() {
    const today = new Date().toISOString().slice(0, 10);
    const passes = stored(DEMO_LOG, "[]").filter((d) => d.ended_at.slice(0, 10) === today);
    const kept = passes.filter((d) => d.outcome === "completed");
    const minutes = (list) => (list.reduce((sum, d) => sum + d.seconds, 0) / 60).toFixed(1);
    const val = kept.filter((d) => /\bval\b/.test(d.notes));
    const dropped = passes.length - kept.length;
    return `Today on this laptop: ${kept.length} completed pass${kept.length === 1 ? "" : "es"}, ${minutes(kept)} min (val ${minutes(val)})${dropped ? `; ${dropped} not completed` : ""}`;
  }
  function renderDemonstration() {
    const chosen = node("session-model").value === DEMONSTRATION;
    const session = snapshot?.session;
    const demonstrating = session ? String(session.model).startsWith(DEMONSTRATION) : chosen;
    node("demo-guide").hidden = !demonstrating;
    node("session-start").textContent = demonstrating ? "Start demonstration" : "Start trial and copy prompt";
    node("session-end").textContent = demonstrating ? "End demonstration" : "End trial";
    node("session-copy-row").hidden = node("session-attempts-label").hidden = demonstrating;
    node("demo-fields").hidden = !chosen;
    node("session-notes-label").hidden = chosen;
    for (const id of ["session-tier-label", "session-harness-label"]) node(id).hidden = chosen;
    const next = chosen && demoNotes();
    node("demo-next").hidden = !next;
    node("demo-next").textContent = next ? `Next: ${next}` : "";
    const speed = speedLevel(snapshot?.speed_mps);
    node("demo-speed").textContent = speed.text;
    node("demo-speed").dataset.level = speed.level;
    const started = session && Date.parse(session.started_at);
    const seconds = started ? Math.max(0, Math.round((Date.now() - started) / 1000)) : null;
    node("demo-elapsed").textContent = seconds === null ? "" : `recording ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    node("demo-tally").textContent = demonstrating ? demoTally() : "";
  }
  async function sessionAction(action) {
    if (pending.has("session")) return;
    // The inference tier is folded into the model label, e.g. "GPT-6 Astra (ultrafast)", until the session contract carries it.
    const model = (node("session-model").value === "other" ? node("session-model-other").value : node("session-model").value).trim();
    const attempts = node("session-attempts").value.trim().toUpperCase();
    const demonstration = model === DEMONSTRATION;
    const notes = demonstration ? demoNotes() : node("session-notes").value.trim();
    const body = action === "start"
      ? { model: demonstration ? model : model && `${model} (${node("session-tier").value})`, harness: demonstration ? "other" : node("session-harness").value, notes }
      : { outcome: node("session-outcome").value, note: node("session-note").value.trim(), attempts };
    if (action === "start" && !model) { result("Enter the model name to start a trial.", "error"); return; }
    if (action === "start" && demonstration && !notes) { result("Name the lot, layout and direction of this pass.", "error"); return; }
    if (action === "end" && !/^[FS]{0,10}$/.test(attempts)) { result("Attempts are one F or S per attempt, like FFS.", "error"); return; }
    pending.add("session");
    renderSession(snapshot?.session);
    try {
      const data = await request(`session/${action}`, "POST", body);
      if (action === "start" && demonstration) {
        store("drivingbench.demo-fields", DEMO_FIELDS.map((id) => node(id).value));
        result(`Demonstration ${data.id} started (${notes}): drive the pass by hand, then end it.`, "ok");
      } else if (action === "end" && String(data.model).startsWith(DEMONSTRATION)) {
        const seconds = data.started_at ? (Date.parse(data.ended_at) - Date.parse(data.started_at)) / 1000 : 0;
        const ended = { id: data.id, notes: data.notes, seconds, outcome: data.outcome, ended_at: data.ended_at || new Date().toISOString() };
        store(DEMO_LOG, [...stored(DEMO_LOG, "[]"), ended]);
        result(`Demonstration ${data.id} ended: ${text(data.outcome)}, ${Math.round(seconds)} s.`, "ok");
        node("session-note").value = "";
      } else if (action === "start") {
        const copied = await copy(node("prompt").value, "Prompt");
        result(`Trial ${data.id} started${copied ? "; prompt copied for a fresh chat" : ""}.`, "ok");
      } else {
        result(`Trial ${data.id} ended: ${text(data.outcome)}${data.attempts ? ` (${data.attempts})` : ""}.`, "ok");
        for (const id of ["session-note", "session-attempts"]) node(id).value = "";
      }
    } catch (error) { result(`Trial ${action} failed: ${error.message}`, "error"); }
    finally { pending.delete("session"); }
    await refresh();
  }
  node("session-start").addEventListener("click", () => sessionAction("start"));
  node("session-model").addEventListener("change", () => {
    node("session-model-other-label").hidden = node("session-model").value !== "other";
    renderDemonstration();
  });
  node("session-end").addEventListener("click", () => sessionAction("end"));
  stored("drivingbench.demo-fields", "[]").forEach((value, i) => { if (value) node(DEMO_FIELDS[i]).value = value; });
  for (const id of DEMO_FIELDS) node(id).addEventListener(id === "demo-split" ? "change" : "input", renderDemonstration);
  const { REFLECTION, CONTINUATION } = window.DrivingBenchPrompt;
  node("reflection-text").textContent = REFLECTION;
  node("continuation-text").textContent = CONTINUATION;
  async function markAttempt(action, prompt, label) {
    const button = node(action === "end" ? "copy-reflection" : "copy-continuation");
    if (button.disabled) return;
    button.disabled = true;
    try {
      const data = await request("evaluations/attempt", "POST", { action, outcome: node("attempt-outcome").value, note: node("attempt-note").value.trim() });
      const copied = await copy(prompt, label);
      node("attempt-marker-status").textContent = `Attempt ${data.attempt} ${action === "end" ? "ended" : "started"}. ${copied ? label + " copied; paste it into your chat." : "Copy the prompt from Setup."}`;
      if (action === "start") { node("attempt-outcome").value = "unknown"; node("attempt-note").value = ""; }
    } catch (error) { node("attempt-marker-status").textContent = error.message; }
    finally { button.disabled = false; }
  }
  node("copy-prompt").addEventListener("click", () => copy(node("prompt").value, "Prompt"));
  node("copy-reflection").addEventListener("click", () => markAttempt("end", REFLECTION, "Reflection"));
  node("copy-continuation").addEventListener("click", () => markAttempt("start", CONTINUATION, "Continuation"));
  node("save-settings").addEventListener("click", save);
  node("reset-settings").addEventListener("click", () => {
    settingKeys.forEach((key) => dirty.delete(key));
    renderSettings(saved, revision);
    controls();
  });
  for (const id of ["online", "drive-online"])
    node(id).addEventListener("click", () => act("online", "Bring online"));
  node("stop").addEventListener("click", () => act("stop", "Stop"));
  node("refresh-status").addEventListener("click", () => { refresh(); clients(); });
  // Evaluation labels live on this laptop and are recorded with each new trial.
  const labels = ["course", "operator"];
  for (const key of [...labels.map((k) => `evaluation-${k}`), "evaluation-save"]) node(key).disabled = true;
  request("evaluations/context").then((context) => {
    for (const key of labels) { node(`evaluation-${key}`).value = context[key] ?? ""; node(`evaluation-${key}`).disabled = false; }
    node("evaluation-save").disabled = false;
  }).catch((error) => { node("evaluation-result").textContent = error.message; });
  node("evaluation-save").addEventListener("click", async () => {
    node("evaluation-save").disabled = true;
    try {
      await request("evaluations/context", "PATCH", Object.fromEntries(labels.map((key) => [key, node(`evaluation-${key}`).value.trim()])));
      node("evaluation-result").textContent = "Saved. New trials record these labels; existing trials keep theirs.";
    } catch (error) { node("evaluation-result").textContent = error.message; }
    finally { node("evaluation-save").disabled = false; }
  });
  node("doctor-run").addEventListener("click", async () => {
    node("doctor-run").disabled = true;
    node("doctor-results").textContent = "Checking this laptop…";
    try {
      const list = document.createElement("ul");
      for (const check of (await request("doctor")).checks) {
        const item = document.createElement("li");
        const detail = typeof check.detail === "string" ? check.detail : JSON.stringify(check.detail);
        item.textContent = `${check.name}: ${check.status} · ${detail}${check.status !== "ok" && check.action ? " · " + check.action : ""}`;
        item.className = `diagnostic-${check.status}`;
        list.append(item);
      }
      node("doctor-results").replaceChildren(list);
    } catch (error) { node("doctor-results").textContent = error.message; }
    finally { node("doctor-run").disabled = false; }
  });
  setInterval(() => {
    node("last-seen").textContent = lastStatusAt ? `Status received ${((performance.now() - lastStatusAt) / 1000).toFixed(1)} s ago` : "No status received";
  }, 500);
  setInterval(refresh, 1000);
  refresh();
  clients();
  return { node, text, request, refresh, speedLevel };
})();
