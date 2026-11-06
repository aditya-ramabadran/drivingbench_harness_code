"use strict";

// Runner tab: set up, start and steer a run attempt by attempt, and watch both layers.
// Pure helpers live on window.DrivingBenchRunner for tests; the rest polls /runner/status.
window.DrivingBenchRunner = (() => {
  const describe = (d) => `${d.timestamp.slice(11, 19)} ${d.text} · ${d.latency_ms} ms · ${d.result}`;
  const words = (part) => part.state + (part.reason ? ` (${part.reason})` : "");
  const stateWords = (s) => s.preparing ? s.preparing : s.running
    ? `run ${s.run} · attempt ${s.attempt}${s.between_attempts ? " ended" : ""} · driver ${words(s.driver)} · supervisor ${words(s.supervisor)}`
    : "stopped";
  const isGoogle = (model) => model.startsWith("gemini/");
  const googleChoices = (models) => [...new Set(models)].map((value) => ({ value, textContent: `${value.replace(/^gemini\//, "")} · Google direct` }));
  const chatModel = (selected, custom) => selected === "gemini/custom" ? `gemini/${custom.trim().replace(/^models\//, "")}` : selected;
  // Which setup fields make sense for a driver and chat model, and a sentence saying what will run.
  // A trained driver alone follows the instruction it was trained on; only a steerable one takes a
  // chat model, which chooses its commands.
  const fieldsFor = (driver, supervisor, o) => {
    const trained = (o.trained || {})[driver];
    const decision = (o.decision_drivers || []).includes(driver);
    const chat = supervisor !== "none";
    const takesChat = !trained || trained.steerable;
    const mode = trained
      ? trained.steerable
        ? chat ? "The chat model chooses the trained driver's commands; the driver steers." : "Steerable trained driver: choose a chat model to give it commands."
        : "Trained driver alone: it follows the objective, the instruction it was trained on."
      : decision ? (chat ? "The chat model writes briefs; the fast decision model steers." : "Decision driver alone: it follows the starting brief.")
      : "The chat model drives alone with the public tools.";
    return { mode, trained: Boolean(trained), supervisor: takesChat, tier: chat && takesChat, brief: decision, prompt: chat && takesChat };
  };
  // Which controls make sense in each state.
  const controls = (s) => ({
    start: !s.running && !s.preparing,
    endAttempt: s.running && !s.between_attempts,
    nextAttempt: s.running && s.between_attempts,
    resume: s.running && s.driver.state === "paused",
    message: s.running && Boolean(s.supervisor.model),
    stop: s.running,
  });

  if (typeof document === "undefined") return { describe, stateWords, controls, fieldsFor, chatModel, googleChoices };
  const node = (id) => document.getElementById(id);
  const fill = (id, values) => node(id).replaceChildren(...values.map((v) => Object.assign(document.createElement("option"), { value: v, textContent: v })));
  let settings = null;
  let options = {};
  let onlinePending = false;
  const pending = new Set();

  function result(text, kind, id = "runner-result") {
    const el = node(id);
    el.textContent = text;
    el.className = `operation-result ${kind}`;
    el.hidden = false;
  }
  async function post(path, body = {}, headers = {}) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  }
  const objective = () => node("runner-objective").value.trim();
  const model = () => chatModel(node("runner-supervisor").value, node("runner-google-model").value);
  function googleModels(models) {
    const selected = node("runner-supervisor").value;
    node("runner-google-group")?.remove();
    const group = Object.assign(document.createElement("optgroup"), { id: "runner-google-group", label: "Google direct · Gemini API" });
    group.append(...googleChoices(models).map((choice) => Object.assign(document.createElement("option"), choice)));
    group.append(Object.assign(document.createElement("option"), { value: "gemini/custom", textContent: "Other Google model…" }));
    node("runner-supervisor").append(group);
    node("runner-supervisor").value = selected;
  }
  function renderPrompt() {
    // A chat model alone gets the chat apps' prompt for its own tools; beside a driver, the supervisor variant.
    const role = node("runner-driver").value !== "none" ? "supervisor" : "solo";
    if (settings) node("runner-prompt").value = window.DrivingBenchPrompt.buildPrompt({ ...settings, objective: objective() || settings.objective }, role);
  }
  function syncTiers(running) {
    // Google's direct API has no ultrafast tier; its tiers carry Google's names.
    node("runner-tier").disabled = running;
    const google = isGoogle(node("runner-supervisor").value);
    const tiers = (options.tiers || []).filter((tier) => !google || tier !== "ultrafast");
    if (Array.from(node("runner-tier").options, (option) => option.value).join() !== tiers.join()) {
      const selected = node("runner-tier").value;
      fill("runner-tier", tiers);
      node("runner-tier").value = tiers.includes(selected) ? selected : tiers[0] || "";
    }
    for (const option of node("runner-tier").options) {
      option.textContent = google ? { normal: "Normal · Google Standard", fast: "Fast · Google Priority" }[option.value] || option.value : option.value;
    }
  }
  function list(id, items, text) {
    node(id).replaceChildren(...items.map((item) => Object.assign(document.createElement("li"), { textContent: text(item) })));
  }
  function render(s) {
    node("runner-state").textContent = stateWords(s) + (s.metadata_error ? ` · Metadata issue: ${s.metadata_error}` : "");
    const on = controls(s);
    node("runner-start").disabled = !on.start || onlinePending || pending.has("runner-start");
    node("runner-online").disabled = s.running || s.preparing || onlinePending;
    node("runner-end-attempt").disabled = !on.endAttempt;
    // Reflection and continuation go to a supervising chat model; a driver alone just stops and restarts.
    const chat = s.running ? Boolean(s.supervisor.model) : node("runner-supervisor").value !== "none";
    node("runner-end-attempt").textContent = chat ? "End attempt (send reflection)" : "End attempt (stop the car)";
    node("runner-next-attempt").textContent = chat ? "Next attempt (send continuation)" : "Next attempt (restart the driver)";
    node("runner-next-attempt").disabled = !on.nextAttempt;
    node("runner-resume").disabled = !on.resume;
    for (const id of ["runner-message", "runner-send"]) node(id).disabled = !on.message;
    for (const id of ["runner-note", "runner-stop"]) node(id).disabled = !on.stop;
    for (const id of ["runner-supervisor", "runner-google-model", "runner-google-models", "runner-google-probe", "runner-driver", "runner-objective", "runner-brief"]) node(id).disabled = s.running || Boolean(s.preparing) || pending.has("runner-start") || pending.has(id);
    syncTiers(s.running || Boolean(s.preparing));
    node("runner-revision").textContent = s.driver.revision ? `revision ${s.driver.revision}` : "";
    node("runner-current-brief").textContent = s.driver.brief || "—";
    list("runner-decisions", s.driver.decisions.slice().reverse(), describe);
    list("runner-transcript", s.supervisor.transcript.slice().reverse(), (t) => `${t.timestamp.slice(11, 19)} ${t.kind}: ${t.text}`);
    node("runner-cost").textContent = s.supervisor.model ? `${s.supervisor.model} (${s.supervisor.tier}) · ${s.supervisor.cost_usd == null ? "billing unavailable" : `$${s.supervisor.cost_usd.toFixed(2)}`}` : "";
  }
  async function refresh() {
    try { render(await (await fetch("/runner/status")).json()); } catch (_) { node("runner-state").textContent = "gateway unavailable"; node("runner-online").disabled = true; }

    if (!settings) {
      try { settings = (await (await fetch("/api/status")).json()).settings || null; if (settings && !objective()) node("runner-objective").value = settings.objective; syncFields(false); } catch (_) { /* producer offline: retry next refresh */ }
    }
  }
  function action(id, path, body, done) {
    node(id).addEventListener("click", async () => {
      if (pending.has(id)) return;
      pending.add(id);
      node(id).disabled = true;
      if (id === "runner-start") result("Preparing: checking the connection and the selected driver…", "pending");
      try {
        await post(path, body ? body() : {});
        result(done, "ok");
        if (["runner-start", "runner-next-attempt"].includes(id)) {
          node("runner-attempt-outcome").value = "unknown";
          node("runner-attempt-note").value = "";
        }
      } catch (error) { result(`${done.split(":")[0]} failed: ${error.message}`, "error"); }
      finally { pending.delete(id); }
      await refresh();
    });
  }

  node("runner-online").addEventListener("click", async () => {
    const button = node("runner-online");
    if (onlinePending || button.disabled) return;
    onlinePending = true;
    button.disabled = true;
    button.textContent = "Bringing online…";
    node("runner-start").disabled = true;
    const onlineResult = (text, kind) => result(text, kind, "runner-online-result");
    onlineResult("Bring online…", "pending");
    try {
      const data = await post("/api/online", {}, { "X-Request-ID": crypto.randomUUID() });
      if (data.error) throw new Error(data.error);
      const rejected = data.status === "rejected" || data.accepted === false;
      const state = String(data.reason || data.status || "request received").replaceAll("_", " ");
      onlineResult(`Bring online: ${state}. Read the live state for confirmation.`, rejected ? "error" : "ok");
    } catch (error) {
      onlineResult(`Bring online: ${error.message}. Check the live state; the request is not retried automatically.`, "unknown");
    } finally {
      onlinePending = false;
      button.textContent = "Bring online";
      await refresh();
    }
  });
  action("runner-start", "/runner/start", () => ({
    supervisor: model(),
    tier: node("runner-tier").value,
    driver: node("runner-driver").value,
    objective: objective(),
    brief: node("runner-brief").value.trim(),
    prompt: node("runner-prompt").value,
    reflection: window.DrivingBenchPrompt.REFLECTION,
    continuation: window.DrivingBenchPrompt.CONTINUATION,
  }), "Started: attempt 1, session labelled.");
  action("runner-end-attempt", "/runner/end-attempt", () => ({ outcome: node("runner-attempt-outcome").value, note: node("runner-attempt-note").value.trim() }), "Attempt ended: driver stopped, reflection sent.");
  action("runner-next-attempt", "/runner/next-attempt", null, "Next attempt: continuation sent, driver restarted.");
  action("runner-resume", "/runner/resume", null, "Driver resumed.");
  action("runner-send", "/runner/message", () => ({ text: node("runner-message").value.trim() }), "Message sent.");
  action("runner-stop", "/runner/stop", () => ({ outcome: node("runner-outcome").value, note: node("runner-note").value.trim(), attempt_outcome: node("runner-attempt-outcome").value, attempt_note: node("runner-attempt-note").value.trim() }), "Run stopped: car stopped, session ended.");
  node("runner-objective").addEventListener("input", renderPrompt);
  for (const [id, operation] of [["runner-google-models", "models"], ["runner-google-probe", "probe"]]) {
    node(id).addEventListener("click", async () => {
      if (pending.has(id)) return;
      pending.add(id);
      node(id).disabled = true;
      const show = (text, kind) => result(text, kind, "runner-google-result");
      show(operation === "models" ? "Reading Google's model catalog…" : "Testing an image and tool-call round trip with Google; no vehicle tools run…", "pending");
      try {
        if (operation === "models") {
          const response = await fetch("/runner/gemini/models");
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || response.statusText);
          googleModels(data.models);
          syncTiers(false);
          syncFields(false);
          show(`${data.models.length} Google model IDs found. A listing does not establish access; test a model before driving.`, "ok");
        } else {
          const data = await post("/runner/gemini/probe", { model: model(), tier: node("runner-tier").value });
          show(`${data.model}: image and tool round trip passed (${data.latency_s}s total). Requested ${data.requested_service_tier}; served tier ${data.service_tier || "not reported"}.`, "ok");
        }
      } catch (error) { show(error.message, "error"); }
      finally { pending.delete(id); await refresh(); }
    });
  }
  function syncFields(driverChanged) {
    const driver = node("runner-driver").value;
    if (!fieldsFor(driver, node("runner-supervisor").value, options).supervisor) node("runner-supervisor").value = "none";
    const trainedObjective = (options.trained || {})[driver]?.objective;
    if (driverChanged && trainedObjective) node("runner-objective").value = trainedObjective;
    const shown = fieldsFor(driver, node("runner-supervisor").value, options);
    const supervisor = node("runner-supervisor").value;
    node("runner-google-model-label").hidden = !shown.supervisor || supervisor !== "gemini/custom";
    node("runner-google-tools").hidden = !shown.supervisor || !isGoogle(supervisor);
    const image = settings?.image_adjustments;
    const adjusted = image && (image.exposure_ev !== 0 || image.contrast !== 1 || image.saturation !== 1);
    node("runner-mode").textContent = shown.mode + (shown.trained && adjusted ? " Image adjustments are on: the trained driver sees adjusted frames." : "");
    for (const [field, id] of [["supervisor", "runner-supervisor-label"], ["tier", "runner-tier-label"], ["brief", "runner-brief-label"], ["prompt", "runner-prompt-label"]]) node(id).hidden = !shown[field];
    renderPrompt();
  }
  node("runner-driver").addEventListener("change", () => { syncTiers(false); syncFields(true); });
  node("runner-supervisor").addEventListener("change", () => { syncTiers(false); syncFields(false); node("runner-google-result").hidden = true; });
  for (const id of ["runner-google-model", "runner-tier"]) node(id).addEventListener("change", () => { node("runner-google-result").hidden = true; });

  fetch("/runner/options").then((r) => r.json()).then((o) => {
    fill("runner-supervisor", o.supervisors);
    googleModels(o.gemini_models || []);
    fill("runner-tier", o.tiers);
    fill("runner-driver", o.drivers);
    fill("runner-outcome", o.outcomes);
    options = o;
    syncTiers(false);
    syncFields(true);
    node("runner-key").textContent = [
      o.key_present ? "OpenRouter key found" : `No OpenRouter key: add the Keychain item ${o.keychain_item}`,
      o.gemini_key_present ? "Gemini key found" : `no Gemini key (Keychain item ${o.gemini_keychain_item})`,
    ].join(" · ");
  });
  refresh();
  setInterval(refresh, 1000);
  return { describe, stateWords, controls, fieldsFor, chatModel, googleChoices };
})();
