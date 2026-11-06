"use strict";

// A tab change only changes visibility. Runner state and model context live on the gateway.
(() => {
  const tabs = ["setup", "drive", "runner"];
  const node = (id) => document.getElementById(id);
  function selectTab(selected) {
    if (!tabs.includes(selected)) selected = "setup";
    for (const name of tabs) {
      node(`${name}-panel`).hidden = name !== selected;
      node(`${name}-tab`).setAttribute("aria-selected", String(name === selected));
      node(`${name}-tab`).tabIndex = name === selected ? 0 : -1;
    }
    window.history?.replaceState(null, "", `/?tab=${selected}`);
    window.dispatchEvent(new CustomEvent("drivingbench-tab", { detail: selected }));
  }
  for (const name of tabs) {
    node(`${name}-tab`).addEventListener("click", () => selectTab(name));
    node(`${name}-tab`).addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
        : (tabs.indexOf(name) + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      selectTab(tabs[next]);
      node(`${tabs[next]}-tab`).focus();
    });
  }
  selectTab(new URLSearchParams(window.location?.search || "").get("tab"));
})();
