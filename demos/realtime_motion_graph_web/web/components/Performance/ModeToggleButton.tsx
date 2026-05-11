"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

const MODE_LABEL: Record<string, string> = {
  graph: "GRAPH",
  video: "VIDEO",
  neon: "NEON",
};

export function ModeToggleButton() {
  const mode = usePerformanceStore((s) => s.mode);
  const toggleMode = usePerformanceStore((s) => s.toggleMode);
  const status = useSessionStore((s) => s.status);
  const kiosk = usePerformanceStore((s) => s.kiosk);

  if (status === "idle" || kiosk) return null;

  return (
    <button
      type="button"
      className="mode-toggle-btn"
      onClick={toggleMode}
      title="Cycle display mode (graph → video → neon)"
      aria-label={`Display mode: ${MODE_LABEL[mode] ?? mode}. Click to cycle.`}
    >
      <span className="mode-toggle-btn__label">{MODE_LABEL[mode] ?? mode}</span>
    </button>
  );
}
