"use client";

import { useEffect, useRef, useState } from "react";

import { frameScheduler } from "@/engine/scheduler/FrameScheduler";

// Dev-only perf overlay. Toggle with Shift+P. Renders a fixed-position
// panel showing instantaneous frame time, p95 over the last 5 s, long-
// task count, GC spikes (proxied as >50 ms dt jumps), and per-tick EMA
// from the FrameScheduler. No-op in production builds — the entire
// component returns null when NODE_ENV !== "development".
//
// Don't pull this into ad-hoc places. The whole point is one panel that
// shows what every render-path tick costs in one place — when someone
// adds work, they see it here, not three months later when Vibor reports
// jitter.

const REFRESH_HZ = 4; // panel updates 4× / s — enough to read, low cost

export function PerfOverlay(): React.ReactElement | null {
  const [visible, setVisible] = useState(false);
  const [, force] = useState(0);
  const ref = useRef({ tick: 0 });

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onKey = (e: KeyboardEvent) => {
      // Shift+P toggle. We require the modifier so plain "p" stays
      // available for input fields. Match either side of Shift.
      if (e.shiftKey && (e.key === "P" || e.key === "p")) {
        e.preventDefault();
        setVisible((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (!visible) return;
    let alive = true;
    const intervalMs = 1000 / REFRESH_HZ;
    const id = window.setInterval(() => {
      if (!alive) return;
      ref.current.tick++;
      force(ref.current.tick);
    }, intervalMs);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [visible]);

  if (process.env.NODE_ENV !== "development") return null;
  if (!visible) return null;

  const stats = frameScheduler.getStats();
  const budget = 16.6;
  const p95Color =
    stats.p95FrameMs < 0
      ? "#888"
      : stats.p95FrameMs <= 8
        ? "#7ec47e"
        : stats.p95FrameMs <= budget
          ? "#e8b95c"
          : "#e8615c";

  return (
    <div
      style={{
        position: "fixed",
        top: 8,
        right: 8,
        zIndex: 999999,
        padding: "8px 10px",
        background: "rgba(10,10,12,0.85)",
        color: "#dcdcdc",
        font: "11px ui-monospace, SFMono-Regular, monospace",
        lineHeight: 1.4,
        borderRadius: 6,
        backdropFilter: "blur(6px)",
        boxShadow: "0 2px 12px rgba(0,0,0,0.5)",
        pointerEvents: "none",
        minWidth: 220,
        maxWidth: 320,
      }}
    >
      <div
        style={{ fontWeight: 600, color: "#fff", marginBottom: 4 }}
      >
        perf · Shift+P
      </div>
      <div>
        last frame:{" "}
        <span style={{ color: stats.lastFrameMs > budget ? "#e8615c" : "#7ec47e" }}>
          {stats.lastFrameMs.toFixed(2)} ms
        </span>
      </div>
      <div>
        p95 (5s):{" "}
        <span style={{ color: p95Color }}>
          {stats.p95FrameMs < 0 ? "—" : `${stats.p95FrameMs.toFixed(2)} ms`}
        </span>
      </div>
      <div>
        long tasks: {stats.longTaskCount} · gc spikes: {stats.gcSpikeCount}
      </div>
      <div
        style={{
          height: 1,
          background: "rgba(255,255,255,0.12)",
          margin: "5px 0",
        }}
      />
      {stats.ticks.length === 0 && <div style={{ opacity: 0.5 }}>no ticks registered</div>}
      {stats.ticks.map((t) => {
        const over = t.emaMs > t.budgetMs;
        return (
          <div key={t.name} style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ opacity: t.phase === "compute" ? 0.7 : 1 }}>
              {t.phase === "compute" ? "·" : " "}
              {t.name}
            </span>
            <span style={{ color: over ? "#e8615c" : "#dcdcdc" }}>
              {t.emaMs.toFixed(2)} / {t.budgetMs}ms
            </span>
          </div>
        );
      })}
    </div>
  );
}
