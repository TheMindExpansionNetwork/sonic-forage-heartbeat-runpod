// Single-rAF policy. Every per-frame callback in the codebase registers
// here and runs in a defined phase order; the scheduler owns the only
// `requestAnimationFrame` call. Goals:
//
//   - One rAF wakeup per vsync, not six. (Pre-2026-perf-pass we had
//     useRenderLoop, tickTweens, useScheduledCurves, AmbientRenderer ×2,
//     VideoLayer, and ScheduleCurvesOverlay all racing for the same
//     vsync. The slowest gated the frame; the cursor felt sticky on
//     beats because that's when all six had work.)
//
//   - Compute → Render phase ordering. Tweens and curves must update
//     store state BEFORE useRenderLoop reads it; otherwise we render
//     a frame behind. Phases enforce that without per-callback
//     coordination.
//
//   - Per-tick instrumentation. The dev perf overlay reads `getStats()`
//     to show which callback is consuming the budget. Without this,
//     "the page feels slow" is unactionable.
//
// Don't add a new requestAnimationFrame call elsewhere — see
// PERFORMANCE.md.

export type TickFn = (now: number, dt: number) => void;

export type TickPhase = "compute" | "render";

interface RegisteredTick {
  name: string;
  phase: TickPhase;
  fn: TickFn;
  // EMA of per-tick wall time in ms. Updated every frame the tick runs.
  // Surfaced via getStats() to the dev overlay so we can see who's
  // eating the budget.
  emaMs: number;
  // Soft budget. When exceeded, the dev overlay flags this tick. No
  // hard enforcement — we don't drop callbacks; we just surface them.
  budgetMs: number;
}

const EMA_ALPHA = 0.1;

export interface FrameStats {
  // Most recent frame's total wall time across all ticks (ms).
  lastFrameMs: number;
  // Rolling p95 frame time over the last 5 s. -1 until first sample.
  p95FrameMs: number;
  // Per-tick stats, sorted by EMA descending so the slowest is first.
  ticks: { name: string; phase: TickPhase; emaMs: number; budgetMs: number }[];
  // Long-task count (>50 ms tasks) since last reset. PerformanceObserver
  // owns this; FrameScheduler just exposes it.
  longTaskCount: number;
  // GC bars approximated as "frames where dt jumped > 50 ms". Browsers
  // don't expose GC pauses directly, so this is the cheapest proxy.
  gcSpikeCount: number;
}

class FrameScheduler {
  private _ticks: RegisteredTick[] = [];
  private _rafId: number | null = null;
  private _lastNow = 0;
  // Rolling 5 s window of frame durations for p95.
  private readonly _frameMs: Float32Array = new Float32Array(300); // ~5s at 60fps
  private _frameMsHead = 0;
  private _frameMsCount = 0;
  private _longTaskCount = 0;
  private _gcSpikeCount = 0;
  private _longTaskObs: PerformanceObserver | null = null;

  /** Register a tick. Returns an unregister function (call in cleanup). */
  register(
    name: string,
    fn: TickFn,
    opts?: { phase?: TickPhase; budgetMs?: number },
  ): () => void {
    const tick: RegisteredTick = {
      name,
      phase: opts?.phase ?? "render",
      fn,
      emaMs: 0,
      budgetMs: opts?.budgetMs ?? 8,
    };
    this._ticks.push(tick);
    this._ensureRunning();
    return () => {
      const i = this._ticks.indexOf(tick);
      if (i >= 0) this._ticks.splice(i, 1);
      if (this._ticks.length === 0) this._stop();
    };
  }

  /** Force-stop; useful for HMR teardown. Tests only. */
  _stop(): void {
    if (this._rafId !== null && typeof cancelAnimationFrame !== "undefined") {
      cancelAnimationFrame(this._rafId);
    }
    this._rafId = null;
    this._lastNow = 0;
  }

  getStats(): FrameStats {
    const ticks = this._ticks
      .map((t) => ({
        name: t.name,
        phase: t.phase,
        emaMs: t.emaMs,
        budgetMs: t.budgetMs,
      }))
      .sort((a, b) => b.emaMs - a.emaMs);
    return {
      lastFrameMs: this._lastFrameMs,
      p95FrameMs: this._computeP95(),
      ticks,
      longTaskCount: this._longTaskCount,
      gcSpikeCount: this._gcSpikeCount,
    };
  }

  resetCounters(): void {
    this._longTaskCount = 0;
    this._gcSpikeCount = 0;
    this._frameMsHead = 0;
    this._frameMsCount = 0;
  }

  private _lastFrameMs = 0;

  private _ensureRunning(): void {
    if (this._rafId !== null) return;
    if (typeof requestAnimationFrame === "undefined") return;
    this._installLongTaskObserver();
    this._rafId = requestAnimationFrame(this._tick);
  }

  private _installLongTaskObserver(): void {
    if (this._longTaskObs) return;
    if (typeof PerformanceObserver === "undefined") return;
    try {
      this._longTaskObs = new PerformanceObserver((list) => {
        this._longTaskCount += list.getEntries().length;
      });
      this._longTaskObs.observe({ entryTypes: ["longtask"] });
    } catch {
      // longtask not supported in this browser — silently skip.
      this._longTaskObs = null;
    }
  }

  private _tick = (now: number): void => {
    if (this._ticks.length === 0) {
      this._rafId = null;
      return;
    }
    const dt = this._lastNow === 0 ? 16 : Math.min(50, now - this._lastNow);
    if (this._lastNow !== 0 && dt > 50) {
      // Likely a GC pause or tab restoration. Count it.
      this._gcSpikeCount++;
    }
    this._lastNow = now;

    const frameStart = performance.now();

    // Compute phase: tweens, curves, anything that updates store state.
    for (let i = 0; i < this._ticks.length; i++) {
      const tick = this._ticks[i];
      if (tick.phase !== "compute") continue;
      const t0 = performance.now();
      try {
        tick.fn(now, dt);
      } catch (err) {
        console.error(`[FrameScheduler] tick "${tick.name}" threw`, err);
      }
      const elapsed = performance.now() - t0;
      tick.emaMs = tick.emaMs === 0 ? elapsed : tick.emaMs * (1 - EMA_ALPHA) + elapsed * EMA_ALPHA;
    }

    // Render phase: canvas draws, DOM mutations driven by latest state.
    for (let i = 0; i < this._ticks.length; i++) {
      const tick = this._ticks[i];
      if (tick.phase !== "render") continue;
      const t0 = performance.now();
      try {
        tick.fn(now, dt);
      } catch (err) {
        console.error(`[FrameScheduler] tick "${tick.name}" threw`, err);
      }
      const elapsed = performance.now() - t0;
      tick.emaMs = tick.emaMs === 0 ? elapsed : tick.emaMs * (1 - EMA_ALPHA) + elapsed * EMA_ALPHA;
    }

    this._lastFrameMs = performance.now() - frameStart;
    this._frameMs[this._frameMsHead] = this._lastFrameMs;
    this._frameMsHead = (this._frameMsHead + 1) % this._frameMs.length;
    if (this._frameMsCount < this._frameMs.length) this._frameMsCount++;

    this._rafId = requestAnimationFrame(this._tick);
  };

  private _computeP95(): number {
    if (this._frameMsCount === 0) return -1;
    // Cheap p95: copy + sort the ring. ~300 entries, runs only when the
    // overlay queries — not per frame.
    const buf = this._frameMs.slice(0, this._frameMsCount);
    const sorted = Array.from(buf).sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * 0.95);
    return sorted[Math.min(idx, sorted.length - 1)];
  }
}

export const frameScheduler = new FrameScheduler();
