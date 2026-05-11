// Title-screen → graph morph. Each of the four wavy ribbons around the
// start logo unwraps into a specific parameter's graph polyline:
//
//   ribbon 0 (teal)    → denoise         ("remix")
//   ribbon 1 (mustard) → hint_strength   ("structure")
//   ribbon 2 (orange)  → timbre_strength ("timbre")
//   ribbon 3 (coral)   → ch_g0           ("channel 0")
//
// Mechanics: a temporary full-viewport canvas mounts above the start
// overlay; at t=0 it draws N_SAMPLES vertices walking around the SVG
// ribbon's closed-circle math (so the first frame is pixel-identical
// to the SVG it replaces). At t=1 those same vertices land on a flat
// polyline at each signal's current normalized value, positioned
// EXACTLY where GraphRenderer will draw its polyline once the session
// is live — same x density (VISIBLE_SAMPLES), same y math, same
// playhead inset. The caller pre-fills GraphRenderer's histories with
// those same flat values before this runs, so the morph canvas
// removal at t=1 is invisible: the graph canvas beneath has the
// identical flat lines drawn into it.
//
// Wave amplitude is naturally damped by the lerp: as t→1, source and
// destination converge and the wiggle vanishes. No special damping
// schedule needed.
//
// The SVG element behind us keeps animating (the global render loop
// doesn't know about us) — but its CSS is overridden to visibility:
// hidden during the morph, so the user never sees the two visuals
// fighting.

import {
  GRAPH_Y_PAD,
  GRAPH_PLAYHEAD_INSET_PX_FRAC,
  GRAPH_VISIBLE_SAMPLES,
} from "./GraphRenderer";

// Must match the START_MARK_* constants in ribbons.ts; the math has
// to be identical so the morph's first frame overlays the SVG path
// pixel-for-pixel. (Importing them from ribbons.ts would also work
// but they're module-private there; duplicating is one-time and the
// values are stable visual constants.)
const RIBBON_BASE_R = 40;
const RIBBON_RADIAL_SPREAD = 2.4;
const RIBBON_NOISE_AMP = 5.5;
const RIBBON_TIME_SCALE = 0.55;

// Polyline resolution. Matches GraphRenderer.VISIBLE_SAMPLES so the
// morph's end-state polyline is identical to what GraphRenderer
// renders on its first post-morph frame.
const N_SAMPLES = GRAPH_VISIBLE_SAMPLES;

export interface MorphSignal {
  /** Which of the four ribbon paths this signal hijacks (0..3). */
  ribbonIdx: 0 | 1 | 2 | 3;
  /** Normalized destination y on the graph (0..1; 0 = bottom, 1 = top). */
  value: number;
  /** Color the graph will draw this signal with (matches GRAPH_COLORS in
   *  GraphRenderer). Used as the destination of the color lerp so the
   *  handoff is color-continuous. */
  destColor: readonly [number, number, number];
  /** Source ribbon stroke color (PALETTE entry as `#rrggbb`). */
  srcColor: string;
}

export interface MorphOpts {
  /** Snapshot of the .start-cta-halo bounding rect, taken at click time
   *  BEFORE the parent flips status out of idle (which would .hidden the
   *  overlay → display:none → live getBoundingClientRect returns
   *  {0,0,0,0}). The wave center stays anchored here for the whole
   *  morph; the wave itself continues oscillating relative to this
   *  frozen anchor. */
  haloRect: { left: number; top: number; width: number };
  /** Anchors the destination. Need not be visible yet (caller is
   *  responsible for un-hiding it via the launching-morph body class). */
  graphCanvas: HTMLCanvasElement;
  signals: MorphSignal[];
  durationMs: number;
  /** #graph's CSS feather distances (pixels). Used to fade the morph
   *  canvas's polyline edges to match the graph's mask-image so the
   *  hand-off doesn't show crisp morph lines extending past where the
   *  graph fades. The fade ramps in by progress so it has no effect
   *  during the early circle state and full effect at the landing. */
  featherX: number;
  featherY: number;
}

/** Run the morph. Resolves when the canvas has been removed. */
export function runStartMorph(opts: MorphOpts): Promise<void> {
  const { haloRect, graphCanvas, signals, durationMs, featherX, featherY } =
    opts;
  // Frozen wave center / scale, derived from the click-time halo rect.
  // The wave keeps oscillating during the morph; the anchor doesn't move.
  const svgScale = haloRect.width / 100;
  const svgCx = haloRect.left + 50 * svgScale;
  const svgCy = haloRect.top + 50 * svgScale;

  const canvas = document.createElement("canvas");
  // Above the start overlay (z-index: 20) and the launching CTA itself
  // (which has its own stacking context via transform). Below any modal
  // root portaled to body.
  canvas.style.cssText =
    "position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;z-index:25";
  document.body.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    document.body.removeChild(canvas);
    return Promise.resolve();
  }

  const dpr = Math.min(2, window.devicePixelRatio || 1);
  function sizeCanvas() {
    canvas.width = Math.max(1, Math.floor(window.innerWidth * dpr));
    canvas.height = Math.max(1, Math.floor(window.innerHeight * dpr));
    ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx!.lineCap = "round";
    ctx!.lineJoin = "round";
  }
  sizeCanvas();
  const onResize = () => sizeCanvas();
  window.addEventListener("resize", onResize);

  return new Promise<void>((resolve) => {
    const startedAt = performance.now();

    function frame(now: number) {
      const elapsed = now - startedAt;
      const linearT = Math.max(0, Math.min(1, elapsed / durationMs));
      // easeInOutCubic — slow at both ends, fast through the middle. The
      // start-end slowness gives the user a beat to register "circles"
      // and "lines" as distinct states; the middle speed sells the
      // transformation as a single motion rather than a tween.
      const t =
        linearT < 0.5
          ? 4 * linearT * linearT * linearT
          : 1 - Math.pow(-2 * linearT + 2, 3) / 2;
      // Live ribbon time (continues oscillating during the morph; the
      // lerp damps the resulting wiggle naturally).
      const ribbonTime = (elapsed / 1000) * RIBBON_TIME_SCALE;

      // Destination geometry is re-measured each frame so a resize
      // mid-morph doesn't desync the landing position. (The source
      // anchor was snapshotted at click time — see comment above.)
      const graphRect = graphCanvas.getBoundingClientRect();

      // Destination polyline geometry — must mirror GraphRenderer.draw().
      // Playhead anchors the newest sample; older samples extend left.
      const playheadXScreen =
        graphRect.left + graphRect.width * (1 - GRAPH_PLAYHEAD_INSET_PX_FRAC);
      const pxPerSample = graphRect.width / (GRAPH_VISIBLE_SAMPLES - 1);
      const dstYBase = graphRect.top + (graphRect.height - GRAPH_Y_PAD);
      const dstYSpan = graphRect.height - 2 * GRAPH_Y_PAD;

      ctx!.clearRect(0, 0, window.innerWidth, window.innerHeight);

      for (let s = 0; s < signals.length; s++) {
        const sig = signals[s];
        const ribbonIdx = sig.ribbonIdx;
        const phase = ribbonIdx * 0.9;
        const radialOffset =
          (ribbonIdx - (4 - 1) / 2) * RIBBON_RADIAL_SPREAD;
        const srcRgb = parseHex(sig.srcColor);

        const r = lerp(srcRgb[0], sig.destColor[0], t);
        const g = lerp(srcRgb[1], sig.destColor[1], t);
        const b = lerp(srcRgb[2], sig.destColor[2], t);
        // Source ribbon: 0.88 alpha + 2.6 stroke (matches CSS).
        // Destination graph: 1.0 alpha + 1.0 stroke (matches GraphRenderer
        // baseAlpha at pulse=0). Lerp both.
        const alpha = lerp(0.88, 1.0, t);
        const lineWidth = lerp(2.6, 1.0, t);
        // Destination y. value is already normalized (0..1).
        const dstY = dstYBase - sig.value * dstYSpan;

        ctx!.strokeStyle = `rgba(${r | 0},${g | 0},${b | 0},${alpha})`;
        ctx!.lineWidth = lineWidth;
        ctx!.beginPath();

        for (let i = 0; i < N_SAMPLES; i++) {
          const u = i / (N_SAMPLES - 1);
          // Source vertex: walk around the ribbon's closed circle. The
          // ribbon is a closed loop so vertex 0 and vertex N-1 are at the
          // same angle (modulo numerical drift), which is fine — the
          // destination has both landing at the same flat y, so the
          // closed circle naturally "opens" without a seam artifact.
          const theta = u * Math.PI * 2;
          const noise =
            Math.sin(theta * 2 + ribbonTime + phase) * 0.65 +
            Math.sin(theta * 5 - ribbonTime * 1.3 + phase * 1.5) * 0.35;
          const rad = RIBBON_BASE_R + radialOffset + noise * RIBBON_NOISE_AMP;
          const sx = svgCx + rad * Math.cos(theta) * svgScale;
          const sy = svgCy + rad * Math.sin(theta) * svgScale;

          // Destination vertex: i=0 is the oldest sample (far left),
          // i=N-1 is at the playhead. Identical layout to GraphRenderer.
          const dx = playheadXScreen - (N_SAMPLES - 1 - i) * pxPerSample;

          const x = sx + (dx - sx) * t;
          const y = sy + (dstY - sy) * t;
          if (i === 0) ctx!.moveTo(x, y);
          else ctx!.lineTo(x, y);
        }
        ctx!.stroke();
      }

      // Edge feather to match the live graph's CSS mask. Without this,
      // the morph's lines extend solid past the graph's faded left/right
      // edges and out of the top/bottom feather strip, so at the
      // hand-off moment the eye lands on the wrong (un-faded) lines.
      //
      // Implementation: paint a destination-out erasure that matches the
      // graph's mask shape, scaled by `t` so it has no effect at the
      // start (lines are still around the halo, far from any graph
      // edges) and full effect at the landing (matches the graph
      // exactly). At t=1, dest_alpha *= (1 - graph_mask_alpha), which is
      // equivalent to dest_alpha *= graph_mask_alpha applied to the
      // already-drawn polylines.
      //
      // Five regions:
      //   - outside graph (all four strips around the rect)
      //   - inside graph, left feather
      //   - inside graph, right feather
      //   - inside graph, top feather
      //   - inside graph, bottom feather
      // Feather strips overlap at corners, which doubles the erasure
      // there — matches the graph's `mask-composite: intersect` (which
      // multiplies horizontal × vertical mask alphas) well enough.
      if (t > 0) {
        ctx!.save();
        ctx!.globalCompositeOperation = "destination-out";
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const gL = graphRect.left;
        const gT = graphRect.top;
        const gR = graphRect.right;
        const gB = graphRect.bottom;
        const erase = `rgba(0,0,0,${t})`;

        // Outside-graph: four strips. graph_mask_alpha = 0 here, so the
        // erasure is uniform (not gradient).
        ctx!.fillStyle = erase;
        ctx!.fillRect(0, 0, gL, vh); // left of graph
        ctx!.fillRect(gR, 0, vw - gR, vh); // right of graph
        ctx!.fillRect(gL, 0, gR - gL, gT); // top of graph
        ctx!.fillRect(gL, gB, gR - gL, vh - gB); // bottom of graph

        // Inside-graph feather strips. Each gradient erases by `t` at
        // the outer edge (matching graph mask alpha = 0 there) and 0
        // at the feather distance inside (matching graph mask alpha = 1).
        const gradL = ctx!.createLinearGradient(gL, 0, gL + featherX, 0);
        gradL.addColorStop(0, erase);
        gradL.addColorStop(1, "rgba(0,0,0,0)");
        ctx!.fillStyle = gradL;
        ctx!.fillRect(gL, gT, featherX, gB - gT);

        const gradR = ctx!.createLinearGradient(gR - featherX, 0, gR, 0);
        gradR.addColorStop(0, "rgba(0,0,0,0)");
        gradR.addColorStop(1, erase);
        ctx!.fillStyle = gradR;
        ctx!.fillRect(gR - featherX, gT, featherX, gB - gT);

        const gradT = ctx!.createLinearGradient(0, gT, 0, gT + featherY);
        gradT.addColorStop(0, erase);
        gradT.addColorStop(1, "rgba(0,0,0,0)");
        ctx!.fillStyle = gradT;
        ctx!.fillRect(gL, gT, gR - gL, featherY);

        const gradB = ctx!.createLinearGradient(0, gB - featherY, 0, gB);
        gradB.addColorStop(0, "rgba(0,0,0,0)");
        gradB.addColorStop(1, erase);
        ctx!.fillStyle = gradB;
        ctx!.fillRect(gL, gB - featherY, gR - gL, featherY);

        ctx!.restore();
      }

      if (linearT < 1) {
        requestAnimationFrame(frame);
      } else {
        window.removeEventListener("resize", onResize);
        if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
        resolve();
      }
    }
    requestAnimationFrame(frame);
  });
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function parseHex(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}
