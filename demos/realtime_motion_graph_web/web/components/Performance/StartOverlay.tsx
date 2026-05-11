"use client";

import { useState } from "react";

import { runStartMorph, type MorphSignal } from "@/engine/render/startMorph";
import { START_MARK_PALETTE } from "@/engine/render/ribbons";
import { useIsMobile } from "@/hooks/useIsMobile";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { SLIDER_META } from "@/types/engine";

interface Props {
  onPlay: () => void;
  hidden?: boolean;
}

// Total length of the click-to-launch CSS animation (icon zoom + whisper
// fade). Kept in sync with the @keyframes durations in globals.css.
const LAUNCH_DURATION_MS = 700;
// Morph duration. Deliberately slow — the ribbon unfurl runs through
// most of the session-start network wait, so the user perceives the
// loading time as part of the transition rather than dead air.
const MORPH_DURATION_MS = 2400;

// Destination signals — one per ribbon. Order matches START_MARK_PALETTE
// (teal / mustard / orange / coral) so each ribbon's stroke color stays
// continuous from its CSS resting state into the graph's plotted color
// (timbre_strength's graph color was set to match the orange ribbon for
// exactly this — see GraphRenderer's GRAPH_COLORS comment).
const MORPH_PARAM_FOR_RIBBON: readonly string[] = [
  "denoise", // ribbon 0 — "remix"
  "hint_strength", // ribbon 1 — "structure"
  "timbre_strength", // ribbon 2 — "timbre"
  "ch_g0", // ribbon 3 — "channel 0"
] as const;
const MORPH_DEST_COLORS: readonly (readonly [number, number, number])[] = [
  [61, 182, 190],
  [199, 181, 102],
  [240, 138, 72],
  [255, 80, 80],
] as const;

// The brand mark IS the play button. The icon halo + the "click/tap to
// begin" whisper sit inside a single <button> so anywhere on either
// element triggers onPlay — testers were instinctively trying to click
// the copy itself. On click, the ribbons spin + explode outward while
// the icon zooms forward; only then does onPlay fire and the overlay
// give way to the app.
export function StartOverlay({ onPlay, hidden }: Props) {
  const isMobile = useIsMobile();
  const [launching, setLaunching] = useState(false);

  function handleClick() {
    if (launching) return;
    setLaunching(true);

    // Compute the destination signals from current slider state. The
    // graph will plot these AT these values once it starts live
    // sampling, so the morph's flat-line end-state matches the live
    // graph pixel-for-pixel.
    const perf = usePerformanceStore.getState();
    const signals: MorphSignal[] = [];
    const prefill: Record<string, number> = {};
    for (let i = 0; i < MORPH_PARAM_FOR_RIBBON.length; i++) {
      const param = MORPH_PARAM_FOR_RIBBON[i];
      const max = SLIDER_META[param]?.max ?? 1;
      const raw = perf.sliderValues[param] ?? 0;
      const value = Math.max(0, Math.min(1, raw / max));
      prefill[param] = value;
      signals.push({
        ribbonIdx: i as 0 | 1 | 2 | 3,
        value,
        destColor: MORPH_DEST_COLORS[i],
        srcColor: START_MARK_PALETTE[i],
      });
    }

    // Pre-fill the live graph's histories so the moment the morph
    // canvas removes itself, the graph underneath already shows the
    // identical flat polylines. useRenderLoop forwards this to
    // GraphRenderer.prefillHistory().
    document.dispatchEvent(
      new CustomEvent("dd:graph-prefill", { detail: { samples: prefill } }),
    );

    // Snapshot the halo rect NOW, before onPlay() fires. useStartSession
    // calls setStatus("loading-fixture", …) synchronously, which flips
    // `started` true in the parent, which applies .hidden → display:none
    // to the overlay. Once that lands, getBoundingClientRect on the halo
    // (or any descendant of the overlay) returns {0,0,0,0} and the wave
    // collapses to the viewport's top-left corner. Reading it here
    // freezes the click-time anchor regardless of when the next paint
    // lands.
    const haloEl = document.querySelector(
      ".start-cta-halo",
    ) as HTMLElement | null;
    const graphCanvas = document.getElementById(
      "graph",
    ) as HTMLCanvasElement | null;
    const haloRectLive = haloEl?.getBoundingClientRect();
    const haloRect =
      haloRectLive && haloRectLive.width > 0
        ? {
            left: haloRectLive.left,
            top: haloRectLive.top,
            width: haloRectLive.width,
          }
        : null;

    // Un-hide the graph for the duration of the morph so its rect is
    // measurable AND so the prefilled polylines are visible beneath
    // the (now-transparent, via CSS) overlay backdrop as the morph
    // canvas lifts off it. Removed by the timeout below.
    document.body.classList.add("launching-morph");

    // Fire onPlay() immediately so the queue.start() network round-trip
    // overlaps with the launch animation rather than starting after it.
    // If we deferred onPlay() to the timeout, the launching class would
    // come off mid-flight while queue.status is still 'idle'/'joining',
    // briefly snapping the CTA back to its base "click to begin" state
    // until the join response landed.
    onPlay();

    // Kick off the morph. Runs in parallel with onPlay() (which has
    // already changed `status` and queued a re-render that will
    // .hidden the overlay — fine, since the morph uses the frozen
    // halo snapshot and runs on a separate canvas mounted to body).
    if (haloRect && graphCanvas) {
      // Read the graph's CSS feather distances so the morph's edge
      // fade matches the live graph's mask-image exactly. clamp()
      // resolves on the computed style even when the element is
      // display:none (CSS vars don't require layout). Fallbacks
      // mirror the CSS formula so a missing var doesn't visibly
      // mis-fade.
      const graphStyle = getComputedStyle(graphCanvas);
      const parsedFx = parseFloat(
        graphStyle.getPropertyValue("--graph-feather-x"),
      );
      const parsedFy = parseFloat(
        graphStyle.getPropertyValue("--graph-feather-y"),
      );
      const featherX =
        Number.isFinite(parsedFx) && parsedFx > 0
          ? parsedFx
          : Math.max(80, Math.min(220, 0.14 * window.innerWidth));
      const featherY =
        Number.isFinite(parsedFy) && parsedFy > 0 ? parsedFy : 36;

      void runStartMorph({
        haloRect,
        graphCanvas,
        signals,
        durationMs: MORPH_DURATION_MS,
        featherX,
        featherY,
      }).then(() => {
        // Morph canvas has removed itself. Hand off complete; the live
        // graph's prefilled flat polylines are now on screen.
        document.body.classList.remove("launching-morph");
      });
    } else {
      // Defensive: if the halo or graph aren't mounted (shouldn't
      // happen in practice), still clean up the body class so the
      // graph isn't stuck visible behind a re-shown overlay.
      document.body.classList.remove("launching-morph");
    }

    window.setTimeout(() => {
      // Happy-path: parent re-renders us with hidden=true (queue admits
      // OR session starts) and this state never matters. Sad-path: gate,
      // error, or paywall keeps us mounted and we'd otherwise be stuck
      // post-launch animation showing nothing — reset so the user can
      // see + interact with the title screen again.
      setLaunching(false);
    }, LAUNCH_DURATION_MS);
  }

  const whisper = isMobile ? "tap to begin" : "click to begin";

  return (
    <div id="start-overlay" className={hidden ? "hidden" : ""}>
      <button
        type="button"
        className={`start-cta${launching ? " start-cta--launching" : ""}`}
        onClick={handleClick}
        aria-label={isMobile ? "Tap to begin" : "Click to begin"}
        disabled={launching}
      >
        <span className="start-cta-halo" aria-hidden="true">
          {/* Writhing ribbon halo around the logo — populated by
              useRenderLoop's tickStartMarkRibbon. */}
          <svg
            className="start-mark-ribbons"
            viewBox="0 0 100 100"
            preserveAspectRatio="xMidYMid meet"
            aria-hidden="true"
          >
            {START_MARK_PALETTE.map((color) => (
              <path
                key={color}
                stroke={color}
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
          </svg>
          <img
            className="start-mark"
            src="/daydream-icon-clean.png"
            alt=""
          />
        </span>
        <span className="start-whisper">{whisper}</span>
      </button>
    </div>
  );
}
