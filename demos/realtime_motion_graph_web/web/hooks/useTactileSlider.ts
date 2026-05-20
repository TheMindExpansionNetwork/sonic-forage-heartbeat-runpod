"use client";

import { useEffect, useRef } from "react";

import { valueToT, type SliderMapping } from "@/lib/sliderMapping";
import { usePerformanceStore } from "@/store/usePerformanceStore";

// Lightweight per-slider tactile augmentation: feature-detected haptics on
// landmark crossings (0, 0.5, 1.0 of the THUMB position).
//
// We do NOT take over the pointer pipeline — callers keep their own drag
// handlers. We just attach a few extra listeners to the same element and
// react to the same events.

const HAPTIC_CROSSINGS = [0, 0.5, 1.0] as const;
const HAPTIC_TOL = 0.04;

function vibrate(ms: number): void {
  if (typeof navigator === "undefined") return;
  const v = (navigator as Navigator & { vibrate?: (n: number) => boolean })
    .vibrate;
  if (typeof v === "function") {
    try {
      v.call(navigator, ms);
    } catch {}
  }
}

interface Options {
  /** sliderTargets key (e.g. "denoise", "lora_blend"). */
  param: string;
  /** Same mapping bundle SliderGroup uses (min/max/unity/reverse). We
   *  fire haptics on the thumb's position crossings (0 / 0.5 / 1 of the
   *  rail), not on engine-value crossings, so reverse + unity-anchored
   *  channels still feel landmarks at the bottom, middle, and top of
   *  the rail. Bypasses any asymmetry between value and thumb position
   *  introduced by the unity-anchored piecewise mapping. */
  mapping: SliderMapping;
}

export function useTactileSlider({ param, mapping }: Options): void {
  // Track previous fraction so we only fire haptic at the moment a crossing
  // happens, not on every redraw at that position.
  const prevFrac = useRef<number | null>(null);

  // Reactively re-bind to mapping changes via the primitive fields, not
  // the object identity (which gets rebuilt every parent render).
  const { min, max, unity, reverse } = mapping;

  // Subscribe directly to the store: cheaper than re-rendering the parent
  // for every value change, and lets us fire vibrate() outside React's
  // commit phase.
  useEffect(() => {
    const m: SliderMapping = { min, max, unity, reverse };
    const fire = () => {
      const v = usePerformanceStore.getState().sliderTargets[param] ?? 0;
      // Thumb position fraction (0 at bottom, 1 at top). For unity-
      // anchored bands, value=unity always lands at frac=0.5 — so the
      // mid-rail haptic fires when the operator drags through unity
      // regardless of where unity sits in the channel's [min, max].
      const frac = valueToT(v, m);
      const prev = prevFrac.current;
      prevFrac.current = frac;
      if (prev === null) return;
      for (const cross of HAPTIC_CROSSINGS) {
        const wasNear = Math.abs(prev - cross) <= HAPTIC_TOL;
        const isNear = Math.abs(frac - cross) <= HAPTIC_TOL;
        if (!wasNear && isNear) {
          vibrate(8);
          break;
        }
      }
    };
    return usePerformanceStore.subscribe(fire);
  }, [param, min, max, unity, reverse]);
}
