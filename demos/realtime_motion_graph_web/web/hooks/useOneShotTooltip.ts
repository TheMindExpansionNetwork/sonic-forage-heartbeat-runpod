"use client";

import { useCallback, useEffect, useState } from "react";

// One-shot variant of the [data-dd-tooltip] custom tooltip — shows the
// tooltip the first time a user hovers the element, then suppresses it
// forever after (persisted in localStorage). For top-level affordances
// (LibraryButton cassette, RecordButton turntable) where the tooltip
// teaches the button's identity once; repeatedly displaying the same
// tooltip every hover becomes noise once the user knows the button.
//
// Usage:
//   const tipProps = useOneShotTooltip("library", "Library");
//   return (
//     <button
//       {...tipProps}
//       data-dd-tooltip-pos="below"
//       aria-label="Open library"  // persistent for screen readers
//     >…</button>
//   );
//
// `aria-label` should stay on the element separately and unchanged —
// the one-shot behavior only governs the visual tooltip pseudo. Screen
// reader users still get the same affordance information every time.

const STORAGE_PREFIX = "dd:tooltip-shown:";

export interface OneShotTooltipProps {
  /** Spread alongside `data-dd-tooltip-pos`. Becomes undefined after the
   *  first hover (or on subsequent page loads if already dismissed). */
  "data-dd-tooltip"?: string;
  onMouseEnter: () => void;
}

export function useOneShotTooltip(key: string, label: string): OneShotTooltipProps {
  const storageKey = `${STORAGE_PREFIX}${key}`;
  // Pessimistic initial: render without the tooltip attribute until
  // the effect confirms first-run (avoids a brief tooltip flash for
  // returning users whose localStorage already has the flag). The
  // effect runs synchronously after mount, well before any plausible
  // user hover.
  const [shown, setShown] = useState<boolean>(true);

  useEffect(() => {
    try {
      setShown(localStorage.getItem(storageKey) === "1");
    } catch {
      // private browsing / quota — treat as first-run. Worst case
      // is the user sees the tooltip once per session.
      setShown(false);
    }
  }, [storageKey]);

  const onMouseEnter = useCallback(() => {
    if (shown) return;
    setShown(true);
    try {
      localStorage.setItem(storageKey, "1");
    } catch {
      // private browsing — suppress for this session only.
    }
  }, [shown, storageKey]);

  return shown ? { onMouseEnter } : { "data-dd-tooltip": label, onMouseEnter };
}
