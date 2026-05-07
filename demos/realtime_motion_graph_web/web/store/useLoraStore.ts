"use client";

import { create } from "zustand";

import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SLIDER_MAX,
} from "@/types/engine";
import type { LoraCatalogEntry } from "@/types/protocol";

// Server-driven LoRA catalog + per-id strength + enabled set. The catalog
// arrives via /api/loras (cheap filesystem scan, available before WS) and
// is updated mid-session via the WS "lora_catalog" frame.

// Preferred LoRA stems flipped on the first time the catalog arrives
// populated. If a preferred stem isn't in the catalog (different LoRA
// library locally), the slot falls back to the catalog entry at the
// same index, then to the next unclaimed entry — so a fresh page-load
// always lands with two LoRAs hot regardless of which files are on disk.
// One-shot: a later WS lora_catalog re-broadcast won't re-enable a LoRA
// the user has explicitly disabled.
const PREFERRED_DEFAULT_LORAS = ["deathstep", "synthpop"] as const;

interface LoraState {
  catalog: LoraCatalogEntry[];
  /** Per-id strength (0..LORA_SLIDER_MAX). */
  strengths: Record<string, number>;
  /** Set of enabled LoRA ids. */
  enabled: Set<string>;
  /** Whether default-on LoRAs have already been seeded for this session. */
  _seeded: boolean;

  setCatalog: (catalog: LoraCatalogEntry[]) => void;
  setStrength: (id: string, value: number) => void;
  enable: (id: string) => void;
  disable: (id: string) => void;
  toggle: (id: string) => void;
  reset: () => void;
}

export const useLoraStore = create<LoraState>((set) => ({
  catalog: [],
  strengths: {},
  enabled: new Set(),
  _seeded: false,

  setCatalog: (catalog) =>
    set((s) => {
      // Seed missing strengths from the server's reported defaults so a
      // freshly-arrived LoRA picks up its on-disk default. The Python
      // backend currently echoes 0.0 for every entry, which we treat as
      // "unset" and fall back to LORA_DEFAULT_STRENGTH_FRACTION so the
      // slider lands at a useful initial level. A genuine non-zero
      // server default still wins.
      const next: Record<string, number> = { ...s.strengths };
      for (const entry of catalog) {
        if (!(entry.id in next)) {
          next[entry.id] =
            typeof entry.strength === "number" && entry.strength > 0
              ? entry.strength
              : LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX;
        }
      }
      // First populated catalog: flip on the preferred default LoRAs so the
      // demo plays with its intended sound out of the box. If a preferred
      // stem isn't in the catalog, fall back to the catalog entry at that
      // slot index (with dedup so two missing defaults don't both claim the
      // first entry). Skipped on later re-broadcasts so disabling a seeded
      // LoRA sticks.
      let enabled = s.enabled;
      let seeded = s._seeded;
      if (!s._seeded && catalog.length > 0) {
        const nextEnabled = new Set(s.enabled);
        const present = new Set(catalog.map((e) => e.id));
        const claimed = new Set<string>();
        for (let i = 0; i < PREFERRED_DEFAULT_LORAS.length; i++) {
          const preferred = PREFERRED_DEFAULT_LORAS[i];
          let pick: string | undefined;
          if (present.has(preferred) && !claimed.has(preferred)) {
            pick = preferred;
          } else {
            const slot = catalog[i]?.id;
            pick = slot && !claimed.has(slot)
              ? slot
              : catalog.find((e) => !claimed.has(e.id))?.id;
          }
          if (pick) {
            nextEnabled.add(pick);
            claimed.add(pick);
          }
        }
        enabled = nextEnabled;
        seeded = true;
      }
      return { catalog, strengths: next, enabled, _seeded: seeded };
    }),
  setStrength: (id, value) =>
    set((s) => ({ strengths: { ...s.strengths, [id]: value } })),
  enable: (id) =>
    set((s) => {
      if (s.enabled.has(id)) return {} as Partial<LoraState>;
      const next = new Set(s.enabled);
      next.add(id);
      return { enabled: next };
    }),
  disable: (id) =>
    set((s) => {
      if (!s.enabled.has(id)) return {} as Partial<LoraState>;
      const next = new Set(s.enabled);
      next.delete(id);
      return { enabled: next };
    }),
  toggle: (id) =>
    set((s) => {
      const next = new Set(s.enabled);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { enabled: next };
    }),
  reset: () =>
    set({ catalog: [], strengths: {}, enabled: new Set(), _seeded: false }),
}));
