"use client";

import { useEffect, useRef } from "react";

import {
  applyLoraCapWithServerSync,
  resolveLoraCapForSource,
  useConfig,
} from "@/lib/config";
import { renderDeckMix } from "@/lib/audio/deckMixer";
import type { DeckTrackAssets } from "@/lib/audio/deckAssets";
import {
  DECK_IDS,
  type DeckId,
  type DeckSlot,
} from "@/store/useDeckStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { isTimeSignature } from "@/types/engine";

const DECK_SWAP_DEBOUNCE_MS = 900;

export function useDeckInferenceSync({
  decks,
  assetsByDeck,
  crossfade,
  enabled,
  revision,
}: {
  decks: Record<DeckId, DeckSlot>;
  assetsByDeck: Partial<Record<DeckId, DeckTrackAssets>>;
  crossfade: number;
  enabled: boolean;
  revision: number;
}) {
  const maxDurationS = useConfig().engine.max_source_duration_s ?? 120;
  const inFlightRef = useRef(false);
  const queuedRef = useRef(false);

  useEffect(() => {
    if (!enabled) return;
    const hasPlayableDeck = DECK_IDS.some((id) => {
      const deck = decks[id];
      return deck.trackName && deck.playing && assetsByDeck[id];
    });
    if (!hasPlayableDeck) return;

    const timer = window.setTimeout(() => {
      const run = async () => {
        const session = useSessionStore.getState();
        if (session.status !== "ready" || !session.remote || !session.player) {
          return;
        }
        if (inFlightRef.current) {
          queuedRef.current = true;
          return;
        }
        const mix = renderDeckMix({
          decks,
          assets: assetsByDeck,
          crossfade,
          maxDurationS,
        });
        if (!mix) return;

        inFlightRef.current = true;
        const { remote, player, setStatus } = useSessionStore.getState();
        if (!remote || !player) {
          inFlightRef.current = false;
          return;
        }
        setStatus("ready", `Mixing decks ${mix.activeDecks.join("+")}…`);

        const ok = await new Promise<boolean>((resolve) => {
          const onReady = (e: Event) => {
            remote.removeEventListener("swap_ready", onReady);
            remote.removeEventListener("swap_failed", onFail);
            const detail = (e as CustomEvent<{
              interleaved: Float32Array;
              channels: number;
              bpm?: number | null;
              key?: string;
              time_signature?: string;
            }>).detail;
            applyLoraCapWithServerSync(resolveLoraCapForSource(remote.duration));
            player.swap(detail.interleaved, detail.channels);
            player.seek(0);
            const perf = usePerformanceStore.getState();
            const detectedTs =
              detail.time_signature != null && isTimeSignature(detail.time_signature)
                ? detail.time_signature
                : null;
            if (detail.bpm != null || detail.key || detectedTs) {
              perf.setDetected(
                typeof detail.bpm === "number" ? detail.bpm : perf.detectedBpm,
                detail.key ?? perf.detectedKey,
                detectedTs ?? perf.detectedTimeSignature,
              );
            }
            resolve(true);
          };
          const onFail = (e: Event) => {
            remote.removeEventListener("swap_ready", onReady);
            remote.removeEventListener("swap_failed", onFail);
            const error = (e as CustomEvent).detail;
            setStatus("ready", `Deck mix failed: ${String(error || "swap failed")}`);
            resolve(false);
          };
          remote.addEventListener("swap_ready", onReady);
          remote.addEventListener("swap_failed", onFail);

          const perf = usePerformanceStore.getState();
          const sent = remote.sendSwapSource(
            mix.interleaved,
            mix.channels,
            perf.promptA,
            undefined,
            `deck-mix-${mix.activeDecks.join("")}.wav`,
            undefined,
            "full",
          );
          if (!sent) {
            remote.removeEventListener("swap_ready", onReady);
            remote.removeEventListener("swap_failed", onFail);
            resolve(false);
          }
        });

        inFlightRef.current = false;
        if (ok) useSessionStore.getState().setStatus("ready", "Playing");
        if (queuedRef.current) {
          queuedRef.current = false;
          // Schedule one follow-up after an in-flight swap settles; the
          // dependency-driven effect will cover further edits.
          window.setTimeout(run, DECK_SWAP_DEBOUNCE_MS);
        }
      };
      void run();
    }, DECK_SWAP_DEBOUNCE_MS);

    return () => window.clearTimeout(timer);
  }, [assetsByDeck, crossfade, decks, enabled, maxDurationS, revision]);
}
