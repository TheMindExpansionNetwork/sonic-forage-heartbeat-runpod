"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";

import {
  decodeAudioFile,
  listFixtures,
  pickDefaultFixture,
  type DecodedFixture,
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import { useDeckAssets } from "@/hooks/useDeckAssets";
import { useDeckInferenceSync } from "@/hooks/useDeckInferenceSync";
import { useDeckMonitor } from "@/hooks/useDeckMonitor";
import { useSeedUserUploads } from "@/hooks/useSeedUserUploads";
import { commitUploadedTrack } from "@/lib/audio/commitUploadedTrack";
import { deckAssetSource } from "@/lib/audio/deckAssets";
import { trimAudioBuffer } from "@/lib/audio/trimAudioBuffer";
import { useConfig } from "@/lib/config";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import {
  DECK_IDS,
  MAX_DECKS,
  useDeckStore,
  type DeckId,
} from "@/store/useDeckStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { TimeSignature } from "@/types/engine";

import { AlmostReadyDialog } from "./AlmostReadyDialog";
import { RefSelect, type RefSelectGroup } from "./RefSelect";
import { WaveformTrimDialog } from "./WaveformTrimDialog";

const DEFAULT_TRIM_CAP_S = 120;
const SOURCE_PARTS: StemSourceMode[] = ["full", "vocals", "instruments"];

export function DecksPanel() {
  const decks = useDeckStore((s) => s.decks);
  const deckIds = useDeckStore((s) => s.deckIds);
  const inputDeckId = useDeckStore((s) => s.inputDeckId);
  const timbreDeckId = useDeckStore((s) => s.timbreDeckId);
  const structureDeckId = useDeckStore((s) => s.structureDeckId);
  const crossfade = useDeckStore((s) => s.crossfade);
  const monitorEnabled = useDeckStore((s) => s.monitorEnabled);
  const inferenceEnabled = useDeckStore((s) => s.inferenceEnabled);
  const revision = useDeckStore((s) => s.mixRevision);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  const activeFixture = usePerformanceStore((s) => s.fixture);
  const timbreName = usePerformanceStore((s) => s.timbreName);
  const structName = usePerformanceStore((s) => s.structName);
  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);
  const [fixtures, setFixtures] = useState<string[]>([]);
  const { assetsByDeck, statuses, errors } = useDeckAssets(decks);
  const trimCapS =
    useConfig().engine.max_source_duration_s ?? DEFAULT_TRIM_CAP_S;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadDeckRef = useRef<DeckId>("A");
  const [uploadingDeckId, setUploadingDeckId] = useState<DeckId | null>(null);
  const [trimming, setTrimming] = useState<{
    deckId: DeckId;
    decoded: DecodedFixture;
    fileName: string;
    originalFile: File;
  } | null>(null);
  const [pending, setPending] = useState<{
    deckId: DeckId;
    decoded: DecodedFixture;
    fileName: string;
    originalFile: File;
  } | null>(null);

  useSeedUserUploads();
  useDeckMonitor({ decks, assetsByDeck, crossfade, enabled: monitorEnabled });
  useDeckInferenceSync({
    decks,
    assetsByDeck,
    crossfade,
    enabled: inferenceEnabled,
    revision,
  });

  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listFixtures()
      .then((names) => {
        setFixtures(names);
        const defaultTrack = activeFixture || pickDefaultFixture(names);
        if (defaultTrack) {
          if (!activeFixture) usePerformanceStore.getState().setFixture(defaultTrack);
          useDeckStore.getState().ensureInitialDeck(defaultTrack);
        }
      })
      .catch(() => setFixtures([]));
  }, [activeFixture, sessionWsUrl]);

  useEffect(() => {
    const defaultTrack = activeFixture || pickDefaultFixture(fixtures) || customNames[0];
    if (defaultTrack) {
      useDeckStore.getState().ensureInitialDeck(defaultTrack);
    }
  }, [activeFixture, customNames, fixtures]);

  async function onFilePicked(deckId: DeckId, file: File) {
    const { setStatus } = useSessionStore.getState();
    setUploadingDeckId(deckId);
    setStatus(useSessionStore.getState().status, "");
    try {
      const decoded = await decodeAudioFile(file);
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      setTrimming({ deckId, decoded, fileName: chosen, originalFile: file });
      setStatus(useSessionStore.getState().status, "");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Deck ${deckId}: ${msg}`);
    } finally {
      setUploadingDeckId(null);
    }
  }

  function onTrimConfirm(startS: number, endS: number) {
    if (!trimming) return;
    const trimmed = trimAudioBuffer(trimming.decoded, startS, endS);
    setPending({
      deckId: trimming.deckId,
      decoded: trimmed,
      fileName: trimming.fileName,
      originalFile: trimming.originalFile,
    });
    setTrimming(null);
  }

  async function commitPending(
    keyOverride: string | null,
    timeSignatureOverride: TimeSignature | null,
    sourceMode: StemSourceMode,
  ) {
    if (!pending) return;
    await commitUploadedTrack({
      pending,
      keyOverride,
      timeSignatureOverride,
      sourceMode,
      addCustomTrack,
      setFixture: (name) =>
        useDeckStore.getState().setTrack(pending.deckId, name),
      setPending: (next) =>
        setPending(next ? { ...next, deckId: pending.deckId } : null),
      setUploading: (next) => setUploadingDeckId(next ? pending.deckId : null),
    });
  }

  function shortTrackName(name: string | null): string {
    if (!name) return "Loading track";
    return name.replace(/\.(wav|mp3|flac|ogg|m4a|aac)$/i, "");
  }

  function deckDurationLabel(id: DeckId): string {
    const assets = assetsByDeck[id];
    if (!assets) return "asset pending";
    const seconds = assets.full.frames / assets.full.sampleRate;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds - mins * 60).toString().padStart(2, "0");
    const stemCount = Number(Boolean(assets.stems.vocals)) + Number(Boolean(assets.stems.instruments));
    return `${mins}:${secs} · ${stemCount}/2 stems`;
  }

  function roleSummary(input: boolean, timbre: boolean, structure: boolean): string {
    const roles = [
      input ? "input" : null,
      timbre ? "timbre" : null,
      structure ? "structure" : null,
    ].filter(Boolean);
    return roles.length ? ` · ${roles.join(" · ")}` : "";
  }

  function setDeckTrack(id: DeckId, name: string): void {
    useDeckStore.getState().setTrack(id, name);
    if (useDeckStore.getState().inputDeckId === id) {
      usePerformanceStore.getState().setFixture(name);
    }
  }

  function assignInputDeck(id: DeckId): void {
    const name = useDeckStore.getState().decks[id].trackName;
    if (!name) return;
    useDeckStore.getState().setInputDeck(id);
    usePerformanceStore.getState().setFixture(name);
  }

  function clearReference(kind: "timbre" | "structure"): void {
    const session = useSessionStore.getState();
    if (kind === "timbre") {
      session.remote?.sendClearTimbreSource();
      usePerformanceStore.getState().setTimbreRef(null);
      useDeckStore.getState().setTimbreDeck(null);
    } else {
      session.remote?.sendClearStructureSource();
      usePerformanceStore.getState().setStructRef(null);
      useDeckStore.getState().setStructureDeck(null);
    }
    session.setStatus(session.status, `${kind} follows input`);
  }

  function applyReferenceDeck(kind: "timbre" | "structure", id: DeckId): void {
    const currentRole = kind === "timbre" ? timbreDeckId : structureDeckId;
    if (currentRole === id) {
      clearReference(kind);
      return;
    }

    const deck = useDeckStore.getState().decks[id];
    const name = deck.trackName;
    if (!name) return;
    const session = useSessionStore.getState();
    if (session.status !== "ready" || !session.remote) {
      session.setStatus(session.status, `Start playback before assigning ${kind}`);
      return;
    }

    if (fixtures.includes(name)) {
      if (kind === "timbre") {
        session.remote.sendSetTimbreFixture(name);
        usePerformanceStore.getState().setTimbreRef({ mode: "fixture", name });
        useDeckStore.getState().setTimbreDeck(id);
      } else {
        session.remote.sendSetStructureFixture(name);
        usePerformanceStore.getState().setStructRef({ mode: "fixture", name });
        useDeckStore.getState().setStructureDeck(id);
      }
      session.setStatus("ready", `Loading ${kind} ${name}…`);
      return;
    }

    const assets = assetsByDeck[id];
    if (!assets) {
      session.setStatus("ready", `Deck ${id} is still loading ${name}`);
      return;
    }
    const ok =
      kind === "timbre"
        ? session.remote.sendSetTimbreSource(
            assets.full.interleaved,
            assets.full.channels,
            name,
          )
        : session.remote.sendSetStructureSource(
            assets.full.interleaved,
            assets.full.channels,
            name,
          );
    if (ok) {
      if (kind === "timbre") {
        usePerformanceStore.getState().setTimbreRef({ mode: "clip", name });
        useDeckStore.getState().setTimbreDeck(id);
      } else {
        usePerformanceStore.getState().setStructRef({ mode: "clip", name });
        useDeckStore.getState().setStructureDeck(id);
      }
      session.setStatus("ready", `Loading ${kind} ${name}…`);
    } else {
      session.setStatus("ready", `${kind} reference failed`);
    }
  }

  const trackGroups: RefSelectGroup[] = [
    {
      label: "Library",
      options: fixtures.map((name) => ({ value: name, label: name })),
    },
    {
      label: "Your tracks",
      options: customNames.map((name) => ({ value: name, label: name })),
    },
  ];
  const defaultAddTrack = activeFixture || pickDefaultFixture(fixtures) || customNames[0];
  const leftDecks = DECK_IDS.filter(
    (id) => deckIds.includes(id) && decks[id].crossfadeSide === "left",
  );
  const rightDecks = DECK_IDS.filter(
    (id) => deckIds.includes(id) && decks[id].crossfadeSide === "right",
  );

  return (
    <section className="decks-panel" aria-label="Deck mixer">
      <div className="decks-panel-head">
        <div>
          <span className="decks-panel-kicker">Input mixer</span>
          <h2 className="decks-panel-title">Decks</h2>
        </div>
        <div className="decks-panel-toggles">
          <button
            type="button"
            className={monitorEnabled ? "is-active" : ""}
            data-dd-tooltip="Hear the deck mix directly in the browser. This is instant and separate from the generated model output."
            onClick={() =>
              useDeckStore.getState().setMonitorEnabled(!monitorEnabled)
            }
          >
            Monitor
          </button>
          <button
            type="button"
            className={inferenceEnabled ? "is-active" : ""}
            data-dd-tooltip="Send debounced deck-mix snapshots to inference through the existing source-swap path."
            onClick={() =>
              useDeckStore.getState().setInferenceEnabled(!inferenceEnabled)
            }
          >
            Infer
          </button>
          <button
            type="button"
            disabled={deckIds.length >= MAX_DECKS || !defaultAddTrack}
            data-dd-tooltip="Add another populated deck using the current/default track. You can switch its track immediately after adding."
            onClick={() => {
              if (defaultAddTrack) useDeckStore.getState().addDeck(defaultAddTrack);
            }}
          >
            Add deck
          </button>
        </div>
      </div>

      <div
        className="decks-scene-crossfader"
        data-dd-tooltip="Crossfade between the left and right deck buses. Assign any number of decks to either side, then move this fader like a DJ mixer."
        data-dd-tooltip-wide=""
        data-dd-tooltip-title="Deck Crossfader"
      >
        <div className="decks-scene-bus">
          <span className="decks-scene-label">Left scene</span>
          <span className="decks-scene-decks">
            {leftDecks.length ? leftDecks.join(" ") : "empty"}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.001}
          value={crossfade}
          onChange={(e) =>
            useDeckStore.getState().setCrossfade(Number(e.target.value))
          }
          aria-label="Deck crossfader"
        />
        <div className="decks-scene-bus decks-scene-bus--right">
          <span className="decks-scene-label">Right scene</span>
          <span className="decks-scene-decks">
            {rightDecks.length ? rightDecks.join(" ") : "empty"}
          </span>
        </div>
      </div>

      <div className="deck-list">
        {deckIds.map((id) => {
          const deck = decks[id];
          const trackStatus = deck.trackName ? statuses[deck.trackName] : "idle";
          const trackError = deck.trackName ? errors[deck.trackName] : undefined;
          const assets = assetsByDeck[id];
          const activeSource = deckAssetSource(assets, deck.sourcePart);
          const missingSelectedStem =
            deck.sourcePart !== "full" && assets && !activeSource;
          const uploadingThisDeck = uploadingDeckId === id;
          const deckBusy = uploadingThisDeck || trackStatus === "loading";
          const isInput = inputDeckId === id;
          const isTimbre = timbreDeckId === id || timbreName === deck.trackName;
          const isStructure =
            structureDeckId === id || structName === deck.trackName;
          return (
            <article
              key={id}
              className={`deck-card${deckBusy ? " is-loading" : ""}`}
              style={{ "--deck-color": deck.color } as CSSProperties}
              aria-busy={deckBusy}
            >
              <div className="deck-card-head">
                <span className="deck-badge">{id}</span>
                <div className="deck-title-block">
                  <span className="deck-name">Deck {id}</span>
                  <strong className="deck-track-title" title={deck.trackName ?? ""}>
                    {shortTrackName(deck.trackName)}
                  </strong>
                  <span className="deck-track-meta">{deckDurationLabel(id)}</span>
                </div>
                <button
                  type="button"
                  className="deck-remove-button"
                  disabled={deckIds.length <= 1}
                  data-dd-tooltip="Remove this deck. At least one loaded deck always remains."
                  onClick={() => useDeckStore.getState().removeDeck(id)}
                >
                  ×
                </button>
              </div>

              <div className="deck-track-picker">
                <RefSelect
                  label="replace"
                  value={deck.trackName ?? ""}
                  pinned={[]}
                  groups={trackGroups}
                  onSelect={(value) => value && setDeckTrack(id, value)}
                  disabled={deckBusy}
                  ariaLabel={`Deck ${id} track`}
                  onUpload={() => {
                    uploadDeckRef.current = id;
                    fileInputRef.current?.click();
                  }}
                  uploadLabel={
                    uploadingThisDeck ? "Decoding…" : `Upload audio to deck ${id}`
                  }
                  tooltip="Replace this deck's track. If this deck owns the input role, the model input swaps to the new track as well."
                />
              </div>

              {deckBusy && (
                <div className="deck-loading-strip" aria-hidden="true">
                  <span />
                </div>
              )}

              <div className="deck-role-row" aria-label={`Deck ${id} reference roles`}>
                <button
                  type="button"
                  className={isInput ? "is-active" : ""}
                  data-dd-tooltip="Make this deck the primary input track. Replacing this deck replaces the model input."
                  onClick={() => assignInputDeck(id)}
                >
                  Input
                </button>
                <button
                  type="button"
                  className={isTimbre ? "is-active" : ""}
                  disabled={deckBusy}
                  data-dd-tooltip="Use this deck's full track as the timbre reference. Click again to return timbre to input."
                  onClick={() => applyReferenceDeck("timbre", id)}
                >
                  Timbre
                </button>
                <button
                  type="button"
                  className={isStructure ? "is-active" : ""}
                  disabled={deckBusy}
                  data-dd-tooltip="Use this deck's full track as the structure reference. Click again to return structure to input."
                  onClick={() => applyReferenceDeck("structure", id)}
                >
                  Structure
                </button>
              </div>

              <div
                className="deck-source-parts"
                role="radiogroup"
                data-dd-tooltip="Choose which version of this deck feeds the deck mix: full track, vocal stem, or instrumental stem. Stem options enable when cached RoFormer assets exist."
                data-dd-tooltip-wide=""
                data-dd-tooltip-title={`Deck ${id} source`}
              >
                {SOURCE_PARTS.map((part) => {
                  const stemPart =
                    part === "vocals" || part === "instruments" ? part : null;
                  const disabled =
                    stemPart !== null &&
                    assets !== undefined &&
                    !assets.stems[stemPart];
                  return (
                    <button
                      key={part}
                      type="button"
                      role="radio"
                      aria-checked={deck.sourcePart === part}
                      className={deck.sourcePart === part ? "is-active" : ""}
                      disabled={disabled}
                      onClick={() =>
                        useDeckStore.getState().setSourcePart(id, part)
                      }
                    >
                      {part === "instruments" ? "instr" : part}
                    </button>
                  );
                })}
              </div>

              <div className="deck-performance-row">
                <button
                  type="button"
                  disabled={!deck.trackName}
                  data-dd-tooltip={
                    deck.playing
                      ? "Pause this deck's local transport. The deck mix updates inference after a short debounce."
                      : "Start this deck from its current playhead."
                  }
                  onClick={() =>
                    useDeckStore.getState().setPlaying(id, !deck.playing)
                  }
                >
                  {deck.playing ? "Pause" : "Play"}
                </button>
                <button
                  type="button"
                  disabled={!deck.trackName}
                  data-dd-tooltip="Jump this deck back to its cue point. The default cue is the start of the song."
                  onClick={() => useDeckStore.getState().jumpToCue(id)}
                >
                  Cue
                </button>
                <button
                  type="button"
                  className={deck.muted ? "is-active" : ""}
                  data-dd-tooltip="Mute this deck in both the monitor and the mixed inference source."
                  onClick={() => useDeckStore.getState().toggleMuted(id)}
                >
                  Mute
                </button>
                <div
                  className="deck-side-switch"
                  role="radiogroup"
                  aria-label={`Deck ${id} crossfader side`}
                >
                  {(["left", "right"] as const).map((side) => (
                    <button
                      key={side}
                      type="button"
                      role="radio"
                      aria-checked={deck.crossfadeSide === side}
                      className={deck.crossfadeSide === side ? "is-active" : ""}
                      data-dd-tooltip={`Assign deck ${id} to the ${side} scene. Multiple decks can share the same side.`}
                      onClick={() =>
                        useDeckStore.getState().setCrossfadeSide(id, side)
                      }
                    >
                      {side === "left" ? "L" : "R"}
                    </button>
                  ))}
                </div>
              </div>

              <div
                className="deck-fader-row"
                data-dd-tooltip="Deck level before the scene crossfader."
              >
                <span>Level</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.001}
                  value={deck.volume}
                  onChange={(e) =>
                    useDeckStore
                      .getState()
                      .setVolume(id, Number(e.target.value))
                  }
                  aria-label={`Deck ${id} volume`}
                />
              </div>

              <div className="deck-status">
                {uploadingThisDeck
                  ? `Deck ${id}: decoding upload…`
                  : trackStatus === "loading"
                  ? `Deck ${id}: loading ${shortTrackName(deck.trackName)}…`
                  : trackStatus === "failed"
                    ? `Failed: ${trackError ?? "track unavailable"}`
                    : missingSelectedStem
                      ? "Stem unavailable"
                      : deck.trackName
                        ? `${deck.sourcePart} ready${roleSummary(isInput, isTimbre, isStructure)}`
                        : "Loading library…"}
              </div>
            </article>
          );
        })}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) void onFilePicked(uploadDeckRef.current, file);
        }}
      />

      {trimming && (
        <WaveformTrimDialog
          decoded={trimming.decoded}
          fileName={trimming.fileName}
          capS={trimCapS}
          onConfirm={onTrimConfirm}
          onCancel={() => setTrimming(null)}
        />
      )}
      {pending && (
        <AlmostReadyDialog
          fileName={pending.fileName}
          wasTrimmed={false}
          defaultKey={usePerformanceStore.getState().activeKey}
          defaultTimeSignature={
            usePerformanceStore.getState().activeTimeSignature
          }
          onContinue={({ keyOverride, timeSignatureOverride, sourceMode }) =>
            commitPending(keyOverride, timeSignatureOverride, sourceMode)
          }
          onPickAnother={() => {
            setPending(null);
            setTimeout(() => fileInputRef.current?.click(), 0);
          }}
          onClose={() => setPending(null)}
        />
      )}
    </section>
  );
}
