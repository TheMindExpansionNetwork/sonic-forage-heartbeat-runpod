"use client";

import { create } from "zustand";

import type {
  DecodedFixture,
  DecodedStemAssets,
  StemSourceMode,
} from "@/engine/audio/loadFixture";

// In-memory cache for user-uploaded tracks. The decoded PCM lives in a
// non-reactive Map (Float32Array doesn't survive JSON / localStorage), and
// the names are mirrored into a reactive list so the fixture dropdown
// re-renders when an upload completes. Cleared on page reload — uploads
// are session-scoped, matching how the pod treats fixtures (it only ever
// sees the decoded PCM, never the file).
//
// `originalFiles` keeps the original encoded File alongside the decoded
// buffer so downstream consumers (e.g. demon-public-demo's saved-sessions
// feature, which uploads the original to a bucket on session save) can
// recover the bytes without having to re-prompt the user. It's a Map of
// name → File and lives next to `decoded` so adds stay atomic. Like
// `decoded`, it's non-reactive and read via getState().

interface CustomTracksState {
  /** Names in upload order. Reactive — components subscribe to this. */
  names: string[];
  /** Decoded buffers keyed by name. Read directly via getState() from
   *  non-React code (loadFixtureAudio); updates don't re-render. */
  decoded: Map<string, DecodedFixture>;
  /** Original encoded File keyed by name. Populated when add() is called
   *  with a File argument (the AudioSourceCrate upload path). May be
   *  empty for tracks added via other paths. */
  originalFiles: Map<string, File>;
  /** Which version of the uploaded track should feed model inference. */
  sourceModes: Map<string, StemSourceMode>;
  /** Model-ripped stems returned by the backend, keyed by upload name. */
  stems: Map<string, DecodedStemAssets>;
  stemStatuses: Map<string, "idle" | "processing" | "ready" | "failed">;
  stemErrors: Map<string, string>;

  add: (
    name: string,
    decoded: DecodedFixture,
    file?: File,
    sourceMode?: StemSourceMode,
  ) => void;
  setStemStatus: (
    name: string,
    status: "idle" | "processing" | "ready" | "failed",
    error?: string,
  ) => void;
  setSourceMode: (name: string, sourceMode: StemSourceMode) => void;
  setStems: (name: string, stems: DecodedStemAssets) => void;
  resolveSourceMode: (name: string) => StemSourceMode | undefined;
  has: (name: string) => boolean;
}

export const useCustomTracksStore = create<CustomTracksState>((set, get) => ({
  names: [],
  decoded: new Map(),
  originalFiles: new Map(),
  sourceModes: new Map(),
  stems: new Map(),
  stemStatuses: new Map(),
  stemErrors: new Map(),

  add: (name, decoded, file, sourceMode = "full") =>
    set((s) => {
      const nextDecoded = new Map(s.decoded);
      nextDecoded.set(name, decoded);
      const nextOriginalFiles = new Map(s.originalFiles);
      if (file) nextOriginalFiles.set(name, file);
      const nextSourceModes = new Map(s.sourceModes);
      nextSourceModes.set(name, sourceMode);
      const nextStemStatuses = new Map(s.stemStatuses);
      nextStemStatuses.set(name, "idle");
      const nextStemErrors = new Map(s.stemErrors);
      nextStemErrors.delete(name);
      const nextNames = s.names.includes(name) ? s.names : [...s.names, name];
      return {
        names: nextNames,
        decoded: nextDecoded,
        originalFiles: nextOriginalFiles,
        sourceModes: nextSourceModes,
        stemStatuses: nextStemStatuses,
        stemErrors: nextStemErrors,
      };
    }),

  setStemStatus: (name, status, error) =>
    set((s) => {
      const nextStemStatuses = new Map(s.stemStatuses);
      nextStemStatuses.set(name, status);
      const nextStemErrors = new Map(s.stemErrors);
      if (error) nextStemErrors.set(name, error);
      else nextStemErrors.delete(name);
      return {
        stemStatuses: nextStemStatuses,
        stemErrors: nextStemErrors,
      };
    }),

  setSourceMode: (name, sourceMode) =>
    set((s) => {
      const nextSourceModes = new Map(s.sourceModes);
      nextSourceModes.set(name, sourceMode);
      return { sourceModes: nextSourceModes };
    }),

  setStems: (name, stems) =>
    set((s) => {
      const nextStems = new Map(s.stems);
      nextStems.set(name, stems);
      const nextStemStatuses = new Map(s.stemStatuses);
      nextStemStatuses.set(name, "ready");
      const nextStemErrors = new Map(s.stemErrors);
      nextStemErrors.delete(name);
      return {
        stems: nextStems,
        stemStatuses: nextStemStatuses,
        stemErrors: nextStemErrors,
      };
    }),

  resolveSourceMode: (name) => {
    const state = get();
    const explicit = state.sourceModes.get(name);
    if (explicit) return explicit;
    return state.decoded.has(name) ? "full" : undefined;
  },

  has: (name) => get().decoded.has(name),
}));
