"use client";

import { create } from "zustand";

import type { DecodedFixture } from "@/engine/audio/loadFixture";

// In-memory cache for user-uploaded tracks. The decoded PCM lives in a
// non-reactive Map (Float32Array doesn't survive JSON / localStorage), and
// the names are mirrored into a reactive list so the fixture dropdown
// re-renders when an upload completes.
//
// Two kinds of entries:
//   - eager:      name + decoded PCM (a track the user just uploaded
//                 in this tab — decoded right after picking the file).
//   - placeholder: name only, no decoded buffer. Used to surface
//                 server-side persisted uploads (from /api/user_uploads)
//                 so the picker can render them on first paint after a
//                 page reload. The decoded PCM is fetched lazily by
//                 loadFixtureAudio on first Play / swap.
//
// `originalFiles` keeps the original encoded File alongside the decoded
// buffer so downstream consumers (e.g. demon-public-demo's saved-sessions
// feature, which uploads the original to a bucket on session save) can
// recover the bytes without having to re-prompt the user. Placeholder
// entries don't have it (the original lives on the pod).

interface CustomTracksState {
  /** Names in upload / discovery order. Reactive — components subscribe
   *  to this. Includes both eager and placeholder entries. */
  names: string[];
  /** Decoded buffers keyed by name. Read directly via getState() from
   *  non-React code (loadFixtureAudio); updates don't re-render.
   *  Placeholder entries are absent from this map until decoded. */
  decoded: Map<string, DecodedFixture>;
  /** Original encoded File keyed by name. Populated when add() is called
   *  with a File argument (the AudioSourceCrate upload path). Empty for
   *  placeholder entries. */
  originalFiles: Map<string, File>;

  add: (name: string, decoded: DecodedFixture, file?: File) => void;
  /** Register a name-only entry for a server-persisted upload. No-op if
   *  the name is already in the store (eager entries win — they carry
   *  the just-decoded buffer). */
  addPlaceholder: (name: string) => void;
  /** True iff the store has any entry (eager or placeholder) for this
   *  name. Used by add()'s collision-suffix loop. */
  has: (name: string) => boolean;
}

export const useCustomTracksStore = create<CustomTracksState>((set, get) => ({
  names: [],
  decoded: new Map(),
  originalFiles: new Map(),

  add: (name, decoded, file) =>
    set((s) => {
      const nextDecoded = new Map(s.decoded);
      nextDecoded.set(name, decoded);
      const nextOriginalFiles = new Map(s.originalFiles);
      if (file) nextOriginalFiles.set(name, file);
      const nextNames = s.names.includes(name) ? s.names : [...s.names, name];
      return {
        names: nextNames,
        decoded: nextDecoded,
        originalFiles: nextOriginalFiles,
      };
    }),

  addPlaceholder: (name) =>
    set((s) => {
      if (s.names.includes(name)) return s;
      return { ...s, names: [...s.names, name] };
    }),

  has: (name) => get().names.includes(name),
}));
