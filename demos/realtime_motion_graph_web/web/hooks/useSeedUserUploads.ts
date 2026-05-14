"use client";

import { useEffect } from "react";

import { listUserUploads } from "@/engine/audio/loadFixture";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { useSessionStore } from "@/store/useSessionStore";

/**
 * Mount effect that fetches the pod's list of server-persisted user
 * uploads and registers placeholder entries in the custom-tracks
 * store. Components that surface a track picker (AudioSourceCrate,
 * LiteTrackCarousel, OperatorStrip, RefControl) call this once so the
 * picker shows persisted uploads on first paint after a page reload,
 * before any of them have been played / decoded.
 *
 * Decoded PCM is fetched lazily by loadFixtureAudio when the user
 * actually plays one of these names — placeholder entries don't carry
 * a buffer.
 *
 * Mirrors the queue-admit gate the fixture-list fetch uses: in
 * daydream-public production `/api/*` on the pod returns 401 until
 * the queue admits the client, so we wait for `wsUrl` to populate
 * before firing. Standalone DEMON has no queue (LOCAL_MODE), so we
 * skip the wait there.
 *
 * Idempotent: ``addPlaceholder`` is a no-op when the name is already
 * in the store, so multiple components mounting this hook is fine
 * (and is the expected case — each picker mounts independently).
 */
export function useSeedUserUploads(): void {
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listUserUploads()
      .then((names) => {
        const store = useCustomTracksStore.getState();
        for (const n of names) {
          if (!store.has(n)) store.addPlaceholder(n);
        }
      })
      .catch(() => {});
  }, [sessionWsUrl]);
}
