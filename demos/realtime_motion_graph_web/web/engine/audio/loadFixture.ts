// Fetch + decode an audio fixture from the DEMON pod, returning interleaved
// float32 PCM at the audio context sample rate. Uses Web Audio's
// decodeAudioData() so any WAV/MP3/FLAC the pod ships with works without a
// custom decoder.
//
// Also handles user-uploaded tracks: useCustomTracksStore caches their
// decoded buffers; loadFixtureAudio() checks that cache first, so the
// existing Play / fixture-swap paths work unchanged when the active
// fixture is an upload.

import { defaultWsUrl, podHttp } from "@/engine/podUrl";
import { SAMPLE_RATE } from "@/engine/protocol";
import { getApiKey } from "@/engine/rtmgConfig";

export interface DecodedFixture {
  interleaved: Float32Array;
  channels: number;
  frames: number;
  sampleRate: number;
}

// Server-side latent pool size (1920 * 5 = 9600 samples = 0.2 s at
// 48 kHz). backend.py and the sidecar precompute both align to this;
// trimming the decoded fixture to the same boundary keeps the runtime
// `samples` count matching the sidecar's recorded `samples` field, so
// `_try_load_sidecar` accepts the cached BPM / key / latents instead
// of falling back to live CNN detection.
const SAMPLE_POOL = 9600;

/** Decoder runs on a short-lived real AudioContext at SAMPLE_RATE so the
 *  PCM matches what the pod's pipeline expects. We previously used
 *  OfflineAudioContext here; recent Chromium builds occasionally never
 *  resolve OfflineAudioContext.decodeAudioData(), leaving the UI stuck on
 *  "Loading fixture…". A regular AudioContext is the documented path and
 *  is safe because Play is a user gesture. */
async function decodeArrayBuffer(bytes: ArrayBuffer): Promise<DecodedFixture> {
  const Ctx: typeof AudioContext =
    (window.AudioContext as typeof AudioContext) ||
    ((window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext as typeof AudioContext);
  const tmpCtx = new Ctx({ sampleRate: SAMPLE_RATE });
  let audioBuffer: AudioBuffer;
  try {
    // decodeAudioData mutates the input ArrayBuffer in some browsers, so
    // we pass a copy via .slice(0).
    audioBuffer = await tmpCtx.decodeAudioData(bytes.slice(0));
  } finally {
    void tmpCtx.close();
  }

  // Always emit exactly 2 channels: mono → duplicate, stereo → pass
  // through, >2 → take front L/R only (Web Audio puts front-L=0,
  // front-R=1 for any layout).
  const srcChannels = audioBuffer.numberOfChannels;
  const rawFrames = audioBuffer.length;
  const channels = 2;

  // Length normalize: trim to a multiple of the server's latent pool
  // (1920 * 5 = 9600 samples = 0.2 s at 48 kHz, mirroring backend.py's
  // `pool` and scripts/precompute_fixture_sidecars.py's POOL). Browsers'
  // decodeAudioData honours the mp3 encoder-padding header and returns
  // a non-pool-aligned sample count for many real-world files (e.g. a
  // 142.96 s mp3 with 23 ms of priming silence at the head). The
  // server-side VAE encode then computes a latent count off that ragged
  // tail and can underflow into a negative time dim — we saw
  // `Trying to create tensor with negative dimension -1: [1, 128, -1]`
  // on a track with exactly that shape. Pool alignment is what every
  // server step (VAE encode, sidecar samples field, TRT engine
  // selection) actually requires; aligning to whole seconds (the
  // previous rule) was strictly coarser and broke fixtures whose
  // natural pool-aligned length isn't a whole second — e.g. the lo-fi
  // loop is 57.6 s, so the whole-second trim shaved it to 57.0 s and
  // missed the sidecar lookup, falling back to CNN key detection.
  const sr = audioBuffer.sampleRate;
  const frames = Math.floor(rawFrames / SAMPLE_POOL) * SAMPLE_POOL;
  if (frames < sr) {
    throw new Error(
      `Audio too short — need ≥ 1 second, got ${(rawFrames / sr).toFixed(2)} s.`,
    );
  }

  const interleaved = new Float32Array(frames * channels);

  if (srcChannels === 1) {
    const m = audioBuffer.getChannelData(0);
    for (let i = 0; i < frames; i++) {
      const v = m[i];
      interleaved[i * 2] = v;
      interleaved[i * 2 + 1] = v;
    }
  } else {
    const l = audioBuffer.getChannelData(0);
    const r = audioBuffer.getChannelData(1);
    for (let i = 0; i < frames; i++) {
      interleaved[i * 2] = l[i];
      interleaved[i * 2 + 1] = r[i];
    }
  }

  return { interleaved, channels, frames, sampleRate: sr };
}

async function fetchAndDecode(url: string): Promise<DecodedFixture> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Fixture fetch failed: ${res.status} ${res.statusText}`);
  }
  const bytes = await res.arrayBuffer();
  return decodeArrayBuffer(bytes);
}

/** Fetch and decode, but treat a 404 as a miss (returns null) so the
 *  caller can try the next library. Any other non-2xx is a real error
 *  and propagates. Mirrors the server-side fallthrough in
 *  ``acestep.audio_clips.resolve_audio_clip``. */
async function fetchAndDecodeOptional(
  url: string,
): Promise<DecodedFixture | null> {
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`Fixture fetch failed: ${res.status} ${res.statusText}`);
  }
  const bytes = await res.arrayBuffer();
  return decodeArrayBuffer(bytes);
}

export async function loadFixtureAudio(name: string): Promise<DecodedFixture> {
  // Resolution order, mirroring the backend's resolve_audio_clip():
  //   1. in-memory custom-tracks store (just-uploaded eager decode)
  //   2. /user_uploads/<name> (persisted upload on the pod's disk)
  //   3. /fixtures/<name> (test fixture from the HF dataset)
  // We don't gate the /user_uploads attempt on listUserUploads() —
  // that would force every swap to wait for an extra HTTP roundtrip
  // when the catalog of names is already known to the server. Instead,
  // we try /user_uploads first and let a 404 fall through to /fixtures.
  // Lazy import to avoid a Zustand cycle at module load.
  const { useCustomTracksStore } = await import("@/store/useCustomTracksStore");
  const cached = useCustomTracksStore.getState().decoded.get(name);
  if (cached) return cached;

  const encoded = encodeURIComponent(name);
  const fromUpload = await fetchAndDecodeOptional(podHttp(`/user_uploads/${encoded}`));
  if (fromUpload) return fromUpload;
  return fetchAndDecode(podHttp(`/fixtures/${encoded}`));
}

// Cap user-supplied audio at DEMON's largest TRT engine profile
// (240 s; see acestep/paths.py:_TRT_ENGINE_PROFILES). Anything longer
// would fail server-side at session init regardless of WS frame size.
// The server's websockets.serve(max_size=...) is sized to fit this
// duration with a comfortable margin.
export const MAX_FIXTURE_DURATION_S = 240;

export interface DecodeFileResult {
  decoded: DecodedFixture;
  /** True iff the input was longer than MAX_FIXTURE_DURATION_S and we
   *  trimmed the head. The UI surfaces this so users know the upload
   *  was clipped. */
  wasTrimmed: boolean;
}

/** Soft-trim to fit DEMON's swap-source limit. Tracks ≤ 240 s pass
 *  through unchanged; longer tracks are clipped to the largest
 *  pool-aligned length ≤ 240 s. Pool alignment matches the rule in
 *  decodeArrayBuffer (multiple of SAMPLE_POOL = 9600), so the trimmed
 *  buffer still satisfies backend.py's VAE-encode constraint. */
function trimToSwapLimit(decoded: DecodedFixture): DecodeFileResult {
  const seconds = decoded.frames / decoded.sampleRate;
  if (seconds <= MAX_FIXTURE_DURATION_S) return { decoded, wasTrimmed: false };

  const maxFramesRaw = MAX_FIXTURE_DURATION_S * decoded.sampleRate;
  const targetFrames = Math.floor(maxFramesRaw / SAMPLE_POOL) * SAMPLE_POOL;
  const trimmed = new Float32Array(targetFrames * decoded.channels);
  trimmed.set(decoded.interleaved.subarray(0, targetFrames * decoded.channels));
  return {
    decoded: {
      interleaved: trimmed,
      channels: decoded.channels,
      frames: targetFrames,
      sampleRate: decoded.sampleRate,
    },
    wasTrimmed: true,
  };
}

/** Decode a user-supplied audio File (mp3, wav, flac, ogg — anything the
 *  browser supports). Used by the upload affordances.
 *  Auto-trims to MAX_FIXTURE_DURATION_S when the source is longer; the UI
 *  shows a "we trimmed your upload" message when wasTrimmed is true. */
export async function decodeAudioFile(file: File): Promise<DecodeFileResult> {
  const bytes = await file.arrayBuffer();
  const decoded = await decodeArrayBuffer(bytes);
  return trimToSwapLimit(decoded);
}

/** Fetch the pod's whitelist of fixture names. */
export async function listFixtures(): Promise<string[]> {
  const res = await fetch(podHttp("/api/fixtures"));
  if (!res.ok) throw new Error(`Fixture list failed: ${res.status}`);
  const json = (await res.json()) as string[];
  return json;
}

/** Fetch the pod's list of persisted user uploads. Same shape as
 *  listFixtures so the picker can merge both lists into one. */
export async function listUserUploads(): Promise<string[]> {
  const res = await fetch(podHttp("/api/user_uploads"));
  if (!res.ok) throw new Error(`User-upload list failed: ${res.status}`);
  const json = (await res.json()) as string[];
  return json;
}

export interface UploadTrackResult {
  /** Server-canonical name (may differ from file.name after sanitisation
   *  or collision-suffixing). Use this as fixture_name for subsequent
   *  WS session inits — the server's sidecar lookup matches by exactly
   *  this string. */
  name: string;
  bpm: number;
  key: string;
  timeSignature: string;
  durationS: number;
  samples: number;
}

/**
 * Upload a track to the pod and run the sidecar precompute synchronously.
 *
 * Wire format (matches backend.py:_handle_upload_track):
 *   client -> {"type":"upload_track","name":"<filename>"}
 *   client -> raw encoded audio bytes (mp3/wav/flac/ogg/m4a)
 *   server -> {"type":"upload_ok",...} (or "upload_failed")
 *
 * Uses a dedicated WebSocket (not the streaming session's) so uploads
 * work before / between Play clicks. Server runs Session.prepare_source
 * + BPM + key detection on its eager encoder Session and writes the
 * sidecar before returning. From that point on, any session that opens
 * with this name as fixture_name hits the sidecar fast path
 * identically to test fixtures.
 */
/** Mirror of useStartSession.ts:resolveWsUrl so uploads share the
 *  streaming session's URL when one has been issued by queue admit.
 *  Pre-Play (no session yet), defaultWsUrl is the right thing for
 *  DEMON local; daydream-public's fork handles queue admit. */
async function resolveUploadWsUrl(): Promise<string> {
  let url = defaultWsUrl();
  // Lazy import to avoid pulling Zustand into the module graph of the
  // RemoteBackend → loadFixture.ts → useSessionStore triangle.
  const { useSessionStore } = await import("@/store/useSessionStore");
  const serverUrl = useSessionStore.getState().wsUrl;
  if (serverUrl) url = serverUrl;
  const apiKey = getApiKey();
  if (apiKey) {
    const sep = url.includes("?") ? "&" : "?";
    url = `${url}${sep}apiKey=${encodeURIComponent(apiKey)}`;
  }
  return url;
}

export async function uploadTrackToServer(
  file: File,
): Promise<UploadTrackResult> {
  const wsUrl = await resolveUploadWsUrl();
  return new Promise((resolve, reject) => {
    let settled = false;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      try {
        ws.close();
      } catch {}
      fn();
    };

    ws.onopen = () => {
      try {
        ws.send(JSON.stringify({ type: "upload_track", name: file.name }));
        // Read the encoded file bytes and forward as one binary frame.
        // The server's max_size (100 MiB) is sized for the streaming
        // PCM path; any browser-decodable encoded file fits well under
        // that ceiling.
        file.arrayBuffer().then(
          (buf) => {
            try {
              ws.send(buf);
            } catch (e) {
              finish(() => reject(e instanceof Error ? e : new Error(String(e))));
            }
          },
          (e) => finish(() => reject(e instanceof Error ? e : new Error(String(e)))),
        );
      } catch (e) {
        finish(() => reject(e instanceof Error ? e : new Error(String(e))));
      }
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(ev.data) as Record<string, unknown>;
      } catch {
        return;
      }
      if (msg.type === "upload_ok") {
        finish(() =>
          resolve({
            name: String(msg.name),
            bpm: Number(msg.bpm ?? 0),
            key: String(msg.key ?? ""),
            timeSignature: String(msg.time_signature ?? "4"),
            durationS: Number(msg.duration_s ?? 0),
            samples: Number(msg.samples ?? 0),
          }),
        );
      } else if (msg.type === "upload_failed") {
        finish(() => reject(new Error(String(msg.error ?? "upload failed"))));
      }
    };

    ws.onerror = () => {
      finish(() => reject(new Error("upload connection failed")));
    };

    ws.onclose = (ev) => {
      if (settled) return;
      finish(() =>
        reject(new Error(ev.reason || `upload connection closed (${ev.code})`)),
      );
    };
  });
}

// Preferred default the UI picks when no fixture is yet selected. Falls
// back to names[0] if the catalog doesn't contain it (e.g. removed from
// KNOWN_FIXTURES upstream).
export const PREFERRED_DEFAULT_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav";

export function pickDefaultFixture(names: readonly string[]): string {
  if (names.includes(PREFERRED_DEFAULT_FIXTURE)) return PREFERRED_DEFAULT_FIXTURE;
  return names[0] ?? "";
}
