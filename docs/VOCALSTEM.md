# Vocal And Instrumental Stem Extraction

This document describes how uploaded-track stem extraction works in the
`realtime_motion_graph_web` backend.

The implementation lives in `demos/realtime_motion_graph_web/backend.py`. The
demo UI and backend now resolve every user-uploaded audio source to a stem mode
(`full` by default), so uploads are stemmed automatically. Built-in fixtures
still omit `stem_source_mode` and skip the stem path.

## When Stem Extraction Runs

The frontend can send one of three source modes:

- `full`: keep the original upload as the inference source, but still generate
  vocal and instrumental overlay assets.
- `vocals`: generate stems, then use the vocal stem as the inference source.
- `instruments`: generate stems, then use the instrumental bed as the inference
  source.

Backend validation is handled by `_normalize_stem_source_mode()`. The frontend
uses `full` as the fallback for custom uploads, so the selector controls only
which waveform feeds inference; it does not gate whether stems are generated.

For initial session setup, the extraction happens after the upload has been
decoded, trimmed/profile-aligned, and prepared with `Session.prepare_source()`.
For source swaps, the same extraction path runs inside `apply_swap_if_pending()`
after the new uploaded waveform has been decoded and prepared.

## Vocal Extraction

Vocal extraction is performed with ACE-Step's native `extract` task through
`_ace_extract_track()`.

The backend builds one ACE extract instruction:

```text
Extract the VOCALS track from the audio:
```

The text conditioning is built with `Session.encode_text()` and the actual
generation goes through `Session.generate()`. This is intentional: the realtime
backend often runs the decoder through TensorRT, so the PyTorch decoder weights
may not be loaded. Routing through `Session.generate()` keeps stem extraction on
the currently selected checkpoint and backend. For example, if the server is
launched with the XL checkpoint, stem extraction uses that XL session instead of
loading a separate base model.

The important generation inputs are:

- `refer_latent`: the raw VAE source latent from `PreparedSource.latent`, used
  as the timbre reference while encoding conditioning.
- `context_latent`: the same raw source latent. This is important: ACE extract
  uses raw source latents as context. The semantic-hint cover path is for
  cover-style regeneration and makes extract output quiet/incorrect here.
- `chunk_mask`: omitted, which causes `Session.generate()` to build an all-ones
  mask matching the context latent shape.
- `guidance_curve`: a constant CFG curve at `7.5`.
- `steps`: `20` for turbo checkpoints, and at least `50` for non-turbo
  checkpoints.
- `infer_method`: `ode`.
- `shift`: `1.0`.
- `dcw_enabled`: `False`, so the stem pass does not apply the realtime DCW
  correction.

The decoded output is normalized back to the upload waveform shape with
`_fit_stem_waveform()`, which fixes batch/channel/length differences and
replaces non-finite values.

The backend extracts only `vocals`. That same stem is sent to the frontend as
the vocal overlay and is also used as the guide for instrumental suppression.

## Why Instrumentals Are Not ACE-Extracted Directly

ACE-Step's native extract task supports named track classes such as:

- `vocals`
- `backing_vocals`
- `drums`
- `bass`
- `guitar`
- `keyboard`
- `strings`
- `percussion`
- `synth`
- `fx`
- `brass`
- `woodwinds`

It does not expose an `instrumental` or "everything except vocals" class. Asking
ACE to extract `INSTRUMENTAL` is out-of-distribution and can return a mix that
still contains vocals.

The backend also does not use simple time-domain subtraction:

```text
instrumental = original - vocals
```

ACE's vocal output can sound perceptually correct, but it is a generated /
reconstructed stem, not the exact phase-aligned vocal waveform that was summed
into the original master. Direct subtraction therefore does not null the vocal
cleanly and can leave obvious vocal leakage.

## Instrumental Generation

The instrumental bed is created by `_spectral_vocal_suppress()`.

This function treats ACE's `vocals` output as a guide for where vocal content
lives in the time-frequency domain. It keeps the original mix phase and
attenuates vocal-dominant STFT bins instead of subtracting the generated vocal
waveform.

For each channel:

1. Compute the complex STFT of the original uploaded mix.
2. Compute STFT magnitudes for the vocal guide stems.
3. Combine the guides by taking the per-bin maximum magnitude.
4. Estimate a scale between the guide magnitude and the original mix magnitude
   using active vocal bins.
5. Build a vocal mask:

   ```text
   vocal_mask = scaled_guide_magnitude / original_mix_magnitude
   ```

6. Clamp the mask to avoid fully deleting bins (`max_mask = 0.985`).
7. Convert the mask into a keep mask:

   ```text
   keep = clamp(1.0 - 1.35 * vocal_mask, 0.0, 1.0)
   ```

8. Apply the keep mask to the original complex STFT.
9. Reconstruct the instrumental with inverse STFT.

Current mask settings:

- `n_fft = 4096`
- `hop_length = 1024`
- `strength = 1.35`
- `max_mask = 0.985`

This produces an instrumental that is still based on the original upload, but
with vocal-dominant regions attenuated using ACE's extracted vocal estimates.

## Returned Stem Assets

`_extract_upload_stems()` returns:

```python
{
    "vocals": vocals,
    "instruments": instruments,
}
```

If the user selected `vocals` or `instruments` as `stem_source_mode`, the
backend prepares that selected waveform as a new `Audio` source and reruns
`Session.prepare_source()` so inference uses the selected stem.

The stem overlay assets are sent to the client with `_send_stem_payload()`:

1. A JSON message of type `stem_assets` with:
   - `fixture_name`
   - `sample_rate`
   - `channels`
   - `frames`
   - `stems`: `["vocals", "instruments"]`
   - `source_mode`
2. Two binary payloads, one per stem, in the same order.

The binary payloads are interleaved `float16` PCM buffers shaped as
`[frames, channels]` on the wire.

If extraction fails and the requested inference source depends on the failed
stem, the backend fails the session or swap. If extraction fails while the full
track is still usable, the backend sends a `stem_failed` message and continues
with the original source.

## Known Limitations

The instrumental stem is vocal-suppressed, not a perfect studio instrumental.
It depends on how accurately ACE's vocal estimate identifies vocal energy.
Strong vocal reverb, doubled vocals, backing vocals, or vocal-like synths can
still leak or be over-suppressed.

The implementation intentionally uses ACE-Step for vocal identification and a
spectral suppression pass for the instrumental complement because ACE-Step does
not provide a native all-non-vocal extract class.
