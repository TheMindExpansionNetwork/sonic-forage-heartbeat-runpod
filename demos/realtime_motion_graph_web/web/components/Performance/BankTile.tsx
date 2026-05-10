"use client";

import { useEffect, useState } from "react";

import { podHttp } from "@/engine/podUrl";
import { usePerformanceStore } from "@/store/usePerformanceStore";

import { SliderGroup } from "./SliderGroup";

// streamA2A feature-bank tile. Five faders (strength, cache depth /
// interval, FF α / threshold) plus a control strip with bank ON/OFF,
// freeze, FF, ToMe toggles and a clear button. Layout mirrors DcwTile.
//
// Gated to eager-mode decoders: the TRT decoder backend skips
// ``enable_feature_bank`` (pipeline.run keys off ``_trt_engine is None``),
// so the bank never installs and the toggles do nothing on TRT. We hide
// the whole tile in that case to avoid showing dead controls. The gate
// fetches /api/server-info once on mount; decoder_accel is the field
// the server publishes (server.py: ``"decoder_accel": _DECODER_ACCEL``).

interface ServerInfo {
  decoder_accel?: string;
}

const SLIDERS: { param: string; label: string }[] = [
  { param: "bank_strength", label: "bank strength" },
  { param: "bank_cache_depth", label: "depth" },
  { param: "bank_cache_interval", label: "interval" },
  { param: "bank_fi_strength", label: "FF α" },
  { param: "bank_fi_threshold", label: "FF thresh" },
];

export function BankTile() {
  // null while loading; "" if the fetch failed (treat as eager so the
  // tile renders rather than silently hiding when the proxy is down).
  const [decoderAccel, setDecoderAccel] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(podHttp("/api/server-info"))
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((info: ServerInfo) => {
        if (cancelled) return;
        setDecoderAccel(info.decoder_accel ?? "");
      })
      .catch(() => {
        if (cancelled) return;
        setDecoderAccel("");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const bankEnabled = usePerformanceStore((s) => s.bankEnabled);
  const bankFreeze = usePerformanceStore((s) => s.bankFreeze);
  const bankFiEnabled = usePerformanceStore((s) => s.bankFiEnabled);
  const bankTomeEnabled = usePerformanceStore((s) => s.bankTomeEnabled);
  const toggleBank = usePerformanceStore((s) => s.toggleBank);
  const toggleBankFreeze = usePerformanceStore((s) => s.toggleBankFreeze);
  const toggleBankFi = usePerformanceStore((s) => s.toggleBankFi);
  const toggleBankTome = usePerformanceStore((s) => s.toggleBankTome);
  const bumpBankClearSeq = usePerformanceStore((s) => s.bumpBankClearSeq);

  // Don't render until we've heard back from the server. Avoids a
  // first-paint flash where the tile briefly appears under TRT while
  // the fetch is in flight.
  if (decoderAccel === null) return null;
  // Strict: only show in eager. Empty string ("" — fetch failed) falls
  // through to hidden as well, since we can't confirm bank is installable.
  if (decoderAccel !== "eager") return null;

  return (
    <div className="mixer-tile" data-tile="stream-a2a">
      <div className="mixer-tile-label">streamA2A</div>
      <div className="mixer-channels">
        {SLIDERS.map(({ param, label }) => (
          <SliderGroup key={param} param={param} label={label} />
        ))}
        <div className="dcw-panel">
          <button
            type="button"
            className={`dcw-toggle${bankEnabled ? " active" : ""}`}
            data-role="bank-enabled"
            onClick={toggleBank}
          >
            bank: {bankEnabled ? "ON" : "OFF"}
          </button>
          <button
            type="button"
            className={`dcw-toggle${bankFreeze ? " active" : ""}`}
            data-role="bank-freeze"
            onClick={toggleBankFreeze}
          >
            freeze: {bankFreeze ? "ON" : "OFF"}
          </button>
          <button
            type="button"
            className={`dcw-toggle${bankFiEnabled ? " active" : ""}`}
            data-role="bank-fi-enabled"
            onClick={toggleBankFi}
          >
            FF: {bankFiEnabled ? "ON" : "OFF"}
          </button>
          <button
            type="button"
            className={`dcw-toggle${bankTomeEnabled ? " active" : ""}`}
            data-role="bank-tome-enabled"
            onClick={toggleBankTome}
          >
            ToMe: {bankTomeEnabled ? "ON" : "OFF"}
          </button>
          <button
            type="button"
            className="dcw-toggle"
            data-role="bank-clear"
            onClick={bumpBankClearSeq}
          >
            clear
          </button>
        </div>
      </div>
    </div>
  );
}
