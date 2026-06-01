"use client";

import { useEffect, useRef, useState } from "react";

import { useUiStore } from "@/store/useUiStore";

const STORAGE_KEY = "demon.graphDisplay";

export function MrdoobFaceDisplay() {
  const graphDisplay = useUiStore((s) => s.graphDisplay);
  const setGraphDisplay = useUiStore((s) => s.setGraphDisplay);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const [mounted, setMounted] = useState(false);
  const [readyToLoad, setReadyToLoad] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "lines" || saved === "face") {
      setGraphDisplay(saved);
    }
    setMounted(true);
  }, [setGraphDisplay]);

  useEffect(() => {
    if (!mounted) return;
    window.localStorage.setItem(STORAGE_KEY, graphDisplay);
  }, [graphDisplay, mounted]);

  useEffect(() => {
    if (!mounted || graphDisplay !== "face") {
      setReadyToLoad(false);
      return;
    }

    const wrap = document.getElementById("graph-wrap");
    if (!wrap) return;

    let raf = 0;
    const update = () => {
      const r = wrap.getBoundingClientRect();
      setReadyToLoad(r.width > 0 && r.height > 0);
    };
    const scheduleUpdate = () => {
      window.cancelAnimationFrame(raf);
      raf = window.requestAnimationFrame(update);
    };

    scheduleUpdate();
    const obs = new ResizeObserver(scheduleUpdate);
    obs.observe(wrap);
    return () => {
      window.cancelAnimationFrame(raf);
      obs.disconnect();
    };
  }, [graphDisplay, mounted]);

  useEffect(() => {
    if (!mounted || graphDisplay !== "face" || !readyToLoad) return;

    let raf = 0;
    let lastBloom = -1;
    const tick = () => {
      const perf = document.getElementById("performance");
      const raw = perf
        ? window.getComputedStyle(perf).getPropertyValue("--bloom-amount")
        : "0";
      const bloom = Number.parseFloat(raw) || 0;
      if (bloom !== lastBloom) {
        iframeRef.current?.contentWindow?.postMessage(
          { type: "demon:face-reactivity", bloom },
          window.location.origin,
        );
        lastBloom = bloom;
      }
      raf = window.requestAnimationFrame(tick);
    };

    tick();
    return () => window.cancelAnimationFrame(raf);
  }, [graphDisplay, mounted, readyToLoad]);

  if (!mounted) return null;

  return (
    graphDisplay === "face" && readyToLoad && (
      <iframe
        ref={iframeRef}
        className="mrdoob-face-display"
        src="/codepenface/index.html?v=demon-reactive"
        title="Face display"
        allow="camera; microphone; fullscreen"
      />
    )
  );
}
