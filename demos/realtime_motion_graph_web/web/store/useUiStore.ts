"use client";

import { create } from "zustand";

interface UiState {
  configOpen: boolean;
  graphDisplay: "lines" | "face";
  setConfigOpen: (v: boolean) => void;
  toggleConfig: () => void;
  setGraphDisplay: (v: "lines" | "face") => void;
  toggleGraphDisplay: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  configOpen: false,
  graphDisplay: "lines",
  setConfigOpen: (v) => set({ configOpen: v }),
  toggleConfig: () => set((s) => ({ configOpen: !s.configOpen })),
  setGraphDisplay: (v) => set({ graphDisplay: v }),
  toggleGraphDisplay: () =>
    set((s) => ({ graphDisplay: s.graphDisplay === "lines" ? "face" : "lines" })),
}));
