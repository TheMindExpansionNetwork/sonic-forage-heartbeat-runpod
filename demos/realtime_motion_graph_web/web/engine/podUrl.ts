// URL helpers — back-compat re-export shim.

export {
  podHttp,
  setPodSessionId,
  getPodSessionId,
} from "./rtmgConfig";

/** WS URL fallback. Returns `?ws=<override>` from the URL or
 *  `NEXT_PUBLIC_POD_BASE_URL` rewritten as `ws://`. On RunPod the page is
 *  served from `{podId}-{webPort}.proxy.runpod.net` while the engine listens
 *  on `{podId}-1318.proxy.runpod.net`, so derive that when the hostname matches. */
export function defaultWsUrl(): string {
  if (typeof window !== "undefined") {
    const params = new URLSearchParams(window.location.search);
    const override = params.get("ws");
    if (override) return override;

    const runpod = window.location.hostname.match(
      /^(.+)-\d+\.proxy\.runpod\.net$/,
    );
    if (runpod) {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${proto}//${runpod[1]}-1318.proxy.runpod.net/`;
    }
  }
  const base = process.env.NEXT_PUBLIC_POD_BASE_URL ?? "";
  return base.replace(/\/$/, "").replace(/^http/, "ws") + "/";
}

export function podBaseUrl(): string {
  return (process.env.NEXT_PUBLIC_POD_BASE_URL ?? "").replace(/\/$/, "");
}
