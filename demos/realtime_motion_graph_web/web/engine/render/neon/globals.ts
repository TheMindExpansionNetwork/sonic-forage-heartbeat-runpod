// Module-level GL singleton, matching wayne-wu/webgl-music-visualizer's
// pattern. The neon visualizer is a single-instance renderer; NeonRenderer
// calls setGL() at start() before constructing any geometry / shader.
//
// SSR-safe: the type is intentionally not initialised at module load so we
// don't touch WebGL during Node prerender.

export let gl: WebGL2RenderingContext = null as unknown as WebGL2RenderingContext;

export function setGL(_gl: WebGL2RenderingContext) {
  gl = _gl;
}
