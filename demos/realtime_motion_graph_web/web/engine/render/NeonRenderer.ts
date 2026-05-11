// Neon icosphere visualizer. Adapted from wayne-wu/webgl-music-visualizer
// (https://github.com/wayne-wu/webgl-music-visualizer) — three nested
// line-rendered icospheres displaced by audio-amplified fractal Perlin
// noise, blended through a two-pass Gaussian bloom pipeline.
//
// Lifecycle mirrors GraphRenderer / EffectsRenderer: construct with a
// canvas, drive with tick(now, kick), tear down with destroy().

import Icosphere from "./neon/Icosphere";
import { setGL, gl as neonGL } from "./neon/globals";
import ShaderProgram, { Shader } from "./neon/ShaderProgram";
import Square from "./neon/Square";
import { BLEND_FRAG, BLUR_FRAG, LINE_FRAG, LINE_VERT, QUAD_VERT } from "./neon/shaders";
import { mat4, vec3, vec4, type Mat4 } from "./neon/mat";

const DPR_CAP = 2;

interface NeonControls {
  separation: number;
  glow: number;
  scale: number;
  persistence: number;
  octaves: number;
}

// Retuned from the wayne-wu defaults to read like the title-screen
// start-mark logo (concentric, gently writhing rings in the teal /
// mustard / orange / coral palette — see `ribbons.ts:14-19` and
// `startMarkRingPathD` for the reference math). Lower `glow`, tighter
// `separation`, fewer octaves, smaller `scale` → calm, brand-aligned.
const DEFAULT_CONTROLS: NeonControls = {
  separation: 0.12,
  glow: 2.5,
  scale: 0.6,
  persistence: 0.7,
  octaves: 1,
};

// Title-screen palette, matched to `PALETTE` in `ribbons.ts`. Four rings
// drawn outermost → innermost so the warmest colour reads on the outside,
// mirroring the start-mark layering.
const RING_COLORS: ReadonlyArray<[number, number, number, number]> = [
  [0.910, 0.310, 0.239, 1.0], // coral  #e84f3d (outer)
  [0.941, 0.541, 0.282, 1.0], // orange #f08a48
  [0.780, 0.710, 0.400, 1.0], // mustard #c7b566
  [0.239, 0.714, 0.745, 1.0], // teal   #3db6be (inner)
];

export class NeonRenderer {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private ro: ResizeObserver | null = null;

  // Four nested icospheres — one per ring colour, matching the start-mark
  // logo's 4-ribbon stack. Subdivisions grow inward so the inner rings have
  // more vertices (visually denser writhe at small radii).
  private spheres!: Icosphere[];
  private square!: Square;

  private line!: ShaderProgram;
  private blur!: ShaderProgram;
  private quad!: ShaderProgram;

  // Main scene FBO with two colour attachments: scene + bright extract.
  private fbo!: WebGLFramebuffer;
  private colorTex!: WebGLTexture;
  private brightTex!: WebGLTexture;
  private depthRbo!: WebGLRenderbuffer;

  // Ping-pong blur FBOs.
  private blurFBOs: [WebGLFramebuffer, WebGLFramebuffer];
  private blurTexs: [WebGLTexture, WebGLTexture];

  private time = 0;
  private bufferW = 1;
  private bufferH = 1;

  controls: NeonControls = { ...DEFAULT_CONTROLS };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("webgl2", { antialias: true, alpha: true });
    if (!ctx) throw new Error("WebGL2 unavailable for NeonRenderer");
    this.gl = ctx;
    setGL(this.gl);

    // 4 icospheres at increasing subdivisions; subdivision count chosen so
    // each ring has a similar number of edge segments to `START_MARK_SEGMENTS`
    // (72) in the SVG logo — fine enough to look smooth, coarse enough that
    // bloom can do the heavy visual work.
    const SUBDIVISIONS = [3, 4, 4, 5];
    this.spheres = SUBDIVISIONS.map((sub) => {
      const s = new Icosphere(vec3.fromValues(0, 0, 0), 1.0, sub, this.gl.LINES);
      s.create();
      return s;
    });
    this.square = new Square(vec3.fromValues(0, 0, 0));
    this.square.create();

    this.line = new ShaderProgram([
      new Shader(this.gl.VERTEX_SHADER, LINE_VERT),
      new Shader(this.gl.FRAGMENT_SHADER, LINE_FRAG),
    ]);
    this.blur = new ShaderProgram([
      new Shader(this.gl.VERTEX_SHADER, QUAD_VERT),
      new Shader(this.gl.FRAGMENT_SHADER, BLUR_FRAG),
    ]);
    this.quad = new ShaderProgram([
      new Shader(this.gl.VERTEX_SHADER, QUAD_VERT),
      new Shader(this.gl.FRAGMENT_SHADER, BLEND_FRAG),
    ]);

    // Bind sampler uniforms once.
    this.blur.use();
    this.gl.uniform1i(this.gl.getUniformLocation(this.blur.prog, "scene"), 0);
    this.quad.use();
    this.gl.uniform1i(this.gl.getUniformLocation(this.quad.prog, "scene"), 0);
    this.gl.uniform1i(this.gl.getUniformLocation(this.quad.prog, "blurred"), 1);

    // Initial FBO allocation. Will be resized as soon as we have a measured
    // client rect.
    this.fbo = this.gl.createFramebuffer()!;
    this.colorTex = this.gl.createTexture()!;
    this.brightTex = this.gl.createTexture()!;
    this.depthRbo = this.gl.createRenderbuffer()!;
    this.blurFBOs = [this.gl.createFramebuffer()!, this.gl.createFramebuffer()!];
    this.blurTexs = [this.gl.createTexture()!, this.gl.createTexture()!];

    for (const tex of [this.colorTex, this.brightTex, ...this.blurTexs]) {
      this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_WRAP_S,
        this.gl.CLAMP_TO_EDGE,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_WRAP_T,
        this.gl.CLAMP_TO_EDGE,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_MIN_FILTER,
        this.gl.LINEAR,
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_MAG_FILTER,
        this.gl.LINEAR,
      );
    }

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);
    this.gl.framebufferTexture2D(
      this.gl.DRAW_FRAMEBUFFER,
      this.gl.COLOR_ATTACHMENT0,
      this.gl.TEXTURE_2D,
      this.colorTex,
      0,
    );
    this.gl.framebufferTexture2D(
      this.gl.DRAW_FRAMEBUFFER,
      this.gl.COLOR_ATTACHMENT1,
      this.gl.TEXTURE_2D,
      this.brightTex,
      0,
    );
    this.gl.framebufferRenderbuffer(
      this.gl.FRAMEBUFFER,
      this.gl.DEPTH_ATTACHMENT,
      this.gl.RENDERBUFFER,
      this.depthRbo,
    );
    this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1]);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);

    for (let i = 0; i < 2; i++) {
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.blurFBOs[i]);
      this.gl.framebufferTexture2D(
        this.gl.DRAW_FRAMEBUFFER,
        this.gl.COLOR_ATTACHMENT0,
        this.gl.TEXTURE_2D,
        this.blurTexs[i],
        0,
      );
    }
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);

    this.gl.enable(this.gl.DEPTH_TEST);

    this.resize();
    if (typeof ResizeObserver !== "undefined") {
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(canvas);
    }
  }

  private resize() {
    const dpr = Math.min(DPR_CAP, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (w === this.bufferW && h === this.bufferH) return;
    this.bufferW = w;
    this.bufferH = h;
    this.canvas.width = w;
    this.canvas.height = h;

    for (const tex of [this.colorTex, this.brightTex, ...this.blurTexs]) {
      this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
      this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGBA,
        w,
        h,
        0,
        this.gl.RGBA,
        this.gl.UNSIGNED_BYTE,
        null,
      );
    }
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);

    this.gl.bindRenderbuffer(this.gl.RENDERBUFFER, this.depthRbo);
    this.gl.renderbufferStorage(
      this.gl.RENDERBUFFER,
      this.gl.DEPTH_COMPONENT16,
      w,
      h,
    );
    this.gl.bindRenderbuffer(this.gl.RENDERBUFFER, null);
  }

  /**
   * Drive one frame.
   * @param now       performance.now()-equivalent timestamp (ms).
   * @param freqAvg   audio amplitude proxy in [0, 1] (drives noise amp).
   * @param timeAvg   waveform energy proxy in [0, 1] (declared in shader but unused).
   */
  tick(now: number, freqAvg: number, timeAvg: number) {
    void now;
    this.time++;
    this.resize();
    const gl = this.gl;

    // Face-on camera with a tiny xy wobble — the icospheres should read as
    // concentric rings (like the SVG logo), not as orbiting globes. Far
    // enough back that perspective foreshortening is gentle.
    const t = this.time * 0.005;
    const eye = vec3.fromValues(Math.sin(t * 0.6) * 0.35, Math.sin(t * 0.4) * 0.25, 7.0);
    const center = vec3.fromValues(0, 0, 0);
    const up = vec3.fromValues(0, 1, 0);
    const aspect = this.bufferW / this.bufferH || 1;
    const proj = mat4.create();
    mat4.perspective(proj, (32 * Math.PI) / 180, aspect, 0.1, 1000);
    const view = mat4.create();
    mat4.lookAt(view, eye, center, up);
    const viewProj: Mat4 = mat4.create();
    mat4.multiply(viewProj, proj, view);

    gl.viewport(0, 0, this.bufferW, this.bufferH);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    this.line.setTime(this.time);
    this.line.setAudio(freqAvg, timeAvg);
    this.line.setViewProjMatrix(viewProj);

    // Per-ring noise offsets give each colour its own writhe phase, so the
    // four rings don't move in lockstep. Signs alternate so adjacent rings
    // drift in opposite directions, echoing the SVG's per-ribbon `phase`
    // multiplier.
    const RING_NOISE_OFFSETS = [0.006, -0.009, 0.011, -0.013];
    const baseScale = 1.0;
    for (let i = 0; i < this.spheres.length; i++) {
      const ringScale = baseScale + (this.spheres.length - 1 - i) * this.controls.separation;
      const color = vec4.fromValues(...RING_COLORS[i]);
      const noise: [number, number, number, number] = [
        this.controls.scale,
        this.controls.persistence,
        // First two rings (outermost) at 1 octave for cleanest sine-like
        // writhe; inner rings get one extra octave for slightly more
        // detail at small radius.
        i < 2 ? this.controls.octaves : this.controls.octaves + 1,
        RING_NOISE_OFFSETS[i],
      ];
      const model = mat4.create();
      mat4.identity(model);
      mat4.scale(model, model, vec3.fromValues(ringScale, ringScale, ringScale));
      this.line.setModelMatrix(model);
      this.line.setNoise(noise[0], noise[1], noise[2], noise[3]);
      this.line.setGeometryColor(color);
      this.line.draw(this.spheres[i]);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Two-pass Gaussian blur on bright extract (10 iterations alternating
    // horizontal/vertical).
    let horizontal = true;
    let firstIteration = true;
    this.blur.use();
    const blurHorizontalLoc = gl.getUniformLocation(this.blur.prog, "u_Horizontal");
    for (let i = 0; i < 10; i++) {
      const idx = horizontal ? 1 : 0;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurFBOs[idx]);
      gl.viewport(0, 0, this.bufferW, this.bufferH);
      gl.uniform1i(blurHorizontalLoc, horizontal ? 1 : 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(
        gl.TEXTURE_2D,
        firstIteration ? this.brightTex : this.blurTexs[horizontal ? 0 : 1],
      );
      gl.clear(gl.COLOR_BUFFER_BIT);
      this.blur.draw(this.square);
      horizontal = !horizontal;
      firstIteration = false;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Final composite: scene + blurred bright, tonemap + gamma.
    gl.viewport(0, 0, this.bufferW, this.bufferH);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    this.quad.use();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.colorTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.blurTexs[horizontal ? 0 : 1]);
    this.quad.setBloom(this.controls.glow);
    this.quad.draw(this.square);
  }

  destroy() {
    this.ro?.disconnect();
    this.ro = null;
    // Refresh module GL pointer back to our context in case any geometry
    // destroy() reaches into globals.gl.
    setGL(this.gl);
    void neonGL;
    this.spheres?.forEach((s) => s.destroy());
    this.square?.destroy();
    this.gl.deleteFramebuffer(this.fbo);
    this.gl.deleteFramebuffer(this.blurFBOs[0]);
    this.gl.deleteFramebuffer(this.blurFBOs[1]);
    this.gl.deleteTexture(this.colorTex);
    this.gl.deleteTexture(this.brightTex);
    this.gl.deleteTexture(this.blurTexs[0]);
    this.gl.deleteTexture(this.blurTexs[1]);
    this.gl.deleteRenderbuffer(this.depthRbo);
    this.gl.deleteProgram(this.line.prog);
    this.gl.deleteProgram(this.blur.prog);
    this.gl.deleteProgram(this.quad.prog);
  }
}
