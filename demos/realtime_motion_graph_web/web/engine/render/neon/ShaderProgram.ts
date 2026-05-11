// Adapted from wayne-wu/webgl-music-visualizer (src/rendering/gl/ShaderProgram.ts).

import Drawable from "./Drawable";
import { gl } from "./globals";
import type { Mat4, Vec4 } from "./mat";
import { mat4 } from "./mat";

let activeProgram: WebGLProgram | null = null;

export class Shader {
  shader: WebGLShader;
  constructor(type: number, source: string) {
    this.shader = gl.createShader(type)!;
    gl.shaderSource(this.shader, source);
    gl.compileShader(this.shader);
    if (!gl.getShaderParameter(this.shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(this.shader) ?? "shader compile failed";
      gl.deleteShader(this.shader);
      throw new Error(log);
    }
  }
}

class ShaderProgram {
  prog: WebGLProgram;

  attrPos: number;
  attrNor: number;
  attrUV: number;

  unifModel: WebGLUniformLocation | null;
  unifModelInvTr: WebGLUniformLocation | null;
  unifViewProj: WebGLUniformLocation | null;
  unifColor: WebGLUniformLocation | null;
  unifTime: WebGLUniformLocation | null;

  unifFBMScale: WebGLUniformLocation | null;
  unifFBMPersistence: WebGLUniformLocation | null;
  unifFBMOctaves: WebGLUniformLocation | null;
  unifFBMOffset: WebGLUniformLocation | null;

  unifAudioFreqAvg: WebGLUniformLocation | null;
  unifAudioTimeAvg: WebGLUniformLocation | null;

  unifGlow: WebGLUniformLocation | null;

  constructor(shaders: Array<Shader>) {
    this.prog = gl.createProgram()!;
    for (const shader of shaders) gl.attachShader(this.prog, shader.shader);
    gl.linkProgram(this.prog);
    if (!gl.getProgramParameter(this.prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(this.prog) ?? "program link failed";
      throw new Error(log);
    }

    this.attrPos = gl.getAttribLocation(this.prog, "vs_Pos");
    this.attrNor = gl.getAttribLocation(this.prog, "vs_Nor");
    this.attrUV = gl.getAttribLocation(this.prog, "vs_UV");

    this.unifModel = gl.getUniformLocation(this.prog, "u_Model");
    this.unifModelInvTr = gl.getUniformLocation(this.prog, "u_ModelInvTr");
    this.unifViewProj = gl.getUniformLocation(this.prog, "u_ViewProj");
    this.unifColor = gl.getUniformLocation(this.prog, "u_Color");
    this.unifTime = gl.getUniformLocation(this.prog, "u_Time");

    this.unifFBMScale = gl.getUniformLocation(this.prog, "u_FBMScale");
    this.unifFBMPersistence = gl.getUniformLocation(this.prog, "u_FBMPersistence");
    this.unifFBMOctaves = gl.getUniformLocation(this.prog, "u_FBMOctaves");
    this.unifFBMOffset = gl.getUniformLocation(this.prog, "u_FBMOffset");

    this.unifAudioFreqAvg = gl.getUniformLocation(this.prog, "u_AudioFreqAvg");
    this.unifAudioTimeAvg = gl.getUniformLocation(this.prog, "u_AudioTimeAvg");

    this.unifGlow = gl.getUniformLocation(this.prog, "u_Glow");
  }

  use() {
    if (activeProgram !== this.prog) {
      gl.useProgram(this.prog);
      activeProgram = this.prog;
    }
  }

  setModelMatrix(model: Mat4) {
    this.use();
    if (this.unifModel) gl.uniformMatrix4fv(this.unifModel, false, model);
    if (this.unifModelInvTr) {
      const m = mat4.create();
      mat4.transpose(m, model);
      mat4.invert(m, m);
      gl.uniformMatrix4fv(this.unifModelInvTr, false, m);
    }
  }

  setViewProjMatrix(vp: Mat4) {
    this.use();
    if (this.unifViewProj) gl.uniformMatrix4fv(this.unifViewProj, false, vp);
  }

  setGeometryColor(color: Vec4) {
    this.use();
    if (this.unifColor) gl.uniform4fv(this.unifColor, color);
  }

  setTime(time: number) {
    this.use();
    if (this.unifTime) gl.uniform1f(this.unifTime, time);
  }

  setNoise(scale: number, persistence: number, octaves: number, offset: number) {
    this.use();
    if (this.unifFBMScale) gl.uniform1f(this.unifFBMScale, scale);
    if (this.unifFBMPersistence) gl.uniform1f(this.unifFBMPersistence, persistence);
    if (this.unifFBMOctaves) gl.uniform1f(this.unifFBMOctaves, octaves);
    if (this.unifFBMOffset) gl.uniform1f(this.unifFBMOffset, offset);
  }

  setAudio(freqAvg: number, timeAvg: number) {
    this.use();
    if (this.unifAudioFreqAvg) gl.uniform1f(this.unifAudioFreqAvg, freqAvg);
    if (this.unifAudioTimeAvg) gl.uniform1f(this.unifAudioTimeAvg, timeAvg);
  }

  setBloom(glow: number) {
    this.use();
    if (this.unifGlow) gl.uniform1f(this.unifGlow, glow);
  }

  draw(d: Drawable) {
    this.use();

    if (this.attrPos !== -1 && d.bindPos()) {
      gl.enableVertexAttribArray(this.attrPos);
      gl.vertexAttribPointer(this.attrPos, 4, gl.FLOAT, false, 0, 0);
    }
    if (this.attrNor !== -1 && d.bindNor()) {
      gl.enableVertexAttribArray(this.attrNor);
      gl.vertexAttribPointer(this.attrNor, 4, gl.FLOAT, false, 0, 0);
    }
    if (this.attrUV !== -1 && d.bindUV()) {
      gl.enableVertexAttribArray(this.attrUV);
      gl.vertexAttribPointer(this.attrUV, 2, gl.FLOAT, false, 0, 0);
    }

    d.bindIdx();
    gl.drawElements(d.drawMode(), d.elemCount(), gl.UNSIGNED_INT, 0);

    if (this.attrPos !== -1) gl.disableVertexAttribArray(this.attrPos);
    if (this.attrNor !== -1) gl.disableVertexAttribArray(this.attrNor);
    if (this.attrUV !== -1) gl.disableVertexAttribArray(this.attrUV);
  }
}

export default ShaderProgram;
