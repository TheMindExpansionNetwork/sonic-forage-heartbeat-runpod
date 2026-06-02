import type { NextConfig } from "next";

const backendUrl = (
  process.env.NEXT_PUBLIC_POD_BASE_URL ?? "http://127.0.0.1:1318"
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  // Allow RunPod proxy domains for HMR/WebSocket connections
  allowedDevOrigins: [
    "*.proxy.runpod.net",
    ...(process.env.RUNPOD_POD_ID
      ? [`${process.env.RUNPOD_POD_ID}-6660.proxy.runpod.net`]
      : []),
  ],

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: "/fixtures/:path*",
        destination: `${backendUrl}/fixtures/:path*`,
      },
      {
        source: "/user_uploads/:path*",
        destination: `${backendUrl}/user_uploads/:path*`,
      },
      {
        source: "/loras/:path*",
        destination: `${backendUrl}/loras/:path*`,
      },
      {
        source: "/videos/:path*",
        destination: `${backendUrl}/videos/:path*`,
      },
    ];
  },
};

export default nextConfig;