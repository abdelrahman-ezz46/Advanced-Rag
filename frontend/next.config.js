/** @type {import('next').NextConfig} */
const nextConfig = {
  // The Python FastAPI backend runs on :8000 during dev.
  // We rewrite /api/* -> the backend so the frontend can use relative URLs
  // without any CORS headaches.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};
module.exports = nextConfig;
