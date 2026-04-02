// Load nu-finance/.env (one level above backend/) for Next.js server-side env vars
require("dotenv").config({ path: require("path").resolve(__dirname, "../.env") });

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow the Vite frontend (port 3000) to call this API (port 3001)
  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "Access-Control-Allow-Origin",  value: "http://localhost:3000" },
          { key: "Access-Control-Allow-Methods", value: "GET,POST,OPTIONS"       },
          { key: "Access-Control-Allow-Headers", value: "Content-Type"           },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
