import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        // A small, deliberate palette so the app feels coherent end-to-end.
        bg:      "#0b0d12",
        panel:   "#13161e",
        panel2:  "#1a1f2b",
        border:  "#262c3a",
        text:    "#e6e8ee",
        muted:   "#8b93a7",
        accent:  "#4a9eed",
        accent2: "#8b5cf6",
        good:    "#22c55e",
        warn:    "#f59e0b",
        bad:     "#ef4444",
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas"],
      },
    },
  },
  plugins: [],
};
export default config;
