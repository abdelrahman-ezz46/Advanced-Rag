import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Advanced RAG",
  description: "Local multimodal RAG assistant — chat, knowledge base, memory.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text antialiased">{children}</body>
    </html>
  );
}
