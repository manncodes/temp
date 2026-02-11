import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "vLLM Tracing — infini-gram",
  description:
    "Scan vLLM deployments, chat with models, and trace responses back to training data using infini-gram",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
