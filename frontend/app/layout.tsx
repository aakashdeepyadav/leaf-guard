import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LeafGuard",
  description: "Explainable plant disease detection",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
