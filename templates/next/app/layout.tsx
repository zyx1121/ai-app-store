import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "{{name}}",
  description: "{{description}}",
};

// No remote font: the build must work on a machine that is offline.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
