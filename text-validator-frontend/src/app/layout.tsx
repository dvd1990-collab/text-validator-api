import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Text Validator",
  description: "Pulisci, normalizza e valida la qualità dei tuoi testi AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`antialiased`}>
        {children}
      </body>
    </html>
  );
}