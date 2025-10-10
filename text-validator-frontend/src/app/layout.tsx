// src/app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";
// --- NUOVA IMPORTAZIONE DI CLERK ---
import { ClerkProvider } from "@clerk/nextjs";

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
    // --- AVVOLGI L'APPLICAZIONE CON CLERKPROVIDER ---
    <ClerkProvider>
      <html lang="en">
        <body className={`antialiased`}>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}