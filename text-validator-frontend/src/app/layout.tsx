// src/app/layout.tsx
// src/app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";
// --- NUOVA IMPORTAZIONE DI CLERK ---
import { ClerkProvider } from "@clerk/nextjs";
import Navbar from '@/components/Navbar'; 

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
    <ClerkProvider>
      <html lang="en">
        <body className={`antialiased bg-gray-900 text-white`}> {/* Aggiunto bg-gray-900 text-white al body */}
          <Navbar /> {/* <--- Includi la Navbar qui */}
          <div className="pt-24"> {/* <--- Aggiunto padding-top per la Navbar fissa */}
            {children}
          </div>
        </body>
      </html>
    </ClerkProvider>
  );
}