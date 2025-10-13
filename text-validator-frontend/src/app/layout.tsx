// src/app/layout.tsx
import { ClerkProvider } from "@clerk/nextjs";
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer'; // <-- 1. IMPORTA IL FOOTER
import { UsageProvider } from '@/context/UsageContext';
import './globals.css';

// ... metadata ...

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <UsageProvider>
        <html lang="en" className="h-full">
          <body className="flex flex-col min-h-full antialiased bg-gray-900 text-white"> {/* Modifiche per layout sticky footer */}
            <Navbar />
            <main className="flex-grow"> {/* Main ora cresce per riempire lo spazio */}
              <div className="pt-24 pb-24"> {/* Aggiunto padding-bottom */}
                {children}
              </div>
            </main>
            <Footer /> {/* <-- 2. POSIZIONA IL FOOTER QUI */}
          </body>
        </html>
      </UsageProvider>
    </ClerkProvider>
  );
}