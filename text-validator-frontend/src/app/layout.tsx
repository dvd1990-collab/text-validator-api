// src/app/layout.tsx
import { ClerkProvider } from "@clerk/nextjs";
import Navbar from '@/components/Navbar';
import { UsageProvider } from '@/context/UsageContext'; // <-- IMPORTA IL PROVIDER
import './globals.css';

// ... metadata ...

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <UsageProvider> {/* <-- AVVOLGI L'APP CON IL PROVIDER */}
        <html lang="en">
          <body className={`antialiased bg-gray-900 text-white`}>
            <Navbar />
            <div className="pt-24">
              {children}
            </div>
          </body>
        </html>
      </UsageProvider>
    </ClerkProvider>
  );
}