// src/app/login/page.tsx
"use client";

import { SignIn } from "@clerk/nextjs";

export default function LoginPage() {
  return (
    // La main tag avrà un padding top dal layout.tsx
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-900 p-8 text-white">
      
      {/* 1. Modulo di Login: Manteniamo solo il modulo Clerk al centro della viewport */}
      <div className="w-full max-w-md rounded-lg bg-gray-800 p-8 shadow-lg border border-gray-700"> {/* Rimosso mb-16 e flex-col items-center */}
        {/* RIMOSSO: h1 "Accedi a Text Validator" */}
        
        {/* Il componente SignIn di Clerk è già centrato nel suo contenitore se il contenitore è flex */}
        {/* o se l'elemento stesso ha margin: auto. Qui non c'è bisogno di classi extra per centrarlo internamente al div */}
        <SignIn
          signUpUrl="/sign-up"
          afterSignInUrl="/"
          afterSignUpUrl="/"
		  routing="hash"
        />
      </div>

      {/* TUTTE LE SEZIONI INFORMATIVE SONO STATE RIMOSSE DA QUI */}
    </main>
  );
}