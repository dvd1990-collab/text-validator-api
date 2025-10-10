// src/app/login/page.tsx
"use client"; // Mantieni questa direttiva se stai usando componenti lato client

// Rimuovi queste importazioni non più necessarie
// import { useState } from 'react';
// import { useRouter } from 'next/navigation';
// import { supabase } from '@/lib/supabase';

import { SignIn } from "@clerk/nextjs"; // Mantieni questa importazione

export default function LoginPage() {
  // Rimuovi tutti questi stati e la funzione handleAuth
  // const [email, setEmail] = useState('');
  // const [password, setPassword] = useState('');
  // const [isLoading, setIsLoading] = useState(false);
  // const [error, setError] = useState<string | null>(null);
  // const [isSigningUp, setIsSigningUp] = useState(false);

  // const router = useRouter(); // Non più necessario se i componenti Clerk gestiscono il reindirizzamento

  // Rimuovi l'intera funzione handleAuth
  // const handleAuth = async (isSignUp: boolean) => { ... };

  return (
    // Il main wrapper rimane uguale, solo il contenuto interno cambia
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-900 p-8 text-white">
      
      {/* 1. Modulo di Login: SOSTITUISCI L'INTERO DIV ESISTENTE CON IL COMPONENTE CLERK */}
      <div className="w-full max-w-md rounded-lg bg-gray-800 p-8 shadow-lg border border-gray-700">
        <h1 className="text-3xl font-bold text-center text-blue-400 mb-6">
          Accedi a Text Validator
        </h1>
        {/* IL COMPONENTE <SignIn /> DI CLERK SOSTITUISCE TUTTO IL FORM MANUALE */}
        <SignIn
          signUpUrl="/sign-up" // URL della pagina di registrazione
          afterSignInUrl="/" // URL di reindirizzamento dopo il login riuscito
          afterSignUpUrl="/" // URL di reindirizzamento dopo la registrazione riuscita (per consistenza)
		  routing="hash" // <-- AGGIUNGI QUESTA LINEA
        />
      </div>

      {/* --- 2. SEZIONE INFORMATIVA (puoi mantenerla invariata) --- */}
      <div className="w-full max-w-md mt-12 text-left px-4">
        <h2 className="text-3xl font-semibold text-blue-300 mb-4">
          Un Tool, 10 Profili Esperti al Tuo Servizio
        </h2>
        <p className="text-gray-400 mb-4 text-lg">
          Accedi per sbloccare una suite di strumenti di validazione AI, ciascuno specializzato in un'area critica della comunicazione aziendale.
        </p>
        <ul className="list-disc list-inside space-y-3 text-gray-300">
          <li>
            <strong>Normalizza e Pulisci:</strong> Dal nostro profilo "Generico" a quello "Legale", rimuovi il caos e uniforma il tono.
          </li>
          <li>
            <strong>Crea Contenuti Persuasivi:</strong> Usa i profili "Copywriter" e "Scrittore di Newsletter" per testi che convertono.
          </li>
          <li>
            <strong>Gestisci Comunicazioni Delicate:</strong> Affidati ai profili "PR di Crisi" e "HR" per messaggi chiari ed empatici.
          </li>
          <li>
            <strong>Domina la Scena Digitale:</strong> Con i profili "Social Media Manager" e "Ottimizzatore Email di Vendita" per massimizzare l'impatto.
          </li>
          <li>
            <strong>Semplifica il Complesso:</strong> I profili "Project Manager" e "Traduttore IT" trasformano note tecniche in insight azionabili.
          </li>
        </ul>
      </div>

    </main>
  );
}