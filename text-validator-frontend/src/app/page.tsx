// src/app/page.tsx
"use client";

import { useState, useEffect } from 'react';
// Rimuovi l'importazione di useRouter e supabase, non più usati direttamente per l'auth
// import { useRouter } from 'next/navigation';
// import { supabase } from '@/lib/supabase';

// --- NUOVE IMPORTAZIONI DI CLERK ---
import { useUser, useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation'; // Manteniamo useRouter per i reindirizzamenti

type QualityReport = {
  reasoning: string;
  human_quality_score: number;
};

export default function HomePage() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [copyButtonText, setCopyButtonText] = useState('Copia');
  const [usageCount, setUsageCount] = useState<number | null>(null);
  const [usageLimit, setUsageLimit] = useState<number | null>(null);

  const profileOptions = [
    "Generico",
    "PM - Interpretazione Trascrizioni",
    "Copywriter Persuasivo",
    "Revisore Legale/Regolatorio",
    "Scrittore di Newsletter",
    "Social Media Manager B2B",
    "Comunicatore di Crisi PR",
    "Traduttore Tecnico IT",
    "Specialista Comunicazioni HR",
    "Ottimizzatore Email di Vendita",
  ];
  const [selectedProfile, setSelectedProfile] = useState(profileOptions[0]);
  
  // --- NUOVI HOOKS DI CLERK ---
  const { isLoaded, isSignedIn, user } = useUser(); // Per lo stato dell'utente
  const { signOut, getToken } = useAuth(); // Per il logout e ottenere il JWT
  const router = useRouter(); // Mantenuto per reindirizzamenti manuali se necessari

  // Rimuovi completamente i vecchi useEffect per la sessione Supabase
  // useEffect(() => { ... }, []);
  // Rimuovi completamente il vecchio useEffect per il reindirizzamento
  // useEffect(() => { ... }, [userSession, loadingSession, router]);

  // Clerk gestirà i reindirizzamenti automaticamente grazie al middleware.
  // Tuttavia, potresti voler mostrare uno stato di caricamento mentre Clerk inizializza.
  useEffect(() => {
    const fetchUsage = async () => {
        if (!isLoaded || !isSignedIn) {
            return; // Aspetta che Clerk abbia caricato e l'utente sia autenticato
        }

        try {
            const token = await getToken(); // Ottieni il token di Clerk
            if (!token) {
                console.error("Impossibile ottenere il token di autenticazione per fetchUsage.");
                return;
            }

            const headers: HeadersInit = {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`, // Invia il token Clerk
            };

            const apiUrl = process.env.NEXT_PUBLIC_API_URL;
            const response = await fetch(`${apiUrl}/user-status`, { // Chiamata al nuovo endpoint
                method: 'GET',
                headers: headers,
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Errore nel recupero dello stato utente:', errorData.detail || 'Errore sconosciuto.');
                if (response.status === 401) {
                    // Se il token è scaduto o non valido, forza il logout
                    await signOut(() => router.push('/login'));
                    alert('La sessione è scaduta durante il caricamento dello stato. Effettua nuovamente il login.');
                }
                return;
            }

            const data = await response.json();
            setUsageCount(data.count);
            setUsageLimit(data.limit);
        } catch (error) {
            console.error('Errore sconosciuto durante il recupero dello stato utente:', error);
        }
    };
    fetchUsage();
  }, [isLoaded, isSignedIn, getToken, signOut, router]); // Dipendenze per rieseguire l'hook quando lo stato di auth cambia
  
  if (!isLoaded) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-gray-900 text-white">
        <p>Caricamento autenticazione...</p>
      </main>
    );
  }

  // Se l'utente non è autenticato e Clerk ha finito di caricare,
  // il middleware dovrebbe già aver reindirizzato. Questo è un fallback.
  if (!isSignedIn) {
     router.push('/login');
     return (
        <main className="flex min-h-screen items-center justify-center bg-gray-900 text-white">
            <p>Reindirizzamento al login...</p>
        </main>
     );
  }

  const handleValidate = async () => {
    // --- MODIFICATO: userSession non esiste più, usiamo isSignedIn di Clerk ---
    if (!isSignedIn) {
      alert('Devi essere autenticato per validare il testo.');
      router.push('/login'); // Reindirizza se per qualche motivo non è loggato
      return;
    }
    if (!inputText.trim()) {
      alert('Per favore, inserisci del testo da validare.');
      return;
    }
    setIsLoading(true);
    setOutputText('Elaborazione in corso...');
    setQualityReport(null);

    try {
      // --- MODIFICATO: Otteniamo il token JWT da Clerk ---
      const token = await getToken(); // Questo ottiene il JWT di Clerk
      if (!token) {
        throw new Error("Impossibile ottenere il token di autenticazione da Clerk.");
      }

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`, // Usiamo il token di Clerk
      };

      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      console.log("URL dell'API che sto per chiamare:", apiUrl);
      const response = await fetch(`${apiUrl}/validate`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
          text: inputText,
          profile_name: selectedProfile,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 401) {
            // Se il token è scaduto o non valido, forza il logout
            await signOut(() => router.push('/login'));
            alert('La sessione è scaduta. Effettua nuovamente il login.');
            return;
        }
        if (response.status === 429) {
          alert(data.detail || "Hai superato il limite di chiamate giornaliere.");
        }
        throw new Error(data.detail || 'Si è verificato un errore.');
      }
      
      if (data && data.error) {
          throw new Error(data.details || data.error);
      }

      setOutputText(data.normalized_text);
      setQualityReport(data.quality_report);
      setUsageCount(data.usage.count);
      setUsageLimit(data.usage.limit);

    } catch (error) {
        if (error instanceof Error) {
            console.error('Errore nella chiamata API:', error);
            let errorMessage = 'Si è verificato un errore sconosciuto.';
            if (error.message) {
                try {
                    const parsedError = JSON.parse(error.message);
                    errorMessage = parsedError.detail || JSON.stringify(parsedError);
                } catch {
                    errorMessage = error.message;
                }
            }
            setOutputText(`Errore: ${errorMessage}`);
        } else {
            console.error('Errore sconosciuto:', error);
            setOutputText(`Errore: ${String(error)}`);
        }
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    if (!outputText || isLoading || outputText.startsWith('Elaborazione') || outputText.startsWith('Errore')) return;
    navigator.clipboard.writeText(outputText)
      .then(() => {
        setCopyButtonText('Copiato!');
        setTimeout(() => setCopyButtonText('Copia'), 2000);
      })
      .catch(err => console.error('Errore durante la copia:', err));
  };

  const handleClear = () => {
    setInputText('');
    setOutputText('');
    setQualityReport(null);
  };

  // --- LOGICA DI LOGOUT MODIFICATA PER CLERK ---
  const handleLogout = async () => {
      setIsLoading(true);
      try {
          // Utilizza il metodo signOut di Clerk
          await signOut(() => router.push('/login'));
      } catch (error: any) {
          console.error('Errore durante il logout:', error.message);
          alert('Errore durante il logout: ' + error.message);
      } finally {
          setIsLoading(false);
      }
  };
  
  // Rimuovi il rendering condizionale basato su loadingSession e userSession
  // if (loadingSession || !userSession) { ... }
  // if (!userSession) { ... }
  // La gestione del caricamento e del reindirizzamento iniziale è ora sopra,
  // gestita da !isLoaded e !isSignedIn.

  // Se il caricamento è finito E l'utente è autenticato, mostra l'applicazione.
  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
        <header className="mb-8 text-center relative">
        
          <div className="absolute top-0 right-0 flex items-center space-x-4">
            {usageCount !== null && usageLimit !== null && (
              <div className="text-sm text-gray-400 bg-gray-800 px-3 py-1 rounded-md">
                {usageLimit === -1 ? (
                  <span>Utilizzo: Illimitato</span>
                ) : (
                  <span>Chiamate: {usageCount} / {usageLimit}</span>
                )}
              </div>
            )}
            <button
              onClick={handleLogout}
              disabled={isLoading}
              className="rounded-md bg-red-600 px-3 py-1 text-sm font-semibold text-white shadow-sm hover:bg-red-500"
            >
              {/* --- MODIFICATO: user.emailAddress per Clerk --- */}
              Logout ({user?.emailAddresses[0]?.emailAddress || 'utente'})
            </button>
          </div>

          <h1 className="text-4xl font-bold text-blue-400">Text Validator</h1>
          <p className="mt-2 text-gray-400">
            Pulisci, normalizza e valida la qualità dei tuoi testi in un solo click.
          </p>
        </header>

        <div className="mb-6">
          <label htmlFor="aiProfile" className="block text-sm font-medium text-gray-300 mb-2">
            Seleziona Profilo AI
          </label>
          <select
            id="aiProfile"
            value={selectedProfile}
            onChange={(e) => setSelectedProfile(e.target.value)}
            disabled={isLoading}
            className="w-full rounded-md border-gray-600 bg-gray-800 p-3 text-gray-200 focus:border-blue-500 focus:ring-blue-500"
          >
            {profileOptions.map((profile) => (
              <option key={profile} value={profile}>
                {profile}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div>
            <div className="flex justify-between items-center mb-2">
              <label htmlFor="inputText" className="block text-sm font-medium text-gray-300">
                Incolla qui il tuo testo
              </label>
              <button
                onClick={handleClear}
                disabled={!inputText || isLoading}
                className="rounded-md bg-gray-600 px-3 py-1 text-xs font-semibold text-white shadow-sm hover:bg-gray-500 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed"
              >
                Pulisci
              </button>
            </div>
            <textarea
              id="inputText"
              rows={15}
              className="w-full rounded-md border-gray-600 bg-gray-800 p-3 text-gray-200 focus:border-blue-500 focus:ring-blue-500"
              placeholder="## Report Settimanale..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={isLoading}
            />
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <label htmlFor="outputText" className="block text-sm font-medium text-gray-300">
                Testo Normalizzato
              </label>
              <button
                onClick={handleCopy}
                disabled={!outputText || isLoading || outputText.startsWith('Elaborazione') || outputText.startsWith('Errore')}
                className="rounded-md bg-gray-600 px-3 py-1 text-xs font-semibold text-white shadow-sm hover:bg-gray-500 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed"
              >
                {copyButtonText}
              </button>
            </div>
            <textarea
              id="outputText"
              rows={15}
              className="w-full rounded-md border-gray-600 bg-gray-700 p-3 text-gray-200"
              placeholder="Il risultato apparirà qui..."
              value={outputText}
              readOnly
            />
          </div>
        </div>

        <div className="mt-6 flex justify-center">
          <button
            onClick={handleValidate}
            disabled={isLoading}
            className="rounded-md bg-blue-600 px-8 py-3 text-lg font-semibold text-white shadow-sm hover:bg-blue-500 disabled:bg-gray-500 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Validazione in corso...' : 'Valida Testo'}
          </button>
        </div>

        {qualityReport && (
          <div className="mt-8 rounded-lg bg-gray-800 p-6 border border-gray-700">
            <h2 className="text-2xl font-semibold text-blue-400">Report di Qualità</h2>
            <div className="mt-4 flex items-center justify-center text-center">
              <p className="text-6xl font-bold text-green-400">{qualityReport.human_quality_score}</p>
              <p className="ml-2 text-2xl text-gray-400">/ 100</p>
            </div>
            <p className="mt-4 text-gray-300 text-center italic">
              {qualityReport.reasoning}
            </p>
          </div>
        )}
      </div>
    </main>
  );
}