// src/app/page.tsx
"use client";

import { useState, useEffect } from 'react';
import { useUser, useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';

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
    "L'Umanizzatore",
  ];
  const [selectedProfile, setSelectedProfile] = useState(profileOptions[0]);
  
  const { isLoaded, isSignedIn, user } = useUser();
  const { signOut, getToken } = useAuth();
  const router = useRouter();

  useEffect(() => {
    const fetchUsage = async () => {
        if (!isLoaded || !isSignedIn) {
            return;
        }

        try {
            const token = await getToken();
            if (!token) {
                console.error("Impossibile ottenere il token di autenticazione per fetchUsage.");
                return;
            }

            const headers: HeadersInit = {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            };

            const apiUrl = process.env.NEXT_PUBLIC_API_URL;
            const response = await fetch(`${apiUrl}/user-status`, {
                method: 'GET',
                headers: headers,
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Errore nel recupero dello stato utente:', errorData.detail || 'Errore sconosciuto.');
                if (response.status === 401) {
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
  }, [isLoaded, isSignedIn, getToken, signOut, router]);
  
  if (!isLoaded) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-gray-900 text-white">
        <p>Caricamento autenticazione...</p>
      </main>
    );
  }

  if (!isSignedIn) {
     router.push('/login');
     return (
        <main className="flex min-h-screen items-center justify-center bg-gray-900 text-white">
            <p>Reindirizzamento al login...</p>
        </main>
     );
  }

  const handleValidate = async () => {
    if (!isSignedIn) {
      alert('Devi essere autenticato per validare il testo.');
      router.push('/login');
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
      const token = await getToken();
      if (!token) {
        throw new Error("Impossibile ottenere il token di autenticazione da Clerk.");
      }

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
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

  const handleLogout = async () => {
      setIsLoading(true);
      try {
          await signOut(() => router.push('/login'));
      } catch (error: any) {
          console.error('Errore durante il logout:', error.message);
          alert('Errore durante il logout: ' + error.message);
      } finally {
          setIsLoading(false);
      }
  };
  
  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
        <header className="mb-8 text-center relative">
        
          <div className="absolute top-0 right-4 flex flex-col items-end space-y-2"> 
            <button
              onClick={handleLogout}
              disabled={isLoading}
              className="rounded-lg bg-red-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-red-600 transition-colors duration-200"
            >
              Logout ({user?.emailAddresses[0]?.emailAddress || 'utente'})
            </button>
            {usageCount !== null && usageLimit !== null && (
              <div className="text-sm text-gray-300 bg-gray-700 px-3 py-1 rounded-md shadow-sm">
                {usageLimit === -1 ? (
                  <span>Utilizzo: Illimitato</span>
                ) : (
                  <span>Chiamate: {usageCount} / {usageLimit}</span>
                )}
              </div>
            )}
          </div>

          {/* INIZIO MODIFICHE: Font Titolo e Sottotitolo */}
          <h1 className="text-4xl font-extrabold text-blue-400">Text Validator</h1> {/* Modificato da font-bold a font-extrabold */}
          <p className="mt-2 text-gray-400 font-semibold"> {/* Aggiunto font-semibold */}
            Pulisci, normalizza e valida la qualità dei tuoi testi in un solo click.
          </p>
          {/* FINE MODIFICHE: Font Titolo e Sottotitolo */}
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
            className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-blue-500 focus:ring-blue-500 transition-all duration-200"
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
                className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-gray-600 transition-colors duration-200 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed"
              >
                Pulisci
              </button>
            </div>
            <textarea
              id="inputText"
              rows={15}
              className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-blue-500 focus:ring-blue-500 resize-none transition-all duration-200"
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
                className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-gray-600 transition-colors duration-200 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed"
              >
                {copyButtonText}
              </button>
            </div>
            <textarea
              id="outputText"
              rows={15}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 p-3 text-gray-200 shadow-sm resize-none"
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
            className="rounded-xl bg-blue-700 px-10 py-4 text-xl font-bold text-white shadow-lg hover:bg-blue-600 transition-all duration-300 disabled:bg-gray-700 disabled:text-gray-400 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Validazione in corso...' : 'Valida Testo'}
          </button>
        </div>

        {qualityReport && (
          <div className="mt-8 rounded-xl bg-gray-800 p-6 border border-gray-700 shadow-xl">
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