// src/app/page.tsx
"use client";

import { useState, useEffect } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useUsage } from '@/context/UsageContext'; // Importa il nostro hook del Context

// Non importiamo più SettingsModal o GearIcon qui, sono gestiti dalla Navbar

type QualityReport = {
  reasoning: string;
  human_quality_score: number;
};

export default function HomePage() {
  // Stati locali specifici di questa pagina
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [copyButtonText, setCopyButtonText] = useState('Copia');
  const [selectedProfile, setSelectedProfile] = useState("Generico");

  // Dati globali provenienti dal nostro Context
  const { validator_profiles: allowedProfiles, userTier, fetchUserStatus } = useUsage();

  // Hook di Clerk e Next.js necessari in questa pagina
  const { isSignedIn, getToken, signOut } = useAuth();
  const router = useRouter();
  
  const profileOptions = [
      "Generico", "L'Umanizzatore", "PM - Interpretazione Trascrizioni", 
      "Copywriter Persuasivo", "Revisore Legale/Regolatorio", "Scrittore di Newsletter", 
      "Social Media Manager B2B", "Comunicatore di Crisi PR", "Traduttore Tecnico IT", 
      "Specialista Comunicazioni HR", "Ottimizzatore Email di Vendita"
  ];

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
      if (!token) throw new Error("Impossibile ottenere il token di autenticazione.");

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ text: inputText, profile_name: selectedProfile }),
      });

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 401) {
          signOut(() => router.push('/login'));
          alert('La sessione è scaduta. Effettua nuovamente il login.');
          return;
        }
        throw new Error(data.detail || 'Si è verificato un errore durante la validazione.');
      }
      
      setOutputText(data.normalized_text);
      setQualityReport(data.quality_report);

      // === LA MODIFICA CHIAVE PER L'AGGIORNAMENTO IN TEMPO REALE ===
      // Dopo una validazione riuscita, chiama la funzione del Context per aggiornare i dati globali.
      await fetchUserStatus(); 
      
    } catch (error) {
        if (error instanceof Error) {
            console.error('Errore nella chiamata API:', error);
            setOutputText(`Errore: ${error.message}`);
        } else {
            console.error('Errore sconosciuto:', error);
            setOutputText('Si è verificato un errore sconosciuto.');
        }
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    if (!outputText || isLoading) return;
    navigator.clipboard.writeText(outputText).then(() => {
        setCopyButtonText('Copiato!');
        setTimeout(() => setCopyButtonText('Copia'), 2000);
    });
  };

  const handleClear = () => {
    setInputText('');
    setOutputText('');
    setQualityReport(null);
  };
  
  return (
    // Il Fragment <> non è più necessario perché non c'è il modale qui
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
          <header className="mb-8 text-center relative">
              {/* L'header è pulito: ingranaggio e contatore sono gestiti solo dalla Navbar */}
              <h1 className="text-4xl font-extrabold text-blue-400">Text Validator</h1>
              <p className="mt-2 text-gray-400 font-semibold">
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
                  className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                  {profileOptions.map((profile) => {
                      // La logica per disabilitare i profili ora usa 'allowedProfiles' dal Context
                      const isDisabled = !!allowedProfiles && allowedProfiles !== 'all' && !allowedProfiles.includes(profile);
                      return (
                          <option
                              key={profile}
                              value={profile}
                              disabled={isDisabled}
                              className={isDisabled ? 'text-gray-500' : ''}
                          >
                              {profile}
                              {isDisabled ? ' 🔒 (Upgrade)' : ''}
                          </option>
                      );
                  })}
              </select>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="flex justify-between items-center mb-2">
                <label htmlFor="inputText" className="block text-sm font-medium text-gray-300">
                  Incolla qui il tuo testo
                </label>
                <button
                  onClick={handleClear}
                  disabled={!inputText || isLoading}
                  className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-gray-600 transition-colors disabled:bg-gray-800 disabled:cursor-not-allowed"
                >
                  Pulisci
                </button>
              </div>
              <textarea
                id="inputText"
                rows={15}
                className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-blue-500 focus:ring-blue-500 resize-none"
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
                  disabled={!outputText || isLoading}
                  className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-gray-600 transition-colors disabled:bg-gray-800 disabled:cursor-not-allowed"
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
              className="rounded-xl bg-blue-700 px-10 py-4 text-xl font-bold text-white shadow-lg hover:bg-blue-600 transition-all disabled:bg-gray-700 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Validazione in corso...' : 'Valida Testo'}
            </button>
          </div>

          {userTier !== 'free' && qualityReport && (
  		    <div className="mt-8 rounded-xl bg-gray-800 p-6 border border-gray-700 shadow-xl">
			  <h2 className="text-2xl font-semibold text-blue-400">Controllo di Qualità Esperto</h2>
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