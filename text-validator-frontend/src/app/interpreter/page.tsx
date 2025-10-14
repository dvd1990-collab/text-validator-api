// src/app/interpreter/page.tsx
"use client";

import { useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useUsage } from '@/context/UsageContext';
import ReactMarkdown from 'react-markdown';

type QualityReport = {
  reasoning: string;
  human_quality_score: number;
};

export default function InterpreterPage() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [copyButtonText, setCopyButtonText] = useState('Copia'); // Stato per il pulsante Copia

  const { interpreter_profiles: allowedProfiles, fetchUserStatus } = useUsage();

  const interpreterProfileOptions = [
      "Spiega in Parole Semplici", "Analista Contratto di Vendita", "Revisore Contratto di Acquisto",
      "Estrattore P&L Aziendale", "Analista Bilancio Aziendale", "Sintesi Legale Breve",
      "Revisore Polizza Assicurativa", "Verificatore Fatture/Bollette",
      "Estrattore Dati Fatti", "Analista Debiti/Liquidità"
  ];
  const [selectedProfile, setSelectedProfile] = useState(interpreterProfileOptions[0]);

  const { isSignedIn, getToken, signOut } = useAuth();
  const router = useRouter();

  const handleInterpret = async () => {
    if (!isSignedIn) {
      alert("Devi essere autenticato per usare l'Interpreter.");
      router.push('/login');
      return;
    }
    if (!inputText.trim()) {
      alert('Per favore, inserisci del testo da interpretare.');
      return;
    }
    setIsLoading(true);
    setOutputText('Analisi in corso...');
    setQualityReport(null);

    try {
      const token = await getToken();
      if (!token) throw new Error("Impossibile ottenere il token.");

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/interpret`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ text: inputText, profile_name: selectedProfile }),
      });

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 401) {
          signOut(() => router.push('/login'));
          return;
        }
        throw new Error(data.detail || 'Si è verificato un errore.');
      }
      
      setOutputText(data.interpreted_text);
      setQualityReport(data.quality_report);
      await fetchUserStatus(); 
      
    } catch (error) {
        if (error instanceof Error) setOutputText(`Errore: ${error.message}`);
        else setOutputText('Si è verificato un errore sconosciuto.');
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
    <main className="flex flex-col items-center bg-gray-900 px-4 md:px-8 text-white">
      <div className="w-full max-w-4xl">
          <header className="mb-8 text-center">
              <h1 className="text-4xl font-extrabold text-blue-400">Interpreter AI</h1>
              <p className="mt-2 text-lg text-gray-400 font-semibold">
                  Trasforma documenti complessi in sintesi chiare e strutturate.
              </p>
          </header>

          <div className="mb-6">
              <label htmlFor="aiProfile" className="block text-sm font-medium text-gray-300 mb-2">
                  Seleziona Profilo di Analisi
              </label>
              <select
                  id="aiProfile"
                  value={selectedProfile}
                  onChange={(e) => setSelectedProfile(e.target.value)}
                  disabled={isLoading}
                  className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                  {interpreterProfileOptions.map((profile) => {
                      const isDisabled = !!allowedProfiles && allowedProfiles !== 'all' && !allowedProfiles.includes(profile);
                      return (
                          <option key={profile} value={profile} disabled={isDisabled} className={isDisabled ? 'text-gray-500' : ''}>
                              {profile}
                              {isDisabled ? ' 🔒 (Upgrade)' : ''}
                          </option>
                      );
                  })}
              </select>
          </div>
          
		  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
		    {/* Pannello di Input */}
		    <div>
			  <div className="flex justify-between items-center mb-2">
			    <label htmlFor="inputText" className="block text-sm font-medium text-gray-300">
				  Incolla qui il tuo documento
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
			    rows={15} // <-- RIPRISTINIAMO L'ALTEZZA FISSA
			    className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-blue-500 focus:ring-blue-500 resize-none" // Manteniamo resize-none
			    placeholder="Incolla il testo del contratto..."
			    value={inputText}
			    onChange={(e) => setInputText(e.target.value)}
			    disabled={isLoading}
			  />
		    </div>

		    {/* Pannello di Output */}
		    <div>
			  <div className="flex justify-between items-center mb-2">
			    <label className="block text-sm font-medium text-gray-300">
				  Analisi Strutturata
			    </label>
			    <button
				  onClick={handleCopy}
  				  disabled={!outputText || isLoading}
    				  className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-gray-600 transition-colors disabled:bg-gray-800 disabled:cursor-not-allowed"
  			    >
				  {copyButtonText}
			    </button>
			  </div>
			  {/* Usiamo un div con altezza fissa per contenere il Markdown */}
			  <div 
			    style={{ height: '385px' }} // <-- Altezza fissa in pixel per combaciare perfettamente
			    className="w-full rounded-lg border border-gray-700 bg-gray-800 p-4 text-gray-300 shadow-inner overflow-y-auto prose prose-invert prose-p:text-gray-300 prose-strong:text-white prose-headings:text-blue-400"
			  >
			    <ReactMarkdown>
				  {outputText || "L'analisi del documento apparirà qui..."}
			    </ReactMarkdown>
			  </div>
		    </div>
		  </div>

          <div className="mt-6 flex justify-center">
            <button
              onClick={handleInterpret}
              disabled={isLoading || !inputText}
              className="rounded-xl bg-green-600 px-10 py-4 text-xl font-bold text-white shadow-lg hover:bg-green-500 transition-all disabled:bg-gray-700 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Analisi in corso...' : 'Spiega Documento'}
            </button>
          </div>

          {/* Il quality report rimane opzionale */}
          {qualityReport && (
		    <div className="mt-8 max-w-4xl mx-auto rounded-xl bg-gray-800 p-6 border border-gray-700 shadow-xl">
			  <h2 className="text-2xl font-semibold text-green-400 text-center mb-4">
			    Controllo di Qualità Esperto
			  </h2>
			  <div className="flex items-center justify-center text-center mb-4">
			    <p className="text-6xl font-bold text-green-400">{qualityReport.human_quality_score}</p>
			    <p className="ml-2 text-2xl text-gray-400">/ 100</p>
			  </div>
			  <p className="text-gray-300 italic text-center">
			    {qualityReport.reasoning}
			  </p>
		    </div>
		  )}
      </div>
    </main>
  );
}