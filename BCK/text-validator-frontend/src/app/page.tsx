"use client";

import { useState } from 'react';

// Definiamo il tipo per il nostro qualityReport per maggiore sicurezza con TypeScript
type QualityReport = {
  reasoning: string;
  human_quality_score: number;
};

export default function HomePage() {
  // --- STATO DEL COMPONENTE ---
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [copyButtonText, setCopyButtonText] = useState('Copia'); // <-- NUOVA RIGA

  // --- FUNZIONE PER CHIAMARE L'API ---
  const handleValidate = async () => {
    if (!inputText.trim()) {
      alert('Per favore, inserisci del testo da validare.');
      return;
    }

    setIsLoading(true);
    setOutputText('Elaborazione in corso...');
    setQualityReport(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Si è verificato un errore durante la validazione.');
      }

      const data = await response.json();
      setOutputText(data.normalized_text);
      setQualityReport(data.quality_report);

    } catch (error: any) {
      console.error('Errore nella chiamata API:', error);
      setOutputText(`Errore: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleCopy = () => {
	  // Non fare nulla se non c'è testo nell'output o se stiamo ancora caricando
	  if (!outputText || isLoading || outputText === 'Elaborazione in corso...') {
		return;
	  }

	  // Usiamo l'API del browser per copiare il testo
	  navigator.clipboard.writeText(outputText)
		.then(() => {
		  // Successo! Cambiamo il testo del pulsante per dare feedback
		  setCopyButtonText('Copiato!');
		  // Dopo 2 secondi, resettiamo il testo del pulsante
		  setTimeout(() => {
			setCopyButtonText('Copia');
		  }, 2000);
		})
		.catch(err => {
		  // Gestiamo un eventuale errore (raro, ma buona pratica)
		  console.error('Errore durante la copia negli appunti:', err);
		  alert('Impossibile copiare il testo. Verifica i permessi del tuo browser.');
		});
	};

  // --- STRUTTURA VISIVA (JSX) ---
  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
        
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-blue-400">Text Validator</h1>
          <p className="mt-2 text-gray-400">
            Pulisci, normalizza e valida la qualità dei tuoi testi in un solo click.
          </p>
        </header>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          
          <div>
            <label htmlFor="inputText" className="block text-sm font-medium text-gray-300 mb-2">
              Incolla qui il tuo testo
            </label>
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
            className="rounded-md bg-blue-600 px-8 py-3 text-lg font-semibold text-white shadow-sm hover:bg-blue-500 disabled:bg-gray-500 disabled:cursor-not-allowed"
            onClick={handleValidate}
            disabled={isLoading}
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