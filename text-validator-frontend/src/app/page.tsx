"use client";

import { useState } from 'react';

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

  const handleValidate = async () => {
    if (!inputText.trim()) {
      alert('Per favore, inserisci del testo da validare.');
      return;
    }
    setIsLoading(true);
    setOutputText('Elaborazione in corso...');
    setQualityReport(null);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
	  const response = await fetch(`${apiUrl}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Si è verificato un errore.');
      }
      const data = await response.json();
      setOutputText(data.normalized_text);
      setQualityReport(data.quality_report);
    } catch (error) { // <-- LA CORREZIONE È QUI: RIMOSSO ": any"
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
    if (!outputText || isLoading || outputText === 'Elaborazione in corso...') return;
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

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
        <header className="mb-8 text-center">
          {/* ... */}
        </header>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {/* ... */}
        </div>
        <div className="mt-6 flex justify-center">
          {/* ... */}
        </div>
        {qualityReport && (
          <div className="mt-8 rounded-lg bg-gray-800 p-6 border border-gray-700">
            {/* ... */}
          </div>
        )}
      </div>
    </main>
  );
}