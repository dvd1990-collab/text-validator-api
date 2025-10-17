// src/app/page.tsx
"use client";

import { useState, useEffect } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useUsage } from '@/context/UsageContext'; // Importa il nostro hook del Context
import CTOVModal from '@/components/CTOVModal';

// Non importiamo più SettingsModal o GearIcon qui, sono gestiti dalla Navbar

type QualityReport = {
  reasoning: string;
  human_quality_score: number;
};

type SelectedProfile = {
    type: 'standard' | 'ctov';
    value: string; // "Generico" per standard, o l'ID per CTOV
};

export default function HomePage() {
  // Stati locali specifici di questa pagina
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [copyButtonText, setCopyButtonText] = useState('Copia');
  const [selectedProfile, setSelectedProfile] = useState<SelectedProfile>({ type: 'standard', value: 'Generico' });
  const [isModalOpen, setIsModalOpen] = useState(false); // Stato per aprire/chiudere il modale

  // Dati globali provenienti dal nostro Context
  const { 
    validator_profiles: allowedProfiles, 
    userTier, 
    fetchUserStatus,
    ctovAccess, // Permesso per creare CTOV
    ctovMaxProfiles, // Limite di profili
    ctovProfiles // Lista dei profili custom
  } = useUsage();

  // Hook di Clerk e Next.js necessari in questa pagina
  const { isSignedIn, getToken, signOut } = useAuth();
  const router = useRouter();
  
  const validatorProfileGroups = {
  "Comunicazione e PR": ["Generico", "L'Umanizzatore", "Comunicatore di Crisi PR"],
  "Marketing e Vendite": ["Copywriter Persuasivo", "Scrittore di Newsletter", "Generatore Descrizioni Prodotto E-commerce", "Scrittore Testi per Landing Page", "Ottimizzatore Email di Vendita", "Social Media Manager B2B"],
  "Business e Strategia": ["Analista Vantaggio Competitivo (UVP)", "Redattore di Sezioni di Business Plan", "Scrittore di Proposte Commerciali"],
  "Risorse Umane": ["Redattore di Annunci di Lavoro", "Assistente Valutazioni Performance", "Generatore di Policy Aziendali Interne", "Scrittore di Manuale del Dipendente"],
  "Documentazione Tecnica e Legale": ["Traduttore Tecnico IT", "Termini e Condizioni E-commerce"],
};

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
	  
	  const payload = {
        text: inputText,
        profile_name: selectedProfile.type === 'standard' ? selectedProfile.value : 'Generico', // Invia un nome standard
        ctov_profile_id: selectedProfile.type === 'ctov' ? selectedProfile.value : null // Invia l'ID solo se è un CTOV
      };
	  
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify(payload),
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
  
  // --- FUNZIONE PER GESTIRE IL CAMBIO NEL DROPDOWN ---
  const handleProfileChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    if (value === 'create_new') {
      setIsModalOpen(true);
    } else {
        const [type, val] = value.split(':');
        setSelectedProfile({ type: type as 'standard' | 'ctov', value: val });
    }
  };
  
  const canCreateNewVoice = ctovAccess && (ctovMaxProfiles === -1 || (ctovProfiles && ctovProfiles.length < (ctovMaxProfiles || 0)));

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
    <>
	<CTOVModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
          <header className="mb-8 text-center relative">
              {/* L'header è pulito: ingranaggio e contatore sono gestiti solo dalla Navbar */}
              <h1 className="text-4xl font-extrabold text-blue-400">Validator AI</h1>
              <p className="mt-2 text-gray-400 font-semibold">
                  Pulisci, normalizza e valida la qualità dei tuoi testi in un solo click.
              </p>
          </header>

          <div className="mb-6">
              <label htmlFor="aiProfile" className="block text-sm font-medium text-gray-300 mb-2">
                  Seleziona Profilo AI o Voce Personalizzata
              </label>
              <select
                  id="aiProfile"
                  value={`${selectedProfile.type}:${selectedProfile.value}`}
                  onChange={handleProfileChange}
                  disabled={isLoading}
                  className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                  {Object.entries(validatorProfileGroups).map(([groupName, profiles]) => (
                      <optgroup key={groupName} label={groupName}>
                          {profiles.map((profile) => {
                              const isDisabled = !!allowedProfiles && allowedProfiles !== 'all' && !allowedProfiles.includes(profile);
                              return (
                                  <option key={profile} value={`standard:${profile}`} disabled={isDisabled} className={isDisabled ? 'text-gray-500' : ''}>
                                      {profile} {isDisabled ? '🔒' : ''}
                                  </option>
                              );
                          })}
                      </optgroup>
                  ))}
                  
                  {ctovAccess && (
                      <optgroup label="Voci Personalizzate">
                          {ctovProfiles.map((profile) => (
                              <option key={profile.id} value={`ctov:${profile.id}`}>
                                  {profile.name}
                              </option>
                          ))}
                          <option value="create_new" disabled={!canCreateNewVoice} className="text-blue-400 font-semibold">
                              + Crea Nuova Voce { !canCreateNewVoice && '(Limite raggiunto)'}
                          </option>
                      </optgroup>
                  )}
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
			  <h2 className="text-2xl font-semibold text-blue-400">Report di Qualità AI²™</h2>
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
  </>
  );
}