// src/app/compliance-checkr/page.tsx
"use client";
import { useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useUsage } from '@/context/UsageContext';
import ReactMarkdown from 'react-markdown';
// Nota: Questo componente non usa QualityReport, quindi l'abbiamo rimosso.
export default function ComplianceCheckrPage() {
const [inputText, setInputText] = useState('');
const [outputText, setOutputText] = useState('');
const [isLoading, setIsLoading] = useState(false);
const [copyButtonText, setCopyButtonText] = useState('Copia');
// Recuperiamo l'accesso specifico per questa feature dal nostro Context
const { complianceAccess, fetchUserStatus } = useUsage();
const router = useRouter();
// Array con i 10 nuovi profili di conformità
const complianceProfileGroups = {
    "Marketing e Comunicazione": ["Analizzatore GDPR per Comunicazioni Marketing", "Validatore Claim Pubblicitari", "Analizzatore Disclaimer E-commerce"],
    "Finanza e Antiriciclaggio (AML)": ["Checker Disclaimer Finanziari", "Verificatore Comunicazioni KYC/AML", "Generatore di Policy AML Interna", "Checker Adeguata Verifica Cliente (KYC)"],
    "Legale e Contratti": ["Revisore Clausole Termini di Servizio"],
    "Risorse Umane": ["Verificatore Anti-Bias Annunci Lavoro"],
    "Sostenibilità (ESG/CSRD)": ["Validatore di Green Claims (CSRD)", "Generatore Report Sostenibilità (VSME)"],
    "Web e Digitale": ["Checker Accessibilità Testuale (WCAG)"],
    "Bandi e Finanziamenti": ["Validatore Formale Domanda di Bando"],
    "Settori Regolamentati": ["Revisore Comunicazioni Mediche"],
};
const [selectedProfile, setSelectedProfile] = useState("Analizzatore GDPR Marketing");
const { isSignedIn, getToken, signOut } = useAuth();
const handleComplianceCheck = async () => {
if (!isSignedIn) {
alert("Devi essere autenticato per usare il Compliance Checkr.");
router.push('/login');
return;
}
if (!inputText.trim()) {
alert('Per favore, inserisci del testo da verificare.');
return;
}
setIsLoading(true);
setOutputText('Verifica di conformità in corso...');
try {
  const token = await getToken();
  if (!token) throw new Error("Impossibile ottenere il token.");

  // Chiamata al nuovo endpoint /compliance-check
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/compliance-check`, {
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
    // Gestisce il caso in cui l'utente non abbia accesso alla feature
    if (response.status === 403) {
        router.push('/pricing'); // Reindirizza alla pagina di pricing
        alert(data.detail || "Questa funzionalità non è inclusa nel tuo piano. Esegui l'upgrade per continuare.");
        return;
    }
    throw new Error(data.detail || 'Si è verificato un errore.');
  }
  
  // La risposta contiene direttamente il report di conformità
  setOutputText(data.compliance_report);
  await fetchUserStatus(); // Aggiorna il contatore globale
  
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
};
// Se l'utente non ha accesso (es. piano Free), mostriamo un messaggio di upgrade
if (!complianceAccess) {
return (
<main className="flex flex-col items-center justify-center min-h-screen bg-gray-900 px-4 md:px-8 text-white">
<div className="text-center p-8 bg-gray-800 rounded-lg shadow-lg border border-gray-700">
<h2 className="text-3xl font-bold text-red-400 mb-4">Accesso Limitato</h2>
<p className="text-lg text-gray-300 mb-6">
Il <strong>Compliance Checkr</strong> è una funzionalità avanzata disponibile solo per i piani a pagamento.
</p>
<button
onClick={() => router.push('/pricing')}
className="rounded-xl bg-blue-600 px-8 py-3 text-lg font-bold text-white shadow-lg hover:bg-blue-500 transition-all"
>
Esegui l'Upgrade
</button>
</div>
</main>
);
}
// Altrimenti, mostriamo la pagina normale
return (
<main className="flex flex-col items-center bg-gray-900 p-8 text-white">
<div className="w-full max-w-4xl">
<header className="mb-8 text-center">
<h1 className="text-4xl font-extrabold text-yellow-400">Compliance Checkr AI</h1>
<p className="mt-2 text-lg text-gray-400 font-semibold">
Verifica i tuoi testi per potenziali rischi di non conformità normativa.
</p>
</header>
<div className="mb-6">
          <label htmlFor="aiProfile" className="block text-sm font-medium text-gray-300 mb-2">
              Seleziona Profilo di Conformità
          </label>
          <select
              id="aiProfile"
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
              disabled={isLoading}
              className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-yellow-500 focus:ring-yellow-500"
          >
              {Object.entries(complianceProfileGroups).map(([groupName, profiles]) => (
                  <optgroup key={groupName} label={groupName}>
                      {profiles.map((profile) => (
                          <option key={profile} value={profile}>
                              {profile}
                          </option>
                      ))}
                  </optgroup>
              ))}
          </select>
        </div>
      
	  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
	    {/* Pannello di Input */}
	    <div>
		  <div className="flex justify-between items-center mb-2">
		    <label htmlFor="inputText" className="block text-sm font-medium text-gray-300">
			  Incolla il testo da verificare
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
		    className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 resize-none"
		    placeholder="Incolla qui l'annuncio di lavoro, l'email marketing o il claim pubblicitario..."
		    value={inputText}
		    onChange={(e) => setInputText(e.target.value)}
		    disabled={isLoading}
		  />
	    </div>

	    {/* Pannello di Output */}
	    <div>
		  <div className="flex justify-between items-center mb-2">
		    <label className="block text-sm font-medium text-gray-300">
			  Report di Conformità
		    </label>
		    <button
			  onClick={handleCopy}
			  disabled={!outputText || isLoading}
			  className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-gray-600 transition-colors disabled:bg-gray-800 disabled:cursor-not-allowed"
		    >
			  {copyButtonText}
		    </button>
		  </div>
		  <div 
		    style={{ height: '385px' }}
		    className="w-full rounded-lg border border-gray-700 bg-gray-800 p-4 text-gray-300 shadow-inner overflow-y-auto prose prose-invert prose-p:text-gray-300 prose-strong:text-white prose-headings:text-yellow-400"
		  >
		    <ReactMarkdown>
			  {outputText || "Il report di analisi apparirà qui..."}
		    </ReactMarkdown>
		  </div>
	    </div>
	  </div>

      <div className="mt-6 flex justify-center">
        <button
          onClick={handleComplianceCheck}
          disabled={isLoading || !inputText}
          className="rounded-xl bg-yellow-600 px-10 py-4 text-xl font-bold text-white shadow-lg hover:bg-yellow-500 transition-all disabled:bg-gray-700 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Analisi in corso...' : 'Verifica Conformità'}
        </button>
      </div>
      
      {/* Non c'è una sezione QualityReport separata per questo componente */}
  </div>
</main>
);
}