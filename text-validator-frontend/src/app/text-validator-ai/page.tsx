// src/app/text-validator-ai/page.tsx
"use client";

import Link from 'next/link';

export default function TextValidatorAIPage() {
  const validatorProfiles = [
    { name: "Generico", description: "Pulisce il markup e normalizza il tono per una comunicazione aziendale chiara e autorevole." },
    { name: "L'Umanizzatore", description: "Rende i testi generati dall'AI naturali, fluidi e autenticamente umani, rimuovendo ogni traccia di robotismo." },
    { name: "PM - Interpretazione Trascrizioni", description: "Estrae azioni, decisioni e responsabili da riunioni e note tecniche." },
    { name: "Copywriter Persuasivo", description: "Potenzia i testi di marketing evidenziando i benefici per il cliente." },
    { name: "Revisore Legale/Regolatorio", description: "Riformula testi per rimuovere ambiguità, garantendo precisione linguistica." },
    { name: "Scrittore di Newsletter", description: "Crea newsletter B2B coinvolgenti con ganci forti e chiare Call to Action." },
    { name: "Social Media Manager B2B", description: "Adatta i contenuti per post LinkedIn, rendendoli concisi e ottimizzati." },
    { name: "Comunicatore di Crisi PR", description: "Trasforma messaggi in comunicazioni di crisi: calme, empatiche e autorevoli." },
    { name: "Traduttore Tecnico IT", description: "Semplifica concetti tecnici complessi per un pubblico non specializzato." },
    { name: "Specialista Comunicazioni HR", description: "Adatta le comunicazioni interne ai dipendenti con un tono professionale ed empatico." },
    { name: "Ottimizzatore Email di Vendita", description: "Riscrive email a freddo B2B per massimizzare il tasso di risposta." },
  ];

  const interpreterProfiles = [
    { name: "Analista Contratto di Vendita", description: "Estrae clausole critiche, scadenze e rischi dai contratti di vendita." },
    { name: "Revisore Contratto di Acquisto", description: "Valuta le condizioni di acquisto e i termini di consegna con i fornitori." },
    { name: "Estrattore P&L Aziendale", description: "Calcola i margini di profitto chiave dal Conto Economico." },
    { name: "Analista Bilancio Aziendale", description: "Analizza i bilanci per la solvibilità e i rapporti di liquidità." },
    { name: "Sintesi Legale Breve", description: "Sintetizza sentenze e decisioni legali in formato 'Case Brief'." },
    { name: "Revisore Polizza Assicurativa", description: "Identifica massimali, esclusioni e clausole vessatorie nelle polizze." },
    { name: "Verificatore Fatture/Bollette", description: "Estrae dati contabili essenziali dalle fatture per l'automazione." },
    { name: "Estrattore Dati Fatti", description: "Estrae fatti, date ed entità da documenti per creare timeline oggettive." },
    { name: "Analista Debiti/Liquidità", description: "Calcola il capitale circolante e analizza i rischi di liquidità." },
    { name: "Spiega in Parole Semplici", description: "Trasforma documenti complessi in linguaggio manageriale chiaro." },
  ];
  
  const complianceProfiles = [
	{ name: "Analizzatore GDPR per Comunicazioni Marketing", description: "Verifica che le tue email e pop-up rispettino i requisiti di consenso e informativa del GDPR." },
	{ name: "Verificatore Anti-Bias per Annunci di Lavoro", description: "Identifica linguaggio potenzialmente discriminatorio nelle offerte di lavoro per genere, età o altri bias." },
	{ name: "Checker per Disclaimer Finanziari", description: "Controlla che le comunicazioni su prodotti di investimento includano i disclaimer di rischio obbligatori." },
	{ name: "Validatore di Claim Pubblicitari", description: "Segnala affermazioni iperboliche o non comprovabili che potrebbero essere contestate come pubblicità ingannevole." },
	{ name: "Revisore di Clausole per Termini di Servizio", description: "Analizza clausole per assicurare che spieghino in modo chiaro le conseguenze di un'azione per l'utente." },
	{ name: "Analizzatore di Green Claims (CSRD)", description: "Verifica che le dichiarazioni ambientali non siano generiche o ingannevoli (anti-greenwashing)." },
	{ name: "Revisore di Comunicazioni Mediche", description: "Controlla l'uso corretto della terminologia in ambito telemedicina e previene claim non autorizzati." },
	{ name: "Checker di Accessibilità Testuale (WCAG)", description: "Verifica requisiti testuali per la compatibilità con screen reader, come alt-text e link descrittivi." },
	{ name: "Verificatore Comunicazioni KYC/AML", description: "Analizza report interni per identificare pattern testuali sospetti o mancanze informative anti-riciclaggio." },
	{ name: "Analizzatore Disclaimer E-commerce", description: "Garantisce che i termini di garanzia e recesso distinguano correttamente tra clienti B2B e B2C." },
];

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
        <header className="mb-16 text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold text-blue-400 mb-4 leading-tight">
            Due Strumenti Potenti. Un'Unica Piattaforma.
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Text Validator AI non è solo uno strumento, ma una suite di soluzioni per garantire qualità, chiarezza e accuratezza in ogni testo.
          </p>
        </header>

        {/* --- SEZIONE REINSERITA: Come Funziona --- */}
        <section className="mb-16 py-16">
          <h2 className="text-4xl font-bold text-center text-blue-300 mb-12">Come Funziona il Doppio Controllo AI</h2>
          <div className="flex flex-col md:flex-row items-center justify-around space-y-8 md:space-y-0 md:space-x-8">
            <div className="md:w-1/2">
              <img 
                src="/placeholder-workflow-image.png"
                alt="Workflow in 3 passaggi: Pulizia, Normalizzazione, Valutazione Qualità" 
                className="rounded-lg shadow-xl w-full h-auto object-cover"
              />
            </div>
            <div className="md:w-1/2 space-y-6 text-left">
              <p className="text-gray-300 text-lg">
                Implementiamo un innovativo <strong>Doppio Controllo AI</strong>: una prima fase di <strong>Igiene e Normalizzazione</strong> trasforma il tuo testo grezzo in un contenuto professionale. Successivamente, un secondo <strong>Agente AI (LLM-as-a-Judge)</strong> valuta oggettivamente la qualità del risultato, assegnando un <strong>Punteggio di Qualità AI²</strong> per garantirne l'eccellenza.
              </p>
               <div className="text-center mt-8">
                  <Link href="/pricing" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition-all duration-300 shadow-md">
                      Scopri i Piani
                  </Link>
              </div>
            </div>
          </div>
        </section>
		
		{/* --- NUOVA SEZIONE: CUSTOM TONE OF VOICE --- */}
        <section className="mb-16 py-16 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
          <h2 className="text-4xl font-bold text-center text-blue-300 mb-0">Prendi il Controllo: Il Tuo Tone of Voice,</h2>
		  <h2 className="text-4xl font-bold text-center text-blue-300 mb-12">La Nostra AI</h2>
          <div className="flex flex-col md:flex-row-reverse items-center justify-around space-y-8 md:space-y-0 md:space-x-8 px-8">
            <div className="md:w-1/2">
              <img 
                src="/placeholder-ctov-image.png"
                alt="Interfaccia per la creazione di una Voce Personalizzata" 
                className="rounded-lg shadow-xl w-full h-auto object-cover"
              />
            </div>
            <div className="md:w-1/2 space-y-6 text-left">
              <h3 className="text-3xl font-semibold text-white">Custom Tone of Voice (CTOV)</h3>
              <p className="text-gray-300 text-lg">
                Vai oltre i profili standard. Con la funzionalità CTOV, puoi <strong>insegnare all'AI il tuo specifico stile di comunicazione aziendale</strong>. Definisci la missione, l'archetipo del brand e i termini da evitare per garantire che ogni testo validato sia perfettamente allineato con la tua identità.
              </p>
              <ul className="list-disc list-inside space-y-2 text-gray-300 pl-5">
				<li><strong>Coerenza Assoluta:</strong> Mantieni lo stesso stile su tutti i canali.</li>
				<li><strong>Controllo del Brand:</strong> Evita termini non approvati o gergo obsoleto.</li>
                <li><strong>Efficienza per i Team:</strong> Crea voci diverse per Marketing, HR e Legale.</li>
			  </ul>
            </div>
          </div>
        </section>
		
        {/* --- SEZIONE REINSERITA: Funzionalità Business --- */}
        <section className="mb-16 py-16">
		  <h2 className="text-4xl font-bold text-center text-blue-400 mb-12">Le Funzionalità che Trasformano il Tuo Business</h2>
		  <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
			{/* Blocco 1: Affidabilità e Conformità */}
			<div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
			  <h3 className="text-3xl font-semibold text-blue-300 mb-4">Zero Rischi: Garanzia di Compliance e Brand Safety.</h3>
			  <p className="text-gray-400 mb-6">Garantiamo che l'output sia formalmente corretto, mitigando il rischio di prompt injection e mantenendo la coerenza del tono per comunicazioni HR o di crisi.</p>
			  <ul className="list-disc list-inside space-y-2 text-gray-300 pl-5">
				<li><strong>No-Storage Policy:</strong> Massima conformità GDPR.</li>
				<li><strong>Robustezza:</strong> Filtraggio di input malevoli.</li>
			  </ul>
			  <img 
				src="/placeholder-compliance-image.png"
				alt="Illustrazione di sicurezza e compliance GDPR" 
				className="mt-8 rounded-lg shadow-md w-full h-48 object-cover"
			  />
			</div>

			{/* Blocco 2: Efficienza e Persuasione */}
			<div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
			  <h3 className="text-3xl font-semibold text-blue-300 mb-4">Massimizza la Conversione, Non la Revisione Manuale.</h3>
			  <p className="text-gray-400 mb-6">La nostra AI è specializzata nell'infondere empatia e coerenza del brand nei tuoi testi, dalle newsletter alle email di vendita.</p>
			  <ul className="list-disc list-inside space-y-2 text-gray-300 pl-5">
				<li><strong>Riduzione Editing:</strong> Riduci il tempo di post-editing.</li>
				<li><strong>Coerenza del Brand:</strong> Mantiene il tuo Tone of Voice.</li>
				<li><strong>Output Persuasivo:</strong> Genera bozze ottimizzate e coinvolgenti.</li>
			  </ul>
			  <img 
				src="/placeholder-persuasion-image.png"
				alt="Illustrazione di copywriting persuasivo ed efficienza" 
				className="mt-8 rounded-lg shadow-md w-full h-48 object-cover"
			  />
			</div>
		  </div>
		</section>

        {/* Sezione Validator */}
        <section className="py-16">
          <h2 className="text-4xl font-bold text-center text-blue-400 mb-4">Validator AI</h2>
          <p className="text-lg text-center text-gray-400 mb-12 max-w-2xl mx-auto">Perfeziona i tuoi testi. Pulisci il formato, normalizza il tono e rendi ogni comunicazione autentica e professionale.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {validatorProfiles.map((profile, index) => (
              <div key={`val-${index}`} className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
                <h3 className="text-2xl font-semibold text-white mb-3">{profile.name}</h3>
                <p className="text-gray-300">{profile.description}</p>
              </div>
            ))}
          </div>
        </section>
        
        {/* Sezione Interpreter */}
        <section className="py-16">
          <h2 className="text-4xl font-bold text-center text-green-400 mb-4">Interpreter AI</h2>
          <p className="text-lg text-center text-gray-400 mb-12 max-w-2xl mx-auto">Comprendi in pochi secondi. Estrai i punti chiave, analizza i rischi e decodifica documenti complessi, da contratti a bilanci.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {interpreterProfiles.map((profile, index) => (
              <div key={`int-${index}`} className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
                <h3 className="text-2xl font-semibold text-white mb-3">{profile.name}</h3>
                <p className="text-gray-300">{profile.description}</p>
              </div>
            ))}
          </div>
        </section>
		
		 {/* --- NUOVA SEZIONE PER IL COMPLIANCE CHECKR --- */}
		<section className="py-16">
		  <h2 className="text-4xl font-bold text-center text-yellow-400 mb-4">Compliance Checkr</h2>
		  <p className="text-lg text-center text-gray-400 mb-12 max-w-2xl mx-auto">Mitiga i rischi. Esegui un pre-screening normativo per identificare potenziali problemi di conformità prima della pubblicazione.</p>
		  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
			{complianceProfiles.map((profile, index) => (
			  <div key={`comp-${index}`} className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
				<h3 className="text-2xl font-semibold text-white mb-3">{profile.name}</h3>
				<p className="text-gray-300">{profile.description}</p>
			  </div>
			))}
		  </div>
		</section>
      </div>
    </main>
  );
}