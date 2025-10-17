// src/app/text-validator-ai/page.tsx
"use client";

import Link from 'next/link';

// --- NUOVE STRUTTURE DATI PER I PROFILI ---

const validatorProfiles = {
  "Comunicazione e PR": [
    { name: "Generico", description: "Pulisce il markup e normalizza il tono per una comunicazione aziendale chiara e autorevole." },
    { name: "L'Umanizzatore", description: "Rende i testi generati dall'AI naturali, fluidi e autenticamente umani, rimuovendo ogni traccia di robotismo." },
    { name: "Comunicatore di Crisi PR", description: "Trasforma messaggi in comunicazioni di crisi strategiche: calme, empatiche e autorevoli." }
  ],
  "Marketing e Vendite": [
    { name: "Copywriter Persuasivo", description: "Potenzia i testi di marketing per evidenziare i benefici e spingere alla conversione." },
    { name: "Scrittore di Newsletter", description: "Crea newsletter B2B coinvolgenti con un oggetto efficace e una CTA chiara." },
    { name: "Generatore Descrizioni Prodotto E-commerce", description: "Scrive descrizioni di prodotto ottimizzate per le vendite e la SEO." },
    { name: "Scrittore Testi per Landing Page", description: "Crea testi per landing page strutturati per massimizzare le conversioni." },
    { name: "Ottimizzatore Email di Vendita", description: "Riscrive email a freddo B2B per massimizzare il tasso di risposta." },
    { name: "Social Media Manager B2B", description: "Adatta i contenuti per post LinkedIn, rendendoli concisi e ottimizzati per l'engagement." }
  ],
  "Business e Strategia": [
    { name: "Analista Vantaggio Competitivo (UVP)", description: "Aiuta a definire il posizionamento unico dell'azienda sul mercato." },
    { name: "Redattore di Sezioni di Business Plan", description: "Aiuta a scrivere le sezioni chiave di un business plan professionale." },
    { name: "Scrittore di Proposte Commerciali", description: "Crea proposte strutturate e persuasive per clienti e partner." }
  ],
  "Risorse Umane": [
    { name: "Redattore di Annunci di Lavoro", description: "Guida alla creazione di job description dettagliate, inclusive ed efficaci." },
    { name: "Assistente Valutazioni Performance", description: "Fornisce modelli e frasi per scrivere feedback costruttivi." },
    { name: "Generatore di Policy Aziendali Interne", description: "Crea bozze di policy interne (es. smart working, codice etico)." },
    { name: "Scrittore di Manuale del Dipendente", description: "Assiste nella creazione di un manuale del dipendente completo e conforme." }
  ],
  "Documentazione Tecnica e Legale": [
    { name: "Traduttore Tecnico IT", description: "Traduce documentazione tecnica IT mantenendo la precisione terminologica." },
    { name: "Termini e Condizioni E-commerce", description: "Genera un modello di T&C per siti e-commerce." }
  ]
};

const interpreterProfiles = {
    "Produttività e Sintesi": [
        { name: "Spiega in Parole Semplici", description: "Semplifica testi complessi, tecnici o legali rendendoli comprensibili." },
        { name: "Sintetizzare di Meeting e Trascrizioni", description: "Estrae punti chiave, decisioni e azioni da trascrizioni di riunioni." }
    ],
    "Analisi Contratti e Documenti Legali": [
        { name: "Analista Contratto di Vendita", description: "Estrae e riassume clausole chiave, obblighi e rischi dai contratti di vendita." },
        { name: "Revisore Contratto di Acquisto", description: "Analizza i contratti di acquisto per identificare termini sfavorevoli." },
        { name: "Sintesi Legale Breve", description: "Crea riassunti concisi di documenti legali lunghi." },
        { name: "Revisore Polizza Assicurativa", description: "Analizza le polizze assicurative evidenziando coperture, esclusioni e massimali." },
        { name: "Analisi Contratti di Fornitura", description: "Revisiona i contratti dei fornitori evidenziando clausole critiche." }
    ],
    "Analisi Finanziaria e Contabile": [
        { name: "Estrattore P&L Aziendale", description: "Estrae dati chiave (ricavi, costi, profitti) da un conto economico." },
        { name: "Analista Bilancio Aziendale", description: "Sintetizza i punti salienti di un bilancio aziendale." },
        { name: "Verificatore Fatture/Bollette", description: "Controlla la correttezza di fatture e bollette e ne estrae i dati." },
        { name: "Analista Debiti/Liquidità", description: "Analizza la posizione debitoria e la liquidità da documenti finanziari." }
    ],
    "Business Intelligence e Dati": [
        { name: "Analizzatore di Feedback dei Clienti", description: "Identifica temi ricorrenti e sentiment da recensioni e sondaggi." },
        { name: "Estrattore di Dati Strutturati", description: "Estrae dati specifici (nomi, date, cifre, indirizzi) da testi non strutturati." },
        { name: "Sintetizzatore di Ricerche di Mercato", description: "Riassume report e articoli di settore per evidenziare trend e insight." }
    ],
    "Bandi e Finanziamenti": [
        { name: "Analisi di Capitolati di Gara e Bandi", description: "Estrae requisiti, scadenze e criteri di valutazione da bandi pubblici." }
    ],
    "Risorse Umane": [
        { name: "Assistente Valutazioni Performance", description: "Fornisce modelli e frasi per scrivere feedback costruttivi." },
        { name: "Generatore di Policy Aziendali Interne", description: "Crea bozze di policy interne (es. smart working, codice etico)." },
        { name: "Scrittore di Manuale del Dipendente", description: "Assiste nella creazione di un manuale del dipendente completo e conforme." }
    ]
};

const complianceProfiles = {
    "Marketing e Comunicazione": [
        { name: "Analizzatore GDPR Marketing", description: "Verifica la conformità delle comunicazioni di marketing al GDPR." },
        { name: "Validatore Claim Pubblicitari", description: "Controlla che le affermazioni pubblicitarie siano verificabili e non ingannevoli." },
        { name: "Analizzatore Disclaimer E-commerce", description: "Verifica la presenza e correttezza dei disclaimer obbligatori per l'e-commerce." }
    ],
    "Finanza e Antiriciclaggio (AML)": [
        { name: "Checker Disclaimer Finanziari", description: "Assicura che i disclaimer in comunicazioni finanziarie siano presenti e corretti." },
        { name: "Verificatore Comunicazioni KYC/AML", description: "Controlla che le comunicazioni ai clienti rispettino gli obblighi KYC/AML." },
        { name: "Generatore di Policy AML Interna", description: "Crea un modello di policy interna antiriciclagggio per entità non finanziarie." },
        { name: "Checker Adeguata Verifica Cliente (KYC)", description: "Guida interattiva per completare e documentare il processo di KYC." }
    ],
    "Legale e Contratti": [
        { name: "Revisore Clausole Termini di Servizio", description: "Analizza le clausole critiche nei Termini di Servizio." }
    ],
    "Risorse Umane": [
        { name: "Verificatore Anti-Bias Annunci Lavoro", description: "Analizza gli annunci di lavoro per rimuovere linguaggio discriminatorio." }
    ],
    "Sostenibilità (ESG/CSRD)": [
        { name: "Validatore di Green Claims (CSRD)", description: "Verifica i claim ambientali rispetto alle direttive UE (es. Green Claims)." },
        { name: "Generatore Report Sostenibilità (VSME)", description: "Guida alla creazione di un report di sostenibilità semplificato per le PMI." }
    ],
    "Web e Digitale": [
        { name: "Checker Accessibilità Testuale (WCAG)", description: "Verifica la conformità di un testo agli standard di accessibilità web." }
    ],
    "Bandi e Finanziamenti": [
        { name: "Validatore Formale Domanda di Bando", description: "Controlla la completezza formale della documentazione per bandi pubblici." }
    ],
    "Settori Regolamentati": [
        { name: "Revisore Comunicazioni Mediche", description: "Controlla la conformità di testi medici a normative specifiche di settore." }
    ]
};

export default function TextValidatorAIPage() {
  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-5xl">
        <header className="mb-16 text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold text-blue-400 mb-4 leading-tight">
            Fiducia, Chiarezza, Conformità e Coerenza.
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Text Validator AI è la suite completa per garantire qualità, coerenza e sicurezza in ogni testo aziendale.
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
          <p className="text-lg text-center text-gray-400 mb-12 max-w-3xl mx-auto">
            Perfeziona ogni testo destinato all'esterno. Il Validator non si limita a correggere la grammatica, ma trasforma bozze grezze, appunti o output di AI generici in comunicazioni professionali, coerenti e pronte per la pubblicazione. Umanizza lo stile, adatta il tono al canale specifico e garantisce che ogni parola rifletta l'identità del tuo brand.
          </p>
          {Object.entries(validatorProfiles).map(([category, profiles]) => (
            <div key={category} className="mb-10">
              <h3 className="text-2xl font-semibold text-blue-300 border-b-2 border-blue-800 pb-2 mb-6">{category}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {profiles.map((profile) => (
                  <div key={profile.name} className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                    <h4 className="text-xl font-bold text-white mb-2">{profile.name}</h4>
                    <p className="text-gray-400">{profile.description}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>
        
        {/* Sezione Interpreter */}
        <section className="py-16">
          <h2 className="text-4xl font-bold text-center text-green-400 mb-4">Interpreter AI</h2>
          <p className="text-lg text-center text-gray-400 mb-12 max-w-3xl mx-auto">
            Decodifica la complessità e accelera le decisioni. L'Interpreter analizza documenti lunghi e strutturati come contratti, bilanci o bandi di gara, estraendo le informazioni cruciali di cui hai bisogno. Trasforma il "legalese" e il "finanzese" in insight chiari e azionabili, identificando rischi, obblighi e opportunità nascoste in poche righe.
          </p>
           {Object.entries(interpreterProfiles).map(([category, profiles]) => (
            <div key={category} className="mb-10">
              <h3 className="text-2xl font-semibold text-green-300 border-b-2 border-green-800 pb-2 mb-6">{category}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {profiles.map((profile) => (
                  <div key={profile.name} className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                    <h4 className="text-xl font-bold text-white mb-2">{profile.name}</h4>
                    <p className="text-gray-400">{profile.description}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>
		
		{/* Sezione Compliance Checkr */}
		<section className="py-16">
		  <h2 className="text-4xl font-bold text-center text-yellow-400 mb-4">Compliance Checkr</h2>
		  <p className="text-lg text-center text-gray-400 mb-12 max-w-3xl mx-auto">
            Mitiga i rischi prima che diventino un problema. Il Compliance Checkr è il tuo consulente virtuale per il pre-screening normativo. Analizza le tue comunicazioni di marketing, gli annunci di lavoro e i disclaimer legali per identificare potenziali non conformità con le normative come GDPR, anti-bias e direttive sulla pubblicità, fornendo un report di rischio e suggerimenti pratici.
          </p>
		  {Object.entries(complianceProfiles).map(([category, profiles]) => (
            <div key={category} className="mb-10">
              <h3 className="text-2xl font-semibold text-yellow-300 border-b-2 border-yellow-800 pb-2 mb-6">{category}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {profiles.map((profile) => (
                  <div key={profile.name} className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                    <h4 className="text-xl font-bold text-white mb-2">{profile.name}</h4>
                    <p className="text-gray-400">{profile.description}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
		</section>
        
        {/* --- FINE SEZIONI PROFILI AGGIORNATE --- */}
      </div>
    </main>
  );
}