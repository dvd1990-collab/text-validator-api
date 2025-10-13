// src/app/text-validator-ai/page.tsx
"use client";

import Link from 'next/link';

export default function TextValidatorAIPage() {
  const profileDescriptions = [
    { name: "Generico", description: "Pulisce il markup e normalizza il tono per una comunicazione aziendale chiara e autorevole." },
    { name: "PM - Interpretazione Trascrizioni", description: "Estrae azioni, decisioni e responsabili da riunioni e note tecniche, strutturando il testo per la chiarezza progettuale." },
    { name: "Copywriter Persuasivo", description: "Potenzia i testi di marketing evidenziando i benefici per il cliente con un tono energico e orientato alla conversione." },
    { name: "Revisore Legale/Regolatorio", description: "Riformula testi per rimuovere ambiguità, garantendo precisione linguistica e conformità formale a standard legali e regolatori." },
    { name: "Scrittore di Newsletter", description: "Crea newsletter B2B coinvolgenti, strutturate con ganci forti e chiare Call to Action, mantenendo un tono professionale ma conversazionale." },
    { name: "Social Media Manager B2B", description: "Adatta i contenuti per post su piattaforme professionali (es. LinkedIn), rendendoli concisi, leggibili e ottimizzati per l'engagement." },
    { name: "Comunicatore di Crisi PR", description: "Trasforma i messaggi in comunicazioni ufficiali di crisi: calme, empatiche, autorevoli e prive di speculazioni." },
    { name: "Traduttore Tecnico IT", description: "Semplifica concetti tecnici complessi in linguaggio comprensibile per un pubblico non specializzato (manager, clienti), focalizzandosi su benefici e implicazioni pratiche." },
    { name: "Specialista Comunicazioni HR", description: "Adatta le comunicazioni interne ai dipendenti, garantendo un tono professionale, empatico, inclusivo e conforme alle policy aziendali." },
    { name: "Ottimizzatore Email di Vendita", description: "Riscrive email a freddo B2B per massimizzare il tasso di risposta, con personalizzazione e Call to Action a basso attrito." },
    { name: "L'Umanizzatore", description: "Rende i testi generati dall'AI naturali, fluidi e autenticamente umani, rimuovendo ogni traccia di robotismo per una comunicazione più coinvolgente." },
  ];

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
        <header className="mb-16 text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold text-blue-400 mb-4 leading-tight">
            L'Unica Soluzione AI per Testi B2B Autentici e di Qualità Verificata.
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Text Validator AI trasforma il contenuto generato automaticamente in messaggi professionali, coerenti e indistinguibili da quelli umani, <strong>validandone la qualità tramite la nostra metodologia proprietaria.</strong>
          </p>
        </header>

		{/* Sezione: Cosa facciamo e come lo certifichiamo */}
        <section className="mb-16 py-16 bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
          <h2 className="text-4xl font-bold text-center text-blue-300 mb-10">Il Nostro Vantaggio: Il Doppio Controllo AI</h2>
          <div className="flex flex-col md:flex-row items-center justify-around space-y-8 md:space-y-0 md:space-x-8">
            <div className="md:w-1/2">
              <img 
                src="/placeholder-workflow-image.png"
                alt="Workflow in 3 passaggi: Pulizia, Normalizzazione, Valutazione Qualità" 
                className="rounded-lg shadow-xl w-full h-auto object-cover"
              />
              <p className="text-sm text-gray-500 mt-2 text-center">
                (Il processo: Input caotico &gt; Igiene Prompt &gt; Normalizzazione Tono &gt; Valutazione Qualità Umana™)
              </p>
            </div>
            <div className="md:w-1/2 space-y-6 text-left">
              {/* MODIFICA: Primo paragrafo per rimuovere i doppi asterischi */}
              <p className="text-gray-300 text-lg">
                Text Validator AI non si limita a pulire il testo. Implementiamo un innovativo <strong>Doppio Controllo AI</strong>:
                una prima fase di <strong>Igiene del Prompt e Normalizzazione del Tono</strong> trasforma il tuo testo grezzo in un contenuto professionale e on-brand.
                Successivamente, un secondo <strong>Agente AI (LLM-as-a-Judge)</strong> valuta oggettivamente la qualità del risultato, assegnando un <strong>Punteggio di Qualità Umana™</strong> per <strong>valutarne e garantirne l'eccellenza.</strong>
              </p>
              <p className="text-gray-300 text-lg">
                Questo processo garantisce che ogni output sia non solo privo di errori e formattazione "AI-ish", ma anche efficace, coerente e indistinguibile da un testo scritto da un esperto umano.
              </p>
               <div className="text-center mt-8">
                  <Link href="/pricing" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition-all duration-300 shadow-md">
                      Scopri i Piani
                  </Link>
              </div>
            </div>
          </div>
        </section>

        {/* III. Sezione Funzionalità (Value-Based Focus) */}
        <section className="mb-16 py-16">
          <h2 className="text-4xl font-bold text-center text-blue-400 mb-12">Le Funzionalità che Trasformano il Tuo Business</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
              <h3 className="text-3xl font-semibold text-blue-300 mb-4">Zero Rischi: Garanzia di Compliance e Brand Safety.</h3>
              <p className="text-gray-400 mb-6">Il caos di formattazione nei file di configurazione, i messaggi HR sensibili o i testi legali non revisionati danneggiano la tua reputazione. Garantiamo che l'output sia formalmente corretto e robusto, mitigando il rischio di prompt injection.</p>
              <ul className="list-disc list-inside space-y-2 text-gray-300 pl-5">
                <li><strong>Compliance Interna:</strong> Coerenza del tono per le comunicazioni di crisi e HR.</li>
                <li><strong>No-Storage Policy:</strong> Il tuo testo non viene mai conservato, garantendo la massima conformità GDPR.</li>
                <li><strong>Robustezza:</strong> Filtraggio di input malevoli (Prompt Injection Mitigation).</li>
              </ul>
              <img 
                src="/placeholder-compliance-image.png"
                alt="Illustrazione di sicurezza e compliance GDPR" 
                className="mt-8 rounded-lg shadow-md w-full h-48 object-cover"
              />
              <p className="text-sm text-gray-500 mt-2 text-center">
                (Illustrazione: Sicurezza e conformità GDPR con "No Storage Policy")
              </p>
            </div>

            <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
              <h3 className="text-3xl font-semibold text-blue-300 mb-4">Massimizza la Conversione, Non la Revisione Manuale.</h3>
              <p className="text-gray-400 mb-6">I copywriter e i SMM non possono sprecare tempo prezioso a "umanizzare" l'output di un LLM. La nostra AI è specializzata nell'infondere empatia, coerenza del brand e tocco persuasivo nei tuoi testi, dalle newsletter alle email di vendita, assicurando un risultato finale convincente.</p>
              <ul className="list-disc list-inside space-y-2 text-gray-300 pl-5">
                <li><strong>Riduzione Editing:</strong> Riduci il tempo di post-editing manuale fino al 50%.</li>
                <li><strong>Coerenza del Brand:</strong> Mantiene il tuo Tone of Voice specifico in tutti i canali.</li>
                <li><strong>Output Persuasivo:</strong> Genera bozze ottimizzate, coinvolgenti e mirate.</li>
              </ul>
              <img 
                src="/placeholder-persuasion-image.png"
                alt="Illustrazione di copywriting persuasivo ed efficienza" 
                className="mt-8 rounded-lg shadow-md w-full h-48 object-cover"
              />
              <p className="text-sm text-gray-500 mt-2 text-center">
                (Illustrazione: Copywriting persuasivo e riduzione dell'editing manuale)
              </p>
            </div>
          </div>
        </section>
		
        {/* Sezione: Elenco Profili AI */}
        <section className="py-16">
          <h2 className="text-4xl font-bold text-center text-blue-400 mb-12">I Nostri 11 Profili Esperti al Tuo Servizio</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {profileDescriptions.map((profile, index) => (
              <div key={index} className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 text-center">
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