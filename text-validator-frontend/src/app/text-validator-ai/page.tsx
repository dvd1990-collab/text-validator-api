// src/app/text-validator-ai/page.tsx
"use client";

import Link from 'next/link';

export default function TextValidatorAIPage() {
  const validatorProfiles = [
    { name: "Generico", description: "Pulisce il markup e normalizza il tono per una comunicazione aziendale chiara e autorevole." },
    { name: "L'Umanizzatore", description: "Rende i testi generati dall'AI naturali, fluidi e autenticamente umani, rimuovendo ogni traccia di robotismo." },
    { name: "PM - Interpretazione Trascrizioni", description: "Estrae azioni, decisioni e responsabili da riunioni e note tecniche, strutturando il testo per la chiarezza progettuale." },
    { name: "Copywriter Persuasivo", description: "Potenzia i testi di marketing evidenziando i benefici per il cliente con un tono energico e orientato alla conversione." },
    { name: "Revisore Legale/Regolatorio", description: "Riformula testi per rimuovere ambiguità, garantendo precisione linguistica e conformità formale." },
    { name: "Scrittore di Newsletter", description: "Crea newsletter B2B coinvolgenti, strutturate con ganci forti e chiare Call to Action." },
    { name: "Social Media Manager B2B", description: "Adatta i contenuti per post su piattaforme professionali (es. LinkedIn), rendendoli concisi e ottimizzati per l'engagement." },
    { name: "Comunicatore di Crisi PR", description: "Trasforma i messaggi in comunicazioni ufficiali di crisi: calme, empatiche e autorevoli." },
    { name: "Traduttore Tecnico IT", description: "Semplifica concetti tecnici complessi in linguaggio comprensibile per un pubblico non specializzato." },
    { name: "Specialista Comunicazioni HR", description: "Adatta le comunicazioni interne ai dipendenti, garantendo un tono professionale, empatico e inclusivo." },
    { name: "Ottimizzatore Email di Vendita", description: "Riscrive email a freddo B2B per massimizzare il tasso di risposta, con personalizzazione e Call to Action a basso attrito." },
  ];

  const interpreterProfiles = [
    { name: "Analista Contratto di Vendita", description: "Estrae clausole critiche, scadenze e rischi finanziari dai tuoi contratti di vendita." },
    { name: "Revisore Contratto di Acquisto", description: "Valuta le condizioni di acquisto, i termini di consegna e le clausole di IP per i contratti con i fornitori." },
    { name: "Estrattore P&L Aziendale", description: "Calcola i margini di profitto chiave (lordo, operativo, netto) dal Conto Economico." },
    { name: "Analista Bilancio Aziendale", description: "Analizza i bilanci per la solvibilità, calcolando i rapporti di liquidità e leva finanziaria." },
    { name: "Sintesi Legale Breve", description: "Sintetizza sentenze e decisioni legali in formato standardizzato 'Case Brief' per lo studio." },
    { name: "Revisore Polizza Assicurativa", description: "Identifica massimali, franchigie, esclusioni e clausole potenzialmente vessatorie nelle polizze." },
    { name: "Verificatore Fatture/Bollette", description: "Estrae i dati contabili essenziali dalle fatture (numero, PO, data scadenza, totali) per l'automazione." },
    { name: "Estrattore Dati Fatti", description: "Estrae fatti, date, entità e luoghi da documenti forensi o dichiarazioni per creare una timeline oggettiva." },
    { name: "Analista Debiti/Liquidità", description: "Calcola il capitale circolante, analizza i rapporti di liquidità (Quick Ratio) e identifica i covenant di debito." },
    { name: "Spiega in Parole Semplici", description: "Trasforma documenti legali e finanziari complessi in linguaggio manageriale chiaro e conciso." },
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
      </div>
    </main>
  );
}