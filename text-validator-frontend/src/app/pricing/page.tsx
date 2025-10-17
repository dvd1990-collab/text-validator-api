// src/app/pricing/page.tsx
"use client";

import { useRouter } from 'next/navigation';

export default function PricingPage() {
  const router = useRouter();

  const handleCtaClick = () => {
    router.push('/login'); 
  };

  // Definiamo i profili qui per riutilizzarli nella sezione di dettaglio
  const freeProfiles = {
    validator: ["Generico", "L'Umanizzatore"],
    interpreter: ["Spiega in Parole Semplici"]
  };
  const starterProfiles = {
    validator: ["Generico", "L'Umanizzatore", "Analista Vantaggio Competitivo (UVP)", "Copywriter Persuasivo", "Scrittore di Newsletter", "Generatore Descrizioni Prodotto E-commerce", "Scrittore Testi per Landing Page", "Redattore di Annunci di Lavoro", "Assistente Valutazioni Performance"],
    interpreter: ["Spiega in Parole Semplici", "Analista Contratto di Vendita", "Revisore Contratto di Acquisto", "Estrattore P&L Aziendale", "Analista Bilancio Aziendale", "Verificatore Fatture/Bollette", "Analizzatore di Feedback dei Clienti", "Sintetizzare di Meeting e Trascrizioni", "Assistente Valutazioni Performance"]
  };

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-6xl">
        <header className="mb-16 text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-center text-blue-400 mb-6">
                Un Piano per Ogni Esigenza
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                Dalla valutazione base all'analisi strategica completa, scegli il piano che cresce con le tue necessità.
            </p>
        </header>

        {/* SEZIONE PRINCIPALE DEI PIANI */}
        <section className="text-center mb-16">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-stretch">
                
                {/* Piano Free */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 h-full flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Free</h3>
                        <p className="text-gray-300 text-lg mb-6">Per iniziare e testare le funzionalità essenziali.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">€0<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>3 Chiamate/giorno</strong> (condivise)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Limite 500 caratteri</strong> per input</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Validator:</strong> {freeProfiles.validator.length} Profili Base</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Interpreter:</strong> {freeProfiles.interpreter.length} Profilo Base</span></li>
							<li className="flex items-start"><span className="text-red-500 mr-2">✗</span><strong>Compliance Checkr</strong></li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span>Senza Controllo di Qualità AI²™</li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span><strong>Voci Personalizzate (CTOV)</strong></li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="w-full bg-gray-600 hover:bg-gray-500 text-white font-bold py-3 px-6 rounded-full transition-all duration-300">
                        Inizia Gratis
                    </button>
                </div>

                {/* Piano Starter (In Evidenza) */}
                <div className="bg-blue-900 p-8 rounded-2xl shadow-2xl border-2 border-blue-500 h-full flex flex-col justify-between transform lg:scale-110">
                    <div>
                        <h3 className="text-4xl font-bold text-white mb-2">Starter</h3>
                        <p className="text-blue-200 text-lg mb-6">Per professionisti che cercano efficienza e specializzazione.</p>
                        <p className="text-6xl font-extrabold text-white mb-6">€9,99<span className="text-xl text-gray-400">/mese</span></p>
                        <ul className="text-blue-100 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>10 Chiamate/giorno</strong> (condivise)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Limite 10.000 caratteri</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Validator:</strong> {starterProfiles.validator.length} Profili Inclusi</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Interpreter:</strong> {starterProfiles.interpreter.length} Profili Inclusi</span></li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span><span><strong>Compliance Checkr</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Con</strong> Controllo di Qualità AI²™</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><strong>Fino a 2 Voci Personalizzate (CTOV)</strong></li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="w-full bg-white text-blue-700 font-bold py-3 px-6 rounded-full shadow-lg hover:bg-gray-100 transition-all duration-300">
                        Scegli Starter
                    </button>
                </div>

                {/* Piano Pro */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 h-full flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Pro</h3>
                        <p className="text-gray-300 text-lg mb-6">Per team che necessitano della suite completa.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">€49<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>100 Chiamate/giorno</strong> (condivise)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Limite 50.000 caratteri</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Tutti i Profili</strong> Validator & Interpreter</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><strong className="text-white">Compliance Checkr INCLUSO</strong></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Con</strong> Controllo di Qualità AI²™</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><strong>Fino a 5 Voci Personalizzate (CTOV)</strong></li>
                            <li className="flex items-start"><span className="text-gray-500 mr-2">●</span>Accesso API e Funzionalità future</li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition-all duration-300">
                        Scegli Pro
                    </button>
                </div>
            </div>
        </section>

        {/* NUOVA SEZIONE: DETTAGLIO PROFILI */}
        <section className="mt-20 w-full">
            <h2 className="text-3xl font-bold text-center text-blue-300 mb-10">
                Dettaglio Profili Inclusi nei Piani
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Colonna Piano Free */}
                <div className="bg-gray-800 p-8 rounded-lg border border-gray-700">
                    <h3 className="text-2xl font-semibold text-white mb-6">Piano Free</h3>
                    <div className="mb-6">
                        <h4 className="font-bold text-gray-200 mb-2">Validator ({freeProfiles.validator.length} Profili)</h4>
                        <ul className="list-disc list-inside text-gray-400 space-y-1">
                            {freeProfiles.validator.map(p => <li key={p}>{p}</li>)}
                        </ul>
                    </div>
                    <div>
                        <h4 className="font-bold text-gray-200 mb-2">Interpreter ({freeProfiles.interpreter.length} Profilo)</h4>
                        <ul className="list-disc list-inside text-gray-400 space-y-1">
                            {freeProfiles.interpreter.map(p => <li key={p}>{p}</li>)}
                        </ul>
                    </div>
                </div>

                {/* Colonna Piano Starter */}
                <div className="bg-gray-800 p-8 rounded-lg border border-gray-700">
                    <h3 className="text-2xl font-semibold text-white mb-6">Piano Starter</h3>
                    <div className="mb-6">
                        <h4 className="font-bold text-gray-200 mb-2">Validator ({starterProfiles.validator.length} Profili)</h4>
                        <ul className="list-disc list-inside text-gray-400 space-y-1">
                            {starterProfiles.validator.map(p => <li key={p}>{p}</li>)}
                        </ul>
                    </div>
                    <div>
                        <h4 className="font-bold text-gray-200 mb-2">Interpreter ({starterProfiles.interpreter.length} Profili)</h4>
                        <ul className="list-disc list-inside text-gray-400 space-y-1">
                            {starterProfiles.interpreter.map(p => <li key={p}>{p}</li>)}
                        </ul>
                    </div>
                </div>
            </div>
        </section>

      </div>
    </main>
  );
}