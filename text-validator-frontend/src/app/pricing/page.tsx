// src/app/pricing/page.tsx
"use client";

import { useRouter } from 'next/navigation';

export default function PricingPage() {
  const router = useRouter();

  const handleCtaClick = (plan: string) => {
    if (plan === 'Business') {
      // Idealmente, questo linkerebbe a una pagina di contatto o aprirebbe un modale
      alert("Contattaci per una demo personalizzata e un preventivo per il tuo team.");
    } else {
      router.push('/login'); 
    }
  };
  
  // Strutture dati per la sezione di dettaglio dei profili
  const freeProfiles = {
    validator: ["Generico", "L'Umanizzatore", "Social Media Manager B2B", "Ottimizzatore Email di Vendita"],
    interpreter: ["Spiega in Parole Semplici"]
  };

  const starterProfiles = {
    strategist: ["Sviluppatore di Buyer Persona", "Ideatore di Pillar Page e Content Cluster", "Generatore di Brief Creativo per Campagne"],
    validator: [
        "Tutti i profili Free", "Copywriter Persuasivo", "Scrittore di Newsletter", "Generatore Descrizioni Prodotto",
        "Scrittore Testi per Landing Page", "Redattore di Annunci di Lavoro", "Scrittore di Proposte Commerciali",
        "Redattore di Sezioni di Business Plan", "Comunicatore di Crisi PR"
    ],
    interpreter: ["Tutti i profili Free", "Sintetizzare di Meeting e Trascrizioni", "Analizzatore di Feedback dei Clienti", "Estrattore di Dati Strutturati"],
    compliance: ["Verificatore Anti-Bias Annunci Lavoro", "Analizzatore Disclaimer E-commerce"]
  };


  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-7xl">
        <header className="mb-16 text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-center text-blue-400 mb-6">
                Un Piano per Ogni Fase della Tua Crescita
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                Dall'idea iniziale all'efficienza di team, la nostra suite si adatta alle tue ambizioni.
            </p>
        </header>

        {/* SEZIONE DEI PIANI A 4 LIVELLI */}
        <section className="text-center mb-16">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 items-stretch">
                
                {/* Piano Free */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 h-full flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Free</h3>
                        <p className="text-gray-300 text-lg mb-6 h-25">L'ideale per professionisti curiosi che vogliono risolvere problemi di testo quotidiani.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">€0<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>5 Chiamate/giorno</strong> (su Validator)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Limite 1.500 caratteri</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>4 Profili Validator</strong> selezionati</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>1 Profilo Interpreter</strong> (teaser)</span></li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span><span>Accesso a Strategist & Compliance</span></li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span><span>Controllo di Qualità AI²™</span></li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span><span>Voci Personalizzate (CTOV)</span></li>
                        </ul>
                    </div>
                    <button onClick={() => handleCtaClick('Free')} className="w-full bg-gray-600 hover:bg-gray-500 text-white font-bold py-3 px-6 rounded-full transition-all">
                        Inizia Ora
                    </button>
                </div>

                {/* Piano Starter (In Evidenza) */}
                <div className="bg-blue-900 p-8 rounded-2xl shadow-2xl border-2 border-blue-500 h-full flex flex-col justify-between transform lg:scale-105">
                    <div>
                        <h3 className="text-4xl font-bold text-white mb-2">Starter</h3>
                        <p className="text-blue-200 text-lg mb-6 h-25">Il toolkit essenziale per freelancer e piccoli team che vogliono produrre contenuti di alta qualità.</p>
                        <p className="text-6xl font-extrabold text-white mb-6">€29<span className="text-xl text-gray-400">/mese</span></p>
                        <ul className="text-blue-100 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>20 Chiamate/giorno</strong> (condivise)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Limite 15.000 caratteri</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>12 Profili Validator</strong> avanzati</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>4 Profili Interpreter</strong> per l'analisi</span></li>
                            <li className="flex items-start"><span className="text-yellow-400 mr-2">★</span><span><strong>Assaggio di Strategist & Compliance</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Controllo di Qualità AI²™</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Fino a 2</strong> Voci Personalizzate (CTOV)</span></li>
                        </ul>
                    </div>
                    <button onClick={() => handleCtaClick('Starter')} className="w-full bg-white text-blue-700 font-bold py-3 px-6 rounded-full shadow-lg hover:bg-gray-100 transition-all">
                        Scegli Starter
                    </button>
                </div>

                {/* Piano Pro */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 h-full flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Pro</h3>
                        <p className="text-gray-300 text-lg mb-6 h-25">Per power user e specialisti che integrano l'AI nel loro workflow per compiti strategici e mission-critical.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">€79<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>150 Chiamate/giorno</strong> (condivise)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Limite 100.000 caratteri</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><strong className="text-white">TUTTI i 65+ Profili Inclusi</strong></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Accesso completo a <strong>Strategist AI</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Accesso completo a <strong>Compliance Checkr</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Controllo di Qualità AI²™</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Fino a 10</strong> Voci Personalizzate (CTOV)</span></li>
                        </ul>
                    </div>
                    <button onClick={() => handleCtaClick('Pro')} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition-all">
                        Scegli Pro
                    </button>
                </div>

                {/* Piano Business */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 h-full flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Business</h3>
                        <p className="text-gray-300 text-lg mb-6 h-25">Per team, PMI e agenzie che necessitano di collaborazione, governance e scalabilità.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">Su richiesta</p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Tutto del piano Pro</strong>, e in più:</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Limiti di utilizzo personalizzati</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Gestione centralizzata degli utenti</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Profili CTOV condivisi per il team</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Fatturazione centralizzata</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Supporto prioritario</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span>Accesso API (in futuro)</span></li>
                        </ul>
                    </div>
                    <button onClick={() => handleCtaClick('Business')} className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-full transition-all">
                        Contatta le Vendite
                    </button>
                </div>

            </div>
        </section>

        {/* SEZIONE DETTAGLIO PROFILI */}
        <section className="mt-20 w-full">
            <h2 className="text-3xl font-bold text-center text-blue-300 mb-10">
                Dettaglio Profili Inclusi nei Piani Free & Starter
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Colonna Piano Free */}
                <div className="bg-gray-800 p-8 rounded-lg border border-gray-700">
                    <h3 className="text-2xl font-semibold text-white mb-6">Piano Free</h3>
                    <div className="mb-6">
                        <h4 className="font-bold text-gray-200 mb-2">Validator ({freeProfiles.validator.length} Profili)</h4>
                        <ul className="list-disc list-inside text-gray-400 space-y-1">{freeProfiles.validator.map(p => <li key={p}>{p}</li>)}</ul>
                    </div>
                    <div>
                        <h4 className="font-bold text-gray-200 mb-2">Interpreter ({freeProfiles.interpreter.length} Profilo)</h4>
                        <ul className="list-disc list-inside text-gray-400 space-y-1">{freeProfiles.interpreter.map(p => <li key={p}>{p}</li>)}</ul>
                    </div>
                </div>

                {/* Colonna Piano Starter */}
                <div className="bg-gray-800 p-8 rounded-lg border border-gray-700">
                    <h3 className="text-2xl font-semibold text-white mb-6">Piano Starter</h3>
                    <div className="space-y-6">
                        <div>
                            <h4 className="font-bold text-purple-300 mb-2">Strategist (Assaggio di {starterProfiles.strategist.length} Profili)</h4>
                            <ul className="list-disc list-inside text-gray-400 space-y-1">{starterProfiles.strategist.map(p => <li key={p}>{p}</li>)}</ul>
                        </div>
                         <div>
                            <h4 className="font-bold text-blue-300 mb-2">Validator ({starterProfiles.validator.length-1} Profili Aggiuntivi)</h4>
                            <ul className="list-disc list-inside text-gray-400 space-y-1">{starterProfiles.validator.map(p => <li key={p}>{p}</li>)}</ul>
                        </div>
                        <div>
                            <h4 className="font-bold text-green-300 mb-2">Interpreter ({starterProfiles.interpreter.length-1} Profili Aggiuntivi)</h4>
                            <ul className="list-disc list-inside text-gray-400 space-y-1">{starterProfiles.interpreter.map(p => <li key={p}>{p}</li>)}</ul>
                        </div>
                        <div>
                            <h4 className="font-bold text-yellow-300 mb-2">Compliance Checkr (Assaggio di {starterProfiles.compliance.length} Profili)</h4>
                            <ul className="list-disc list-inside text-gray-400 space-y-1">{starterProfiles.compliance.map(p => <li key={p}>{p}</li>)}</ul>
                        </div>
                    </div>
                </div>
            </div>
            <p className="text-center text-gray-400 mt-8">I piani <strong>Pro</strong> e <strong>Business</strong> includono l'accesso a tutti i 65+ profili disponibili nella suite.</p>
        </section>

      </div>
    </main>
  );
}