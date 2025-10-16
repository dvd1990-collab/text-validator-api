// src/app/pricing/page.tsx
"use client";

import { useRouter } from 'next/navigation';

export default function PricingPage() {
  const router = useRouter();

  const handleCtaClick = () => {
    router.push('/login'); 
  };

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-5xl">
        <header className="mb-16 text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-center text-blue-400 mb-6">
                Un Piano per Ogni Esigenza
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                Scegli il piano che sblocca il vero potenziale della tua comunicazione, dalla validazione di base all'analisi documentale esperta.
            </p>
        </header>

        <section className="text-center mb-16">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-stretch">
                
                {/* Piano Free */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 h-full flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Free</h3>
                        <p className="text-gray-300 text-lg mb-6">Per iniziare e testare le funzionalità base.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">€0<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>3 Chiamate/giorno</strong> (condivise)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Limite 500 caratteri</strong> per input</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Validator:</strong> 2 Profili Base</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Interpreter:</strong> 1 Profilo Base</span></li>
							<li className="flex items-start"><span className="text-red-500 mr-2">✗</span><strong>Compliance Checkr</strong></li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span>Senza Controllo di Qualità AI²™</li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span><strong>Voci Personalizzate (CTOV)</strong></li> {/* <-- RIGA AGGIUNTA */}
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
                        <p className="text-blue-200 text-lg mb-6">Ideale per sbloccare tutta la potenza dell'AI.</p>
                        <p className="text-6xl font-extrabold text-white mb-6">€9,99<span className="text-xl text-gray-400">/mese</span></p>
                        <ul className="text-blue-100 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>10 Chiamate/giorno</strong> (condivise)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Limite 10.000 caratteri</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Validator:</strong> 2 Profili base + 3 advanced</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Interpreter:</strong> 1 Profilo base + 4 advanced</span></li>
                            <li className="flex items-start"><span className="text-red-500 mr-2">✗</span><span><strong>Compliance Checkr</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Con</strong> Controllo di Qualità AI²™</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><strong>Fino a 2 Voci Personalizzate (CTOV)</strong></li> {/* <-- RIGA AGGIUNTA */}
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
                        <p className="text-gray-300 text-lg mb-6">Per professionisti e team con volumi elevati.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">€49<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8">
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>100 Chiamate/giorno</strong> (condivise)</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Limite 50.000 caratteri</strong></span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Tutti i Profili</strong> Validator & Interpreter</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Compliance Checkr</strong> INCLUSO</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><span><strong>Con</strong> Controllo di Qualità AI²™</span></li>
                            <li className="flex items-start"><span className="text-green-500 mr-2">✓</span><strong>Fino a 5 Voci Personalizzate (CTOV)</strong></li> {/* <-- RIGA AGGIUNTA */}
                            <li className="flex items-start"><span className="text-gray-500 mr-2">●</span>Accesso API e Funzionalità future</li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition-all duration-300">
                        Scegli Pro
                    </button>
                </div>
            </div>
        </section>
      </div>
    </main>
  );
}