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
      <div className="w-full max-w-5xl"> {/* Leggermente allargato per 3 card */}
        <header className="mb-16 text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-center text-blue-400 mb-6">
                Un Piano per Ogni Esigenza
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                Scegli il piano che sblocca il vero potenziale della tua comunicazione B2B, dalla prova gratuita alle funzionalità professionali.
            </p>
        </header>

        <section className="text-center mb-16">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-center">
                
                {/* Piano Free */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 h-full flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Free</h3>
                        <p className="text-gray-300 text-lg mb-6">Inizia a pulire i tuoi testi e a scoprire la qualità.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">€0<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8 pl-4 list-disc list-inside">
                            <li><strong>10</strong> Validazioni AI/giorno</li>
                            <li><strong>2 Profili AI:</strong></li>
                            <li className="ml-4">- Generico</li>
                            <li className="ml-4">- L'Umanizzatore</li>
                            <li>Report di Qualità Umana™</li>
                            <li>Supporto via email</li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="bg-gray-600 hover:bg-gray-500 text-white font-bold py-3 px-6 rounded-full transition-all duration-300">
                        Inizia Gratis
                    </button>
                </div>

                {/* Piano Starter (In Evidenza) */}
                <div className="bg-blue-900 p-8 rounded-2xl shadow-2xl border-2 border-blue-500 h-full flex flex-col justify-between transform md:scale-110">
                    <div>
                        <h3 className="text-4xl font-bold text-white mb-2">Starter</h3>
                        <p className="text-blue-200 text-lg mb-6">Il modo migliore per sbloccare tutti i profili AI.</p>
                        <p className="text-6xl font-extrabold text-white mb-6">€9,99<span className="text-xl text-gray-400">/mese</span></p>
                        <ul className="text-blue-100 text-left space-y-3 mb-8 pl-4 list-disc list-inside">
                            <li><strong>30</strong> Validazioni AI/giorno</li>
                            <li><strong>Tutti gli 11 Profili AI</strong></li>
                            <li>Report di Qualità Umana™</li>
                            <li>Supporto prioritario</li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="bg-white text-blue-700 font-bold py-3 px-6 rounded-full shadow-lg hover:bg-gray-100 transition-all duration-300">
                        Scegli Starter
                    </button>
                </div>

                {/* Piano Pro */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 h-full flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Pro</h3>
                        <p className="text-gray-300 text-lg mb-6">Per professionisti e team con volumi elevati.</p>
                        <p className="text-5xl font-extrabold text-white mb-6">€49<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8 pl-4 list-disc list-inside">
                            <li><strong>200</strong> Validazioni AI/giorno</li>
                            <li><strong>Tutti gli 11 Profili AI</strong></li>
                            <li>Funzionalità future (API, CTOV)</li>
                            <li>Supporto dedicato</li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition-all duration-300">
                        Scegli Pro
                    </button>
                </div>
            </div>
        </section>
      </div>
    </main>
  );
}