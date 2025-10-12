// src/app/pricing/page.tsx
"use client";

import { useRouter } from 'next/navigation';
// Rimosso Link, non serve più il pulsante Torna all'App diretto

export default function PricingPage() {
  const router = useRouter();

  // Funzione per il pulsante CTA finale, lo facciamo puntare al login per coerenza
  const handleCtaClick = () => {
    router.push('/login'); 
  };

  return (
    // La main tag avrà un padding top dal layout.tsx
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl"> {/* Allargato il max-w per un layout più ampio */}
        <header className="mb-16 text-center relative">
          {/* RIMOSSO: h1 "Pricing Trasparente per Qualità Inestimabile" */}
          {/* RIMOSSO: p "Scegli il piano che ti garantisce..." */}
          {/* RIMOSSO: Pulsante Torna all'App */}
        </header>

        {/* --- INIZIO: SEZIONE SPECIFICA PER I PRICING --- */}
        <section className="text-center mb-16 py-16">
            {/* Aggiungiamo un titolo qui per la pagina specifica, dato che la Navbar ha solo il titolo generale dell'app */}
            <h2 className="text-4xl font-bold text-center text-blue-400 mb-12">I Nostri Piani</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {/* Piano Free */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Free</h3>
                        <p className="text-gray-300 text-lg mb-6">Perfetto per test iniziali e piccole necessità.</p>
                        <p className="text-5xl font-extrabold text-green-400 mb-6">€x,xx<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8 pl-4 list-disc list-inside">
                            <li>5 Validazioni AI/giorno</li>
                            <li>Tutti i profili AI disponibili</li>
                            <li>Report di Qualità Umana™</li>
                            <li>Supporto email standard</li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition-all duration-300">
                        Inizia Gratis
                    </button>
                </div>

                {/* Piano Pro */}
                <div className="bg-blue-900 p-8 rounded-xl shadow-xl border border-blue-700 flex flex-col justify-between transform scale-105">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Pro</h3>
                        <p className="text-blue-200 text-lg mb-6">Per professionisti e team che cercano affidabilità.</p>
                        <p className="text-5xl font-extrabold text-green-400 mb-6">€x,xx<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-blue-100 text-left space-y-3 mb-8 pl-4 list-disc list-inside">
                            <li>200 Validazioni AI/giorno</li>
                            <li>Tutti i profili AI e futuri aggiornamenti</li>
                            <li>Report di Qualità Umana™ avanzato</li>
                            <li>Accesso API (su richiesta)</li>
                            <li>Supporto prioritario</li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="bg-white text-blue-700 font-bold py-3 px-6 rounded-full shadow-lg hover:bg-gray-100 transition-all duration-300">
                        Scegli Pro
                    </button>
                </div>

                {/* Piano Business */}
                <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700 flex flex-col justify-between">
                    <div>
                        <h3 className="text-3xl font-bold text-white mb-4">Business</h3>
                        <p className="text-gray-300 text-lg mb-6">Soluzioni personalizzate per grandi volumi.</p>
                        <p className="text-5xl font-extrabold text-green-400 mb-6">€x,xx<span className="text-xl text-gray-500">/mese</span></p>
                        <ul className="text-gray-400 text-left space-y-3 mb-8 pl-4 list-disc list-inside">
                            <li>Validazioni illimitate</li>
                            <li>Tutti i profili AI e funzionalità personalizzate</li>
                            <li>SLA (Service Level Agreement) dedicato</li>
                            <li>Integrazioni personalizzate</li>
                            <li>Account Manager dedicato</li>
                        </ul>
                    </div>
                    <button onClick={handleCtaClick} className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition-all duration-300">
                        Contattaci
                    </button>
                </div>
            </div>
        </section>
        {/* --- FINE: SEZIONE SPECIFICA PER I PRICING --- */}

      </div>
    </main>
  );
}