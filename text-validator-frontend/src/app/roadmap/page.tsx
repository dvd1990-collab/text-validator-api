// src/app/roadmap/page.tsx
"use client";

import Link from 'next/link';

export default function RoadmapPage() {
  const roadmapMilestones = [
    {
      date: "Q4 2025",
      title: "Foundation & Core Logic Launch (MVP 'AI² Verified')",
      description: "Lancio della versione minima funzionante (MVP). Workflow 'Doppio Controllo AI' attivo, focalizzato sulla normalizzazione del tono e il Punteggio di Qualità Umana™."
    },
	{
      date: "Q4 2025",
      title: "Enhanced AI Profiles & Customization",
      description: "Espansione della libreria di profili AI specializzati. Introduzione della funzionalità 'Custom Tone of Voice' per addestrare l'AI sul brand e lo stile unico del cliente."
    },
    {
      date: "Q1 2026",
      title: "Beta Program & Free Tier Activation",
      description: "Apertura delle registrazioni per un programma Beta esclusivo (max 500 utenti). Introduzione del piano Free con limiti chiari per la validazione del prodotto sul mercato e la raccolta di feedback qualitativo critico."
    },
    {
      date: "Q2 2026",
      title: "Official Launch & Pro/Business Plans",
      description: "Apertura completa delle registrazioni. Introduzione dei piani Pro e Business."
    },
    {
      date: "Q3 2026",
      title: "Advanced Analytics & AI Content Insights",
      description: "Dashboard avanzate per analizzare la qualità dei contenuti nel tempo. Strumenti per identificare pattern di 'AI-ishness' e suggerimenti proattivi per migliorare il Tone of Voice aziendale e le performance di comunicazione."
    },
  ];

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-4xl">
        <header className="mb-16 text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold text-blue-400 mb-4 leading-tight">
            La Nostra Roadmap: Il Futuro della Qualità dei Testi AI
          </h1>
          {/* INIZIO MODIFICHE: Descrizione iniziale */}
          <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-6">
            Questo progetto visionario nasce nel <strong>Q4 2025</strong> dall'iniziativa di un singolo professionista con oltre un decennio di esperienza nel settore IT. Mosso da una profonda passione per la tecnologia e l'intelligenza artificiale, l'obiettivo è costruire uno strumento che elevi la qualità delle comunicazioni B2B nell'era dell'AI. Ogni passo di questa roadmap è guidato da un impegno per l'innovazione, la sicurezza e l'eccellenza.
          </p>
          {/* FINE MODIFICHE: Descrizione iniziale */}
        </header>

        <section className="relative py-12">
          {/* Linea verticale più spessa e con un leggero gradiente */}
          <div className="absolute left-1/2 transform -translate-x-1/2 w-2 bg-gradient-to-b from-blue-600 to-gray-700 h-full rounded-full shadow-lg"></div>

          {roadmapMilestones.map((milestone, index) => (
            <div key={index} className={`flex items-center w-full mb-20 ${index % 2 === 0 ? 'flex-row-reverse' : ''}`}> {/* Aumentato mb-12 a mb-20 */}
              {/* Contenuto del milestone */}
              <div className="w-1/2 px-4 text-right md:text-left">
                <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 transition-all duration-300 hover:shadow-xl hover:border-blue-500 transform hover:-translate-y-1"> {/* Aggiunte animazioni/shadow */}
                  <h3 className="text-2xl font-bold text-blue-300 mb-2">{milestone.title}</h3>
                  <p className="text-gray-400">{milestone.description}</p>
                </div>
              </div>

              {/* Punto sulla linea più grande e animato */}
              <div className="relative w-1/2 flex justify-center items-center">
                <div className="absolute w-12 h-12 rounded-full bg-blue-600 border-4 border-gray-900 z-10 flex items-center justify-center text-xl font-bold text-white shadow-xl animate-pulse-once"> {/* Aumentato w-8 h-8 a w-12 h-12, aggiunto text-xl, shadow-xl, animate-pulse-once */}
                  {index + 1}
                </div>
                {/* Linea orizzontale di collegamento al punto, più evidente */}
                <div className={`absolute top-1/2 w-1/2 h-1 bg-blue-500 ${index % 2 === 0 ? 'right-0' : 'left-0'}`}></div>
              </div>

              {/* Data del milestone */}
              <div className="w-1/2 px-4 text-left md:text-right"> {/* Allineato a destra per i reverse */}
                <p className="text-xl font-extrabold text-blue-300">{milestone.date}</p> {/* Font più grande e audace */}
              </div>
            </div>
          ))}
        </section>

      </div>
    </main>
  );
}