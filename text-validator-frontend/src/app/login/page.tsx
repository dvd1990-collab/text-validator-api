// src/app/login/page.tsx
"use client";

import { SignIn } from "@clerk/nextjs";

export default function LoginPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-900 p-8 text-white">
      
      {/* 1. Modulo di Login: Applichiamo le classi per il centraggio al div contenitore */}
      {/* AGGIUNTO: flex flex-col items-center */}
      <div className="w-full max-w-md rounded-lg bg-gray-800 p-8 shadow-lg border border-gray-700 mb-16 flex flex-col items-center">
        <h1 className="text-3xl font-bold text-center text-blue-400 mb-6">
          Accedi a Text Validator
        </h1>
        {/* Il componente SignIn di Clerk */}
        <SignIn
          signUpUrl="/sign-up"
          afterSignInUrl="/"
          afterSignUpUrl="/"
		  routing="hash"
        />
      </div>

      {/* Nuove sezioni informative */}
      <div className="container mx-auto px-4">
        
        {/* I. Hero Section (Titolo Principale e Sottotitolo - SENZA CTA) */}
        <section className="text-center mb-16">
          <h1 className="text-5xl md:text-6xl font-extrabold text-blue-300 mb-4 leading-tight">
            Smetti di sembrare un robot. Ottieni testi AI con Qualità Umana, Certificata.
          </h1>
          <p className="text-xl text-gray-400 mb-8 max-w-3xl mx-auto">
            L'unico strumento che pulisce, normalizza il tono e certifica l'affidabilità del tuo contenuto con il nostro Punteggio di Qualità Umana™.
          </p>
        </section>

        {/* II. Sezione Workflow (Come Funziona) */}
        <section className="mb-16 py-16">
          <h2 className="text-4xl font-bold text-center text-blue-400 mb-12">Come Funziona il Doppio Controllo AI</h2>
          <div className="flex flex-col md:flex-row items-center justify-around space-y-8 md:space-y-0 md:space-x-8">
            {/* Workflow Image Placeholder */}
            <div className="md:w-1/2">
              <img 
                src="/placeholder-workflow-image.png"
                alt="Workflow in 3 passaggi: Pulizia, Normalizzazione, Certificazione Qualità Umana" 
                className="rounded-lg shadow-xl w-full h-auto object-cover"
              />
              <p className="text-sm text-gray-500 mt-2 text-center">
                (Illustrazione del processo: Input caotico &gt; Igiene Prompt &gt; Normalizzazione Tono &gt; Punteggio di Qualità Umana™)
              </p>
            </div>
            {/* Workflow Steps */}
            <div className="md:w-1/2 space-y-8 text-left">
              <div className="flex items-start space-x-4">
                <span className="text-blue-400 text-3xl font-bold">1.</span>
                <div>
                  <h3 className="text-2xl font-semibold text-gray-200 mb-2">Incolla e Puliamo il Caos</h3>
                  <p className="text-gray-400">Il tuo input (che sia una bozza da ChatGPT, un JSON o un testo con Markdown "sporco") viene immediatamente isolato. Il nostro motore AI esegue un'igiene del prompt avanzata, eliminando la formattazione inutile e normalizzando lo stile di scrittura.</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <span className="text-blue-400 text-3xl font-bold">2.</span>
                <div>
                  <h3 className="text-2xl font-semibold text-gray-200 mb-2">Normalizzazione del Tono Aziendale</h3>
                  <p className="text-gray-400">La nostra AI riscrittura, ottimizzata per il B2B italiano, trasforma il testo "AI-ish" in una voce professionale, coerente e on-brand. Dici addio alla freddezza robotica e ottieni coerenza su larga scala.</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <span className="text-blue-400 text-3xl font-bold">3.</span>
                <div>
                  <h3 className="text-2xl font-semibold text-gray-200 mb-2">Certificazione: Punteggio di Qualità Umana™</h3>
                  <p className="text-gray-400">Il nostro vantaggio competitivo: un secondo Agente AI ("LLM-as-a-Judge") valuta la coerenza, la persuasione e la conformità del testo finale, assegnando un punteggio e un report di qualità dettagliato. È il tuo senior editor virtuale.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* III. Sezione Funzionalità (Value-Based Focus) */}
        <section className="mb-16 py-16">
          <h2 className="text-4xl font-bold text-center text-blue-400 mb-12">Le Funzionalità che Trasformano il Tuo Business</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            {/* Blocco 1: Affidabilità e Conformità */}
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

            {/* Blocco 2: Efficienza e Persuasione */}
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

        {/* IV. Sezione Call to Action (CTA) Finale */}
        <section className="text-center bg-blue-900/20 p-12 rounded-xl shadow-xl border border-blue-700 mb-16 py-16">
          <h2 className="text-4xl font-bold text-blue-300 mb-4">Il Tuo Tempo e la Tua Credibilità Valgono Più di 50€ al Mese.</h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">Il valore della tranquillità è inestimabile. Scegli il piano che ti dà la certezza che ogni testo è perfetto.</p>
          <button 
            onClick={() => { /* Per ora non fa nulla. Puoi puntare a /sign-up o a un'altra sezione. */ }} 
            className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-bold py-4 px-12 rounded-full text-xl shadow-lg transform hover:scale-105 transition-all duration-300"
          >
            Inizia Subito il tuo Piano Pro (Pricing Trasparente)
          </button>
        </section>

        {/* V. Sezione Trust & Testimonianze (Social Proof) */}
        <section className="py-16">
          <h2 className="text-4xl font-bold text-center text-blue-400 mb-12">La Scelta dei Professionisti che non si accontentano.</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 text-center">
              <h3 className="text-2xl font-semibold text-green-400 mb-3">Affidabilità Serverless</h3>
              <p className="text-gray-300">Il nostro stack su Google Cloud Run garantisce velocità e scalabilità istantanea. Ottieni risposte in meno di 1 secondo.</p>
            </div>
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 text-center">
              <h3 className="text-2xl font-semibold text-green-400 mb-3">Margine &gt;94%</h3>
              <p className="text-gray-300">Il nostro modello di costo è ultra-efficiente e ti assicura che non paghi per l'inefficienza, ma solo per il valore generato.</p>
            </div>
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 text-center">
              <h3 className="text-2xl font-extrabold text-yellow-400 mb-3">Punteggio di Qualità Umana™</h3>
              <p className="text-gray-300 italic">"Certifichiamo l'eccellenza: ogni testo passa un rigoroso controllo AI, garantendo risultati indistinguibili da un'opera umana e un'affidabilità senza precedenti."</p>
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}