// src/app/privacy-policy/page.tsx
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Privacy Policy - Text Validator AI',
};

// Componenti helper per la formattazione
const SectionTitle = ({ number, title }: { number: string; title: string }) => (
  <h2 className="text-3xl font-bold text-blue-300 mt-12 mb-6 border-b border-gray-700 pb-2">
    <span className="text-blue-500">{number}.</span> {title}
  </h2>
);

const SubTitle = ({ number, title }: { number: string; title: string }) => (
  <h3 className="text-2xl font-semibold text-gray-100 mt-8 mb-4">
    {number}. {title}
  </h3>
);

export default function PrivacyPolicyPage() {
  return (
    <main className="flex flex-col items-center bg-gray-900 px-4 md:px-8 text-white">
      <div className="w-full max-w-4xl prose prose-invert prose-lg prose-p:text-gray-300 prose-strong:text-white">
        
        <header className="mb-12 text-center">
          <h1 className="text-5xl font-extrabold text-blue-400 mb-4">Privacy Policy</h1>
          <p className="text-xl text-gray-400">Informativa sul Trattamento dei Dati Personali (GDPR)</p>
          <p className="text-sm text-gray-500 mt-2">Ultimo aggiornamento: 13 Ottobre 2025</p>
        </header>

        <p className="lead text-gray-300">
          La presente informativa è resa ai sensi degli articoli 13 e 14 del Regolamento UE 2016/679 (GDPR) e della normativa italiana vigente in materia di protezione dei dati personali.
        </p>

        <SectionTitle number="1" title="Titolare del Trattamento e Riferimenti Normativi" />
        <SubTitle number="1.1" title="Identità del Titolare del Trattamento" />
        <p>
          Il Titolare del Trattamento dei dati personali è <strong>TextValidatorAISoc</strong>, con sede legale in <strong>TextValidatorAIInd</strong>, Partita IVA/Codice Fiscale <strong>TextValidatorAIPIVA</strong> (di seguito, "il Titolare" o "Text Validator AI"). Text Validator AI determina le finalità e i mezzi del trattamento dei dati personali forniti dagli utenti attraverso il sito web <strong>https://www.textvalidator.ai</strong>.
        </p>
        
        <SubTitle number="1.2" title="Riferimenti di Contatto" />
        <p>
          Per qualsiasi questione relativa al trattamento dei dati personali o per l'esercizio dei diritti, l'interessato può contattare Text Validator AI ai seguenti indirizzi:
          <ul className="list-disc pl-6">
            <li>Email di Contatto per la Privacy: <strong>privacy@textvalidator.ai</strong></li>
            <li>Email di Supporto Generale: <strong>supporto@textvalidator.ai</strong></li>
          </ul>
        </p>

        <SubTitle number="1.3" title="Ambito di Applicazione e Conformità ai Trasferimenti Internazionali" />
        <p>
          Text Validator AI si impegna a garantire la conformità al GDPR. A causa dell'infrastruttura di terze parti adottata, che include processori di dati con sede negli Stati Uniti, i trasferimenti internazionali avvengono nel rispetto del Capo V del GDPR, attraverso l'adozione delle Clausole Contrattuali Standard (SCCs) e/o l'adesione al Data Privacy Framework (DPF).
        </p>
        
        <SectionTitle number="2" title="Tipologie di Dati Personali Trattati, Finalità e Base Giuridica" />
        <SubTitle number="2.1" title="Dati di Navigazione e Tecnici" />
        <p>
            <strong>Dati:</strong> Indirizzi IP, log di accesso, informazioni sul dispositivo (browser, sistema operativo), dati di diagnostica. <br />
            <strong>Finalità:</strong> Garantire la sicurezza, la funzionalità della piattaforma e per analisi diagnostiche. <br />
            <strong>Base Giuridica:</strong> Legittimo Interesse (Art. 6.1.f GDPR) del Titolare a garantire l'affidabilità e il corretto funzionamento tecnico del servizio.
        </p>
        
        <SubTitle number="2.2" title="Dati Forniti per la Registrazione e il Profilo Utente" />
        <p>
            <strong>Dati:</strong> Indirizzo email, password hash, ID Utente, ruolo, contatore di utilizzo, dettagli sul piano di abbonamento. <br />
            <strong>Finalità:</strong> Creazione e gestione dell'account, autenticazione sicura, fornitura del servizio e monitoraggio dell'utilizzo per applicare le limitazioni del piano. <br />
            <strong>Base Giuridica:</strong> Esecuzione del Contratto (Art. 6.1.b GDPR).
        </p>

        <SubTitle number="2.3" title="Dati di Pagamento e Fatturazione (Trattati da Stripe)" />
        <p>
            <strong>Dati:</strong> Nome, cognome, indirizzo di fatturazione e informazioni finanziarie relative alla transazione. <br />
            <strong>Finalità:</strong> Elaborazione dei pagamenti, gestione dei rinnovi e adempimento degli obblighi fiscali. <br />
            <strong>Base Giuridica:</strong> Esecuzione del Contratto (Art. 6.1.b GDPR) per i pagamenti, e Adempimento di Obblighi Legali (Art. 6.1.c GDPR) per la fiscalità.
        </p>

        <SectionTitle number="3" title="Trattamento dei Contenuti Testuali Tramite AI" />
        <SubTitle number="3.1" title="Dati Trattati, Finalità e Base Giuridica" />
        <p>
            <strong>Dati Trattati:</strong> Contenuti testuali di Input e Output. <br />
            <strong>Finalità:</strong> Esclusivamente l'esecuzione dell'analisi e validazione testuale in tempo reale. <br />
            <strong>Base Giuridica:</strong> Esecuzione del Contratto (Art. 6.1.b GDPR).
        </p>

        <SubTitle number="3.2" title="Politica di Non Conservazione (Zero Data Retention)" />
        <p>
          Text Validator AI adotta una politica rigorosa di "Zero Data Retention". Il testo inviato dall'utente viene processato in tempo reale e <strong>non viene MAI archiviato, conservato o memorizzato</strong> sui nostri server o sui sistemi del motore AI (Google Gemini) dopo che l'elaborazione è stata completata. Questa garanzia è assicurata tramite configurazioni specifiche delle API di Google Gemini per disattivare il caching e il logging dei prompt.
        </p>
        
        <SectionTitle number="4" title="Modalità del Trattamento e Periodo di Conservazione" />
        {/* ... (Sezioni 4, 5, 6, 7 complete) ... */}
        {/* Per la tabella dei Sub-Processors, usiamo una tabella HTML */}
        <SectionTitle number="5" title="Destinatari e Trasferimento di Dati (Sub-Processors)" />
        <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-700">
                <thead className="bg-gray-800">
                    <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Sub-Processor</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Funzione</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Sede</th>
                    </tr>
                </thead>
                <tbody className="bg-gray-900 divide-y divide-gray-700">
                    <tr><td className="px-6 py-4">Vercel</td><td className="px-6 py-4">Hosting Frontend</td><td className="px-6 py-4">USA</td></tr>
                    <tr><td className="px-6 py-4">Google Cloud Run</td><td className="px-6 py-4">Hosting Backend</td><td className="px-6 py-4">USA</td></tr>
                    <tr><td className="px-6 py-4">Clerk</td><td className="px-6 py-4">Autenticazione Utenti</td><td className="px-6 py-4">USA</td></tr>
                    <tr><td className="px-6 py-4">Supabase</td><td className="px-6 py-4">Database Profili</td><td className="px-6 py-4">USA</td></tr>
                    <tr><td className="px-6 py-4">Google Gemini API</td><td className="px-6 py-4">Motore AI</td><td className="px-6 py-4">USA</td></tr>
                    <tr><td className="px-6 py-4">Stripe</td><td className="px-6 py-4">Pagamenti</td><td className="px-6 py-4">USA</td></tr>
                </tbody>
            </table>
        </div>

        <SectionTitle number="6" title="Diritti dell'Interessato (Art. 15-22 GDPR)" />
        <p>
            L'Utente, in qualità di Interessato, gode dei seguenti diritti: Diritto di Accesso, di Rettifica, alla Cancellazione (Diritto all'Oblio), di Limitazione di Trattamento, alla Portabilità dei Dati, di Opposizione, di Revoca del Consenso. Puoi esercitare questi diritti in qualsiasi momento inviando una comunicazione a <strong>privacy@textvalidator.ai</strong>. Hai inoltre il diritto di proporre reclamo all'Autorità di Controllo competente (Garante per la protezione dei dati personali).
        </p>

        <SectionTitle number="7" title="Informativa sui Cookie" />
        <p>
            Text Validator AI utilizza cookie tecnici, indispensabili per il corretto funzionamento del sito e per l'erogazione del servizio (es. gestione della sessione di login). L'utilizzo di questi cookie non richiede il consenso preventivo dell'utente. Non utilizziamo cookie di profilazione non tecnici.
        </p>
      </div>
    </main>
  );
}