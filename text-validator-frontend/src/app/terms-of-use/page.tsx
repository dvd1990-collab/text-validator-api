// src/app/terms-of-use/page.tsx
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Terms of Use - Text Validator AI',
};

// Riutilizziamo il componente SectionTitle per coerenza
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

export default function TermsOfUsePage() {
  return (
    <main className="flex flex-col items-center bg-gray-900 px-4 md:px-8 text-white">
      <div className="w-full max-w-4xl prose prose-invert prose-lg prose-p:text-gray-300 prose-strong:text-white">

        <header className="mb-12 text-center">
          <h1 className="text-5xl font-extrabold text-blue-400 mb-4">Termini di Utilizzo</h1>
          <p className="text-xl text-gray-400">(Terms of Use)</p>
          <p className="text-sm text-gray-500 mt-2">Ultimo aggiornamento: 13 Ottobre 2025</p>
        </header>

        <p className="lead text-gray-300">
          I presenti Termini di Utilizzo ("Termini") regolano l'accesso e l'uso del servizio SaaS Text Validator AI, accessibile all'URL <strong>https://www.textvalidator.ai</strong> e fornito da <strong>TextValidatorAISoc</strong> ("il Titolare"). L'accettazione dei Termini è un requisito essenziale per l'utilizzo del servizio.
        </p>

        <SectionTitle number="1" title="Accettazione e Oggetto del Servizio" />
        <p>
            Il Titolare offre l'accesso al software come servizio (SaaS) denominato Text Validator AI (il "Servizio"), il cui scopo principale è l'analisi, la validazione e il miglioramento di contenuti testuali tramite motori di intelligenza artificiale. L'utilizzo del Servizio è subordinato all'accettazione integrale dei presenti Termini.
        </p>

        <SectionTitle number="2" title="Registrazione dell'Account e Sicurezza" />
        <p>
            Il Servizio è destinato a maggiorenni con capacità legale. L'Utente garantisce la veridicità delle informazioni fornite ed è l'unico responsabile della custodia delle proprie credenziali di accesso.
        </p>

        <SectionTitle number="3" title="Piani di Abbonamento e Termini Commerciali" />
        <SubTitle number="3.1" title="Tipologie di Piani" />
        <p>Il Servizio è offerto nei piani <strong>Free</strong>, <strong>Starter</strong> e <strong>Pro</strong>, ciascuno con limiti di utilizzo e caratteristiche specifiche. Il Titolare si riserva il diritto di modificare i limiti di ciascun piano previa comunicazione.</p>
        
        <SubTitle number="3.2" title="Termini di Pagamento, Rinnovo e Cancellazione" />
        <p>
            I pagamenti sono gestiti da Stripe. Salvo disdetta, gli abbonamenti si rinnovano automaticamente. L'Utente può cancellare in qualsiasi momento, con effetto alla fine del periodo già pagato. Non sono previsti rimborsi per periodi non utilizzati.
        </p>
        
        <SectionTitle number="4" title="Regole di Utilizzo Consentito (Acceptable Use Policy)" />
        <SubTitle number="4.1" title="Uso Proibito" />
        <p>È severamente vietato utilizzare il Servizio per scopi illegali, diffamatori, per violare diritti di terzi, o per tentare di eludere le misure di sicurezza del sistema.</p>
        
        <SubTitle number="4.2" title="Uso Responsabile dell'Intelligenza Artificiale" />
        <p>
            L'Utente accetta quanto segue:
            <ul className="list-disc pl-6">
                <li><strong>Esclusione di Usi ad Alto Rischio:</strong> Il Servizio non deve essere usato per applicazioni critiche (mediche, giudiziarie, finanziarie) che possono avere un impatto significativo sulla vita o sui diritti delle persone.</li>
                <li><strong>Obbligo di Verifica:</strong> L'output generato dall'AI può contenere errori ("allucinazioni"). L'Utente ha l'obbligo contrattuale di <strong>verificare e revisionare criticamente</strong> ogni risultato prima di farvi affidamento.</li>
            </ul>
        </p>

        <SectionTitle number="5" title="Proprietà Intellettuale" />
        <p>
            Il Titolare mantiene tutti i diritti di proprietà intellettuale sul Servizio. L'Utente mantiene la piena proprietà dei contenuti testuali immessi nella piattaforma.
        </p>

        <SectionTitle number="6" title="Esclusioni e Limitazione di Responsabilità" />
        <SubTitle number="6.1" title="Esclusione di Garanzia (Servizio 'As-Is')" />
        <p>
            Il Servizio è fornito "così com'è", senza garanzie di alcun tipo. Il Titolare non garantisce che il Servizio sarà ininterrotto, sicuro, o che i risultati dell'AI saranno sempre accurati e affidabili.
        </p>
        
        <SubTitle number="6.2" title="Limitazione di Responsabilità" />
        <p>
            La responsabilità complessiva massima del Titolare nei confronti dell'Utente sarà limitata all'importo totale pagato dall'Utente nei dodici (12) mesi precedenti l'evento che ha dato origine al danno.
        </p>

        <SectionTitle number="7" title="Legge Applicabile e Foro Competente" />
        <p>
            I presenti Termini sono regolati dalla Legge Italiana. Per qualsiasi controversia, la competenza esclusiva spetta al <strong>Foro di Padova</strong>.
        </p>
      </div>
    </main>
  );
}