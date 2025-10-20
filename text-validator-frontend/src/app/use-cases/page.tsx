// src/app/use-cases/page.tsx
"use client";

import { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';

// --- TIPI DI DATO PER LA PAGINA ---
type UseCase = {
  module: 'strategist' | 'validator' | 'interpreter' | 'compliance';
  profile: string;
  title: string;
  context: string;
  input: string;
  output: string;
  qualityScore?: number;  // Opzionale
  reasoning?: string;     // Opzionale
};

type WorkflowStep = {
  module: 'strategist' | 'validator' | 'interpreter' | 'compliance';
  profile: string;
  action: string;
  output: string; // Aggiunto per mostrare l'output di ogni fase
};

type Workflow = {
  title: string;
  scenario: string;
  initialInput: string;
  steps: WorkflowStep[];
  presentationGuide: {
    visual: string;
    narrative: string;
  };
};

// --- CONTENUTO STRATEGICO DAI RISULTATI DELLA RICERCA ---

const strategistUseCases: UseCase[] = [
    {
        module: 'strategist',
        profile: "Sviluppatore di Business Model Canvas",
        title: "Da Idea Vaga a Business Model Canvas a Prova di Investitore",
        context: "Una startup early-stage ha sviluppato un'innovativa tecnologia di compressione dati per immagini mediche. Il team di ingegneri fatica ad articolarla in un modello di business coerente per le prime conversazioni con gli angel investor.",
        input: "La nostra idea è un'app che usa AI per comprimere le TAC e le risonanze magnetiche senza perdita di qualità, anzi, migliorandola. Gli ospedali risparmierebbero un sacco di soldi sullo storage. Potremmo venderla in abbonamento, forse per singolo reparto o per tutto l'ospedale. I nostri clienti sono i direttori IT degli ospedali e forse anche i radiologi. Dobbiamo parlare con loro. I competitor sono le soluzioni di storage tradizionali, che sono lente e costose. Noi siamo più veloci e più economici. Forse potremmo fare una partnership con i produttori di macchinari per le TAC. Le risorse chiave sono i nostri algoritmi e il nostro team di data scientist. I costi sono principalmente per il cloud e gli stipendi. Come revenue, pensiamo a un modello SaaS a tier, tipo 10k/anno per un piccolo ospedale, 50k per uno grande.",
        output: ""
    },
    {
        module: 'strategist',
        profile: "Pianificatore di Strategia Go-to-Market (GTM)",
        title: "Da 'Vogliamo Lanciare' a un Piano Go-to-Market Azionabile",
        context: "Un Product Manager di una PMI software deve lanciare una nuova funzionalità di reporting avanzato. Ha bisogno di un piano GTM completo che copra posizionamento, canali, pricing e metriche, ma non ha accesso a un team di marketing strategico dedicato.",
        input: "Oggetto: Lancio nuova feature \"Analytics Dashboard\"\n\nCiao team,\ndobbiamo lanciare la nuova dashboard di analytics per il Q3.\nLa feature permette ai project manager di visualizzare i dati di performance dei progetti in tempo reale.\nIl target sono i nostri utenti power, principalmente team leader e manager in aziende tech tra 50 e 200 dipendenti.\nL'obiettivo è aumentare l'upselling al nostro piano \"Enterprise\".\nDobbiamo comunicare che questa feature li aiuta a prendere decisioni migliori e più veloci.\nCanali: pensavo al nostro blog, una newsletter e forse qualche post su LinkedIn.\nData di lancio desiderata: 15 Settembre.\nMi serve una bozza di piano GTM da presentare alla direzione la prossima settimana.\nGrazie!",
        output: ""
    },
    {
        module: 'strategist',
        profile: "Strutturatore di Pitch per Investitori (Outline)",
        title: "Da Dati Sparsi a una Storyline Vincente per il Tuo Pitch Deck",
        context: "La founder di una piattaforma EdTech deve preparare un pitch deck per un round di finanziamento seed. Ha tutti i dati chiave (metriche, proiezioni), ma fatica a organizzarli in una narrazione convincente richiesta dagli investitori.",
        input: "Dati per il pitch:\nProblema: gli studenti universitari non hanno un modo efficace per collaborare su progetti di gruppo a distanza. Usano un mix di Google Docs, WhatsApp, Trello... è un caos.\nSoluzione: la nostra piattaforma \"CollabStudy\" integra video chat, gestione task, e un editor di documenti condiviso in un unico posto.\nMetriche: siamo cresciuti a 5,000 MAU (utenti attivi mensili) in 6 mesi, crescita del 20% mese su mese. Il 15% degli utenti si converte al piano premium da 5$/mese.\nMercato: il mercato globale dell'EdTech vale oltre 200 miliardi. Il nostro TAM (Total Addressable Market) è di circa 5 miliardi, focalizzandoci sugli studenti universitari in EU e Nord America.\nTeam: Io (CEO) ho un background in product management in una nota azienda EdTech. Il mio co-founder (CTO) è un ex ingegnere di Google.\nChiediamo 500k per 18 mesi di runway, per marketing e assunzioni nel team di sviluppo.",
        output: ""
    }
];

const validatorUseCases: UseCase[] = [
    {
        module: 'validator',
        profile: "Scrittore di Proposte Commerciali",
        title: "Da Appunti su un Post-it a Proposta Commerciale che Chiude i Contratti",
        context: "Un sales account deve inviare rapidamente una proposta a un cliente di alto valore. Ha preso appunti veloci ma non ha il tempo di redigere un documento formale, strutturato e persuasivo.",
        input: "Cliente: Acme Corp, contatto: John Doe.\nProblema: il loro team di customer support è sovraccarico. Tempo medio di risposta > 24h. CSAT basso.\nObiettivo: ridurre il tempo di risposta sotto le 8 ore e migliorare il CSAT del 20% in 6 mesi.\nSoluzione proposta: il nostro software \"SupportFlow\".\nImplementazione del nostro chatbot AI per gestire il 40% delle richieste di primo livello.\nSistema di ticketing intelligente per prioritizzare le richieste urgenti.\nDashboard di analytics per monitorare le performance.\nPricing: piano Enterprise, 2.500$/mese.\nOnboarding incluso, 2 sessioni di training.\nSupporto dedicato via email e telefono.\nProssimi passi: inviare proposta entro domani e fissare call di follow-up.",
        output: ""
    },
    {
        module: 'validator',
        profile: "L'Umanizzatore",
        title: "Umanizza l'AI: Da Bozza Robotica a Blog Post che Connette con i Lettori",
        context: "Un content manager ha una bozza di articolo generata da un'AI generica. Il testo è corretto ma manca di personalità, fluidità e non riflette il tono di voce aziendale.",
        input: "Il lavoro ibrido è un modello di lavoro che combina il lavoro in ufficio con il lavoro da remoto. Questo modello offre numerosi vantaggi sia per i datori di lavoro che per i dipendenti. Per i datori di lavoro, i vantaggi includono una riduzione dei costi immobiliari e l'accesso a un bacino di talenti più ampio. Per i dipendenti, i vantaggi includono una maggiore flessibilità e un migliore equilibrio tra vita professionale e vita privata. Tuttavia, il lavoro ibrido presenta anche delle sfide. Le sfide includono la garanzia di una comunicazione efficace e il mantenimento di una forte cultura aziendale. Le aziende devono implementare le giuste tecnologie e policy per avere successo con il modello di lavoro ibrido.",
        output: ""
    },
    {
        module: 'validator',
        profile: "Generico",
        title: "Coerenza Totale: Allinea Ogni Testo al Tuo DNA Aziendale",
        context: "Un'azienda ha diversi team che comunicano all'esterno in modo frammentato. L'obiettivo è garantire che ogni comunicazione su un nuovo aggiornamento sia immediatamente riconoscibile.",
        input: "Testo 1 (Marketing):\"Annuncio Esplosivo! Rivoluziona il tuo workflow con il nostro nuovo, potentissimo aggiornamento Project Atlas! Clicca qui per scoprire una produttività MAI VISTA PRIMA!\"\nTesto 2 (Sales, in una email a un prospect):\"La informo che abbiamo rilasciato un aggiornamento incrementale alla piattaforma, denominato Project Atlas, che include nuove funzionalità di reporting. Sarei lieto di illustrargliele.\"\nTesto 3 (Customer Support, in risposta a un ticket):\"Sì, l'aggiornamento Project Atlas è live. Ora puoi trovare i nuovi report nella sezione Analytics. Se hai altri problemi fammi sapere.\"",
        output: ""
    }
];

const interpreterUseCases: UseCase[] = [
    {
        module: 'interpreter',
        profile: "Analista Contratto di Vendita",
        title: "Decodifica il Legalese: Estrai Obblighi e Scadenze da Contratti Complessi",
        context: "Un manager operativo riceve un contratto di fornitura di 50 pagine. Deve capire rapidamente gli obblighi chiave, SLA, clausole di recesso e penali, senza attendere giorni per una revisione legale formale.",
        input: "...ai sensi della clausola 7.1, il Fornitore si impegna a garantire un Uptime del Servizio non inferiore al 99.8% su base mensile, calcolato come specificato nell'Allegato B. Qualsiasi inadempienza a tale obbligo, che non sia sanata entro 5 (cinque) giorni lavorativi dalla notifica scritta da parte del Cliente, darà luogo all'applicazione di una penale pari al 5% del canone mensile per ogni punto percentuale di Uptime inferiore alla soglia pattuita, come meglio dettagliato nella sezione 11.3. Il presente accordo avrà durata di 24 (ventiquattro) mesi dalla Data di Efficacia e si rinnoverà tacitamente per ulteriori periodi di 12 (dodici) mesi, salvo disdetta da inviarsi a mezzo posta raccomandata con avviso di ricevimento da una delle Parti con un preavviso non inferiore a 90 (novanta) giorni dalla scadenza. I pagamenti dovranno essere effettuati entro 30 giorni data fattura fine mese.",
        output: ""
    },
    {
        module: 'interpreter',
        profile: "Analista Bilancio Aziendale",
        title: "Dal Bilancio al Board: Sintetizza Report Finanziari Complessi",
        context: "Un CEO deve presentare i risultati del Q2 al board. Ha un report dettagliato ma necessita di una sintesi esecutiva che evidenzi performance, variazioni e aree di attenzione.",
        input: "Nel secondo trimestre, i ricavi totali hanno raggiunto 1.250.000 €, un aumento del 15% rispetto al Q2 dell'anno precedente (1.087.000 €) e del 5% sopra il budget previsto (1.190.000 €). La crescita è stata trainata principalmente dal segmento Enterprise (+25% YoY), mentre il segmento SMB ha registrato una crescita più modesta del 5% YoY. Il Costo del Venduto (COGS) è aumentato a 300.000 €, portando il Margine Lordo al 76%, in linea con le aspettative. Le Spese Operative (OpEx) sono state di 850.000 €, di cui 450.000 € per Ricerca e Sviluppo (in aumento a causa di nuove assunzioni) e 400.000 € per Vendite e Marketing. L'Utile Operativo si attesta a 100.000 €, rispetto ai 50.000 € del Q2 precedente. La liquidità a fine periodo è di 2.500.000 €.",
        output: ""
    },
    {
        module: 'interpreter',
        profile: "Analizzatore di Feedback dei Clienti",
        title: "La Voce del Cliente, Decodificata: Trasforma il Feedback in Priorità",
        context: "Un Product Manager ha centinaia di risposte da un sondaggio. Deve analizzare i commenti per identificare temi ricorrenti, sentiment e richieste per informare la roadmap di prodotto.",
        input: "Adoro il vostro tool ma l'interfaccia è un po' datata, specialmente sulla dashboard principale.\nPERCHÉ NON C'È ANCORA L'INTEGRAZIONE CON SLACK??? Tutti i vostri competitor ce l'hanno.\nIl customer support è fantastico, Sara mi ha risolto un problema in 5 minuti. 10/10.\nSarebbe utile poter esportare i report in formato PDF, non solo CSV.\nL'app mobile è lenta e a volte va in crash.\nIl prezzo è un po' alto, ma la funzionalità X mi fa risparmiare così tanto tempo che ne vale la pena.\nNon riesco a trovare la funzione per invitare nuovi membri del team. È nascosta?\nL'integrazione con Slack sarebbe un punto di svolta per noi.",
        output: ""
    }
];

const complianceUseCases: UseCase[] = [
    {
        module: 'compliance',
        profile: "Analizzatore GDPR Marketing",
        title: "Marketing Sicuro: Verifica la Conformità GDPR delle Tue Email",
        context: "Un team marketing sta per lanciare una campagna DEM. Devono assicurarsi che il testo non contenga violazioni del GDPR, come consensi non chiari o trattamento dati non trasparente.",
        input: "Oggetto: Sconti fino al 70% solo per te, Mario!\n\nCiao Mario,\nAbbiamo notato che di recente hai guardato i nostri nuovi smartphone. Abbiamo preparato un'offerta che non puoi rifiutare!\nClicca qui per ottenere uno sconto esclusivo fino al 70% su tutta la categoria elettronica.\nPer personalizzare ulteriormente le tue offerte, condividi con noi la tua data di nascita e riceverai un regalo speciale per il tuo compleanno!\nNon vuoi più ricevere le nostre fantastiche offerte? Puoi disiscriverti, ma ti perderesti il meglio.\nIl team di TechDeals\n(In calce, in piccolo) Per info sulla privacy, leggi il nostro regolamento.",
        output: ""
    },
    {
        module: 'compliance',
        profile: "Verificatore Anti-Bias Annunci Lavoro",
        title: "Assumi i Migliori, Senza Bias: Analizza i Tuoi Annunci di Lavoro",
        context: "Un recruiter vuole attrarre un pool di candidati diversificato. Deve revisionare un annuncio per una posizione Senior per eliminare linguaggio che possa scoraggiare candidati.",
        input: "Titolo: Sviluppatore Rockstar per il nostro Team di Ninja!\nStiamo cercando un giovane e dinamico Sviluppatore Software con almeno 10 anni di esperienza in architetture a microservizi. Il candidato ideale è un vero guru del codice, un guerriero che non si arrende di fronte a nessuna sfida e lavora bene sotto pressione. Sarà responsabile di guidare lo sviluppo del nostro prodotto di punta. Richiesta una forte competitività e la volontà di spaccare tutto. Se sei un ragazzo ambizioso e vuoi unirti a una squadra di soli numeri uno, questo è il posto per te. Offriamo birra gratis il venerdì e un ambiente da \"work hard, play hard\".",
        output: ""
    },
    {
        module: 'compliance',
        profile: "Validatore Claim Pubblicitari",
        title: "Promesse Credibili: Valida i Claim Pubblicitari per Evitare Contestazioni",
        context: "Per il lancio di un nuovo integratore, il team marketing ha preparato un testo con affermazioni forti. È necessario un pre-screening per verificare che i claim siano supportabili.",
        input: "Scopri \"Vitality Boost\", la rivoluzione per la tua energia! La nostra formula unica e brevettata è scientificamente provata per curare la stanchezza cronica e garantisce un aumento del 50% della tua produttività in una sola settimana. A differenza di tutti gli altri prodotti sul mercato, Vitality Boost è la soluzione più veloce e sicura per ritrovare il benessere. Migliaia di clienti soddisfatti confermano che i risultati sono immediati e permanenti. Non aspettare, trasforma la tua vita oggi! È l'unico integratore di cui avrai mai bisogno.",
        output: ""
    }
];

const allUseCases = {
    strategist: strategistUseCases,
    validator: validatorUseCases,
    interpreter: interpreterUseCases,
    compliance: complianceUseCases,
};

const integratedWorkflows: Workflow[] = [
    {
        title: "Workflow: Lancio di una Campagna Marketing a Prova di Rischio",
        scenario: "Una PMI SaaS sta lanciando una nuova funzionalità basata su AI. L'obiettivo è creare una campagna digitale completa (landing page, email, social) che sia persuasiva, allineata al brand e conforme alle normative GDPR e sulla pubblicità.",
        initialInput: "Brief per la campagna \"Project Insight\"\nProdotto: nuova feature di analisi predittiva nel nostro CRM. Usa AI per prevedere quali lead hanno più probabilità di chiudere.\nTarget: Direttori Vendite in aziende B2B tech.\nBenefici chiave: aumenta il tasso di conversione, fa risparmiare tempo ai sales, migliora l'accuratezza delle previsioni di vendita.\nOfferta di lancio: Prova gratuita di 30 giorni della feature per i primi 100 che si registrano.\nMessaggio principale: \"Smetti di tirare a indovinare. Il nostro AI ti dice su quali clienti puntare.\"\nCanali: Landing page, email alla nostra lista, 3 post per LinkedIn.\nDobbiamo essere aggressivi, dire che è la soluzione migliore sul mercato.\nObiettivo: 500 registrazioni alla prova gratuita nel primo mese.",
		imageSrc: "/WF1.png", 
        steps: [
            { module: 'strategist', profile: "Pianificatore di Strategia Go-to-Market (GTM)", action: "Il brief grezzo viene trasformato in una strategia di comunicazione multicanale completa, con messaggi chiave differenziati e angoli narrativi per ogni canale." },
            { module: 'validator', profile: "Scrittore Testi per Landing Page", action: "I messaggi chiave per la landing page vengono espansi in un copy di vendita fluente e persuasivo, allineato al Custom Tone of Voice aziendale." },
            { module: 'compliance', profile: "Validatore Claim Pubblicitari", action: "Il testo finale viene analizzato. Il claim 'la soluzione migliore sul mercato' viene segnalato e riformulato, e il form di contatto viene verificato per la conformità GDPR." }
        ],
        presentationGuide: {
            visual: "Una rappresentazione a 'swimlane' orizzontale. Un'icona (l'asset della campagna) si muove da sinistra a destra, trasformandosi a ogni fase: da 'appunti' a 'documento strutturato', poi 'testo brillante' e infine 'documento con scudo'.",
            narrative: "Enfatizzare il flusso continuo e la sicurezza: 'Idea -> Strategia -> Testo Perfetto -> Lancio Sicuro. Il tuo intero processo di marketing, orchestrato in un'unica piattaforma.'"
        }
    },
    {
        title: "Workflow: Dalla Due Diligence all'Offerta di Acquisizione",
        scenario: "Un team di corporate development sta valutando l'acquisizione di una startup. Devono analizzare rapidamente i documenti in data room (contratti, bilanci) per identificare rischi/opportunità e preparare una bozza di term sheet non vincolante.",
        initialInput: "(Dal contratto con il cliente più grande) \"Il Cliente avrà diritto a un'esclusiva sui Servizi nel settore FinTech per l'intera durata del presente Accordo, che si rinnoverà automaticamente per periodi di 24 mesi...\"\n(Dal bilancio) \"Anno 2022: Ricavi 1.5M €, Perdita Netta (300k €). Anno 2023: Ricavi 2.5M €, Utile Netto 300k €. La concentrazione dei ricavi è elevata, con il 60% del fatturato generato da un singolo cliente.\"\n(Dalla descrizione della tecnologia) \"La nostra piattaforma si basa su un algoritmo proprietario di machine learning (in attesa di brevetto)...\"",
		imageSrc: "/WF2.png", 
        steps: [
            { module: 'interpreter', profile: "Analista Contratto di Vendita", action: "I documenti vengono analizzati per estrarre automaticamente dati critici e 'red flag': clausole di rischio (esclusiva, concentrazione clienti), termini di rinnovo e trend finanziari." },
            { module: 'strategist', profile: "Strutturatore di Pitch per Investitori (Outline)", action: "Le 'red flag' e i punti di forza estratti vengono usati come input per generare un'analisi SWOT della target company e, successivamente, una bozza strutturata di term sheet non vincolante." },
            { module: 'validator', profile: "Generico", action: "La bozza del term sheet e la sintesi SWOT vengono raffinate per renderle chiare, concise e adatte a un pubblico di alto livello come il consiglio di amministrazione." }
        ],
        presentationGuide: {
            visual: "Un diagramma a imbuto. In alto, una pioggia di icone 'documento' entra nell'imbuto. L'imbuto si restringe passando per le tre fasi. Alla fine, esce un'unica icona a forma di 'diamante', a simboleggiare il valore estratto.",
            narrative: "Focus sulla velocità e la fiducia: 'Dalla Complessità alla Chiarezza Strategica. Analizza giorni di documenti in minuti, struttura la tua offerta e presentala con la massima fiducia.'"
        }
    },
    {
        title: "Workflow: Sviluppo e Implementazione di una Nuova Policy HR",
        scenario: "Il dipartimento HR deve creare e comunicare una nuova policy sul lavoro da remoto. Il processo deve essere efficiente, garantire che la policy sia equa e conforme, e la comunicazione ai dipendenti deve essere chiara e positiva.",
        initialInput: "Email dal CEO al Resp. HR:\nOggetto: Nuova policy lavoro da remoto\nCiao,\ndobbiamo formalizzare la nostra policy sul remote work, sta diventando un far west.\nPunti da includere:\n- Fino a 3 giorni a settimana da remoto.\n- 2 giorni obbligatori in ufficio per tutti per la collaborazione.\n- Budget una tantum di 500 € per la postazione a casa.\n- Per chi usa il proprio laptop (BYOD), serve sicurezza: antivirus aziendale e VPN obbligatori.\n- Chiarire aspettative su orari e reperibilità.\nPrepara una bozza di policy completa. Grazie.",
		imageSrc: "/WF3.png", 
        steps: [
            { module: 'strategist', profile: "Generatore di Policy Aziendali (Bozza Strategica)", action: "Il brief informale del CEO viene espanso in una bozza di policy completa e professionale, organizzata in sezioni logiche (Scopo, Eleggibilità, Sicurezza, etc.)." },
            { module: 'compliance', profile: "Verificatore Anti-Bias Annunci Lavoro", action: "La bozza viene analizzata per identificare potenziali aree di rischio, come implicazioni sulla privacy per il software su dispositivi personali (BYOD) o rischi di trattamenti non equi tra dipendenti." },
            { module: 'interpreter', profile: "Sintetizzatore di Meeting e Trascrizioni", action: "La policy finale e densa viene data in input per generare due output: una sintesi di una pagina per i manager e una lista di FAQ per tutti i dipendenti in linguaggio semplice." },
            { module: 'validator', profile: "L'Umanizzatore", action: "Utilizzando le FAQ e la sintesi, viene redatta l'email di annuncio ufficiale, applicando il Custom Tone of Voice aziendale ('trasparente e di supporto') per un messaggio positivo." }
        ],
        presentationGuide: {
            visual: "Un'infografica 'step-by-step' che mostra il ciclo di vita: Idea -> Documento con Scudo (Compliance) -> Documento scomposto in pezzi (Sintesi/FAQ) -> Megafono (Comunicazione).",
            narrative: "Sottolineare la completezza del supporto: 'Crea Policy Efficaci e Sicure. Dalla bozza alla comunicazione, un flusso integrato che protegge l'azienda e coinvolge i dipendenti.'"
        }
    },
];


const getModuleStyles = (module: 'strategist' | 'validator' | 'interpreter' | 'compliance') => {
    switch (module) {
        case 'strategist': return { accent: 'text-purple-400', border: 'border-purple-500', bg: 'bg-purple-600' };
        case 'validator': return { accent: 'text-blue-400', border: 'border-blue-500', bg: 'bg-blue-600' };
        case 'interpreter': return { accent: 'text-green-400', border: 'border-green-500', bg: 'bg-green-600' };
        case 'compliance': return { accent: 'text-yellow-400', border: 'border-yellow-500', bg: 'bg-yellow-600' };
        default: return { accent: 'text-gray-400', border: 'border-gray-500', bg: 'bg-gray-600' };
    }
};


export default function UseCasesPage() {
    const [view, setView] = useState<'useCases' | 'workflows'>('useCases');
    const [activeModule, setActiveModule] = useState<'strategist' | 'validator' | 'interpreter' | 'compliance'>('strategist');
    
    const [selectedUseCase, setSelectedUseCase] = useState<UseCase | null>(strategistUseCases[0]);
    const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);

    const handleViewChange = (newView: 'useCases' | 'workflows') => {
        setView(newView);
        setSelectedUseCase(null);
        setSelectedWorkflow(null);
        if (newView === 'useCases') {
            setSelectedUseCase(allUseCases[activeModule][0]);
        }
        if (newView === 'workflows') {
            setSelectedWorkflow(integratedWorkflows[0]);
        }
    };

    const handleModuleChange = (module: 'strategist' | 'validator' | 'interpreter' | 'compliance') => {
        setActiveModule(module);
        setSelectedUseCase(allUseCases[module][0]);
    };

    const currentUseCases = useMemo(() => allUseCases[activeModule], [activeModule]);

    return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
        <div className="w-full max-w-7xl">
            <header className="mb-12 text-center">
                <h1 className="text-5xl md:text-6xl font-extrabold text-blue-400 mb-4 leading-tight">
                    Casi Pratici e Flussi di Lavoro Integrati
                </h1>
                <p className="text-xl text-gray-400 max-w-4xl mx-auto">
                    Scopri come la nostra suite trasforma problemi complessi in soluzioni concrete, mostrando sia la potenza dei singoli moduli sia la magia della loro sinergia.
                </p>
            </header>

            {/* Main Toggler */}
            <div className="mb-12 flex justify-center p-2 bg-gray-800 rounded-full border border-gray-700 w-fit mx-auto">
                <button
                    onClick={() => handleViewChange('useCases')}
                    className={`px-8 py-3 text-xl font-semibold rounded-full transition-colors ${view === 'useCases' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
                >
                    Casi d'Uso per Modulo
                </button>
                <button
                    onClick={() => handleViewChange('workflows')}
                    className={`px-8 py-3 text-xl font-semibold rounded-full transition-colors ${view === 'workflows' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'}`}
                >
                    Workflow Integrati
                </button>
            </div>
            
            {/* --- SEZIONE CASI D'USO --- */}
            {view === 'useCases' && (
                <div className="animate-fade-in">
                    {/* Selettore Tab Moduli */}
                    <div className="mb-12 flex justify-center border-b border-gray-700">
                        {(Object.keys(allUseCases) as Array<keyof typeof allUseCases>).map(moduleKey => {
                            const styles = getModuleStyles(moduleKey);
                            return (
                                <button
                                    key={moduleKey}
                                    onClick={() => handleModuleChange(moduleKey)}
                                    className={`px-6 py-3 text-xl font-semibold capitalize transition-colors ${activeModule === moduleKey ? `border-b-2 ${styles.border} text-white` : 'text-gray-400 hover:text-white'}`}
                                >
                                    {moduleKey}
                                </button>
                            );
                        })}
                    </div>

                    {/* Selettore Profili */}
                    <section className="mb-12 py-8 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
                        <h2 className={`text-3xl font-bold text-center ${getModuleStyles(activeModule).accent} mb-6`}>Seleziona un Profilo per Vedere l'Esempio</h2>
                        <div className="flex flex-wrap justify-center gap-4 px-4">
                            {currentUseCases.map((uc) => {
                                const styles = getModuleStyles(uc.module);
                                return (
                                    <button
                                        key={uc.profile}
                                        onClick={() => setSelectedUseCase(uc)}
                                        className={`px-6 py-3 rounded-full text-lg font-semibold transition-all shadow-md ${selectedUseCase?.profile === uc.profile ? `${styles.bg} text-white transform scale-105` : 'bg-gray-700 text-gray-200 hover:bg-gray-600'}`}
                                    >
                                        {uc.profile}
                                    </button>
                                );
                            })}
                        </div>
                    </section>

                    {/* Dettaglio Use Case */}
                    {selectedUseCase && (
                        <section className="animate-fade-in">
                            <h2 className={`text-4xl font-bold ${getModuleStyles(selectedUseCase.module).accent} mb-6 text-center`}>
                                {selectedUseCase.title}
                            </h2>
                            <div className="bg-gray-800 p-16 rounded-xl shadow-lg border border-gray-700">
                                <h3 className={`text-2xl font-semibold ${getModuleStyles(selectedUseCase.module).accent} mb-4`}>La Sfida (Challenge):</h3>
                                <p className="text-gray-300 mb-8 italic">{selectedUseCase.context}</p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                    <div>
                                        <h3 className="text-2xl font-semibold text-white mb-3">Input Grezzo (Prima)</h3>
                                        <textarea rows={15} readOnly value={selectedUseCase.input} className="w-full h-full bg-gray-900 text-gray-200 p-4 rounded-lg border border-gray-700 resize-none shadow-inner" />
                                    </div>
                                    <div>
                                        <h3 className="text-2xl font-semibold text-white mb-3">Output Elaborato (Dopo)</h3>
                                        <div className="w-full h-full rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-inner overflow-y-auto prose prose-invert prose-p:text-gray-300">
                                            <ReactMarkdown>{selectedUseCase.output || "L'output generato dalla nostra AI apparirà qui una volta elaborato l'input..."}</ReactMarkdown>
                                        </div>
                                    </div>
                                </div>

                                {/* BOX QUALITY SCORE CORRETTO */}
                                {(selectedUseCase.module === 'validator' || selectedUseCase.module === 'interpreter') && (
                                    <div className="mt-16 p-6 rounded-xl bg-gray-900 border border-gray-700 shadow-md">
                                        <h3 className="text-2xl font-semibold text-green-400 mb-4 text-center">Controllo di Qualità AI²™</h3>
                                        <div className="flex items-center justify-center text-center mb-4">
                                            <p className="text-6xl font-bold text-green-400">{selectedUseCase.qualityScore || '...'}</p>
                                            <p className="ml-2 text-2xl text-gray-400">/ 100</p>
                                        </div>
                                        <p className="text-gray-300 italic text-center">
                                            {selectedUseCase.reasoning || "L'analisi qualitativa del nostro secondo agente AI apparirà qui..."}
                                        </p>
                                    </div>
                                )}
                            </div>
                        </section>
                    )}
                </div>
            )}

            {/* --- SEZIONE WORKFLOWS CON NUOVO LAYOUT --- */}
            {view === 'workflows' && (
                 <div className="animate-fade-in">
                    {/* Selettore Workflow */}
                     <section className="mb-12 py-8 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
                         <h2 className="text-3xl font-bold text-center text-purple-400 mb-6">Seleziona un Flusso di Lavoro Integrato</h2>
                         <div className="flex flex-wrap justify-center gap-4 px-4">
                             {integratedWorkflows.map((wf) => (
                                 <button
                                     key={wf.title}
                                     onClick={() => setSelectedWorkflow(wf)}
                                     className={`px-6 py-3 rounded-full text-lg font-semibold transition-all shadow-md ${selectedWorkflow?.title === wf.title ? 'bg-purple-600 text-white transform scale-105' : 'bg-gray-700 text-gray-200 hover:bg-gray-600'}`}
                                 >
                                     {wf.title.replace("Workflow: ", "")}
                                 </button>
                             ))}
                         </div>
                     </section>

                    {/* Dettaglio Workflow */}
                    {selectedWorkflow && (
                        <section className="animate-fade-in">
                            <h2 className="text-4xl font-bold text-purple-400 mb-6 text-center">{selectedWorkflow.title}</h2>
                            <div className="bg-gray-800 p-14 rounded-xl shadow-lg border border-gray-700">
                                <p className="text-xl text-gray-300 mb-10 italic text-center max-w-4xl mx-auto">{selectedWorkflow.scenario}</p>

                                {/* SEZIONE VISIVA CON IMMAGINE E NARRATIVA */}
                                <div className="flex flex-col md:flex-row items-center gap-8 mb-12 bg-gray-900/50 p-6 rounded-lg border border-gray-700">
                                    <div className="md:w-2/7">
                                        <img 
                                            src={selectedWorkflow.imageSrc} 
                                            alt={`Visualizzazione del workflow: ${selectedWorkflow.presentationGuide.visual}`}
                                            className="rounded-lg shadow-2xl w-full h-auto object-cover"
                                        />
                                    </div>
                                    <div className="md:w-3/5 flex flex-col justify-center">
                                        <p className="text-2xl italic text-gray-300 font-semibold text-center md:text-left">
                                            "{selectedWorkflow.presentationGuide.narrative}"
                                        </p>
                                    </div>
                                </div>
                                
                                <h3 className="text-3xl font-bold text-center text-white mb-8">Percorso del Workflow Dettagliato</h3>
                                <div className="space-y-12">
                                    {selectedWorkflow.steps.map((step, index) => {
                                        const styles = getModuleStyles(step.module);
                                        const isFirstStep = index === 0;
                                        const previousStepOutput = isFirstStep ? selectedWorkflow.initialInput : selectedWorkflow.steps[index - 1].output;

                                        return (
                                            <div key={index} className="border-l-4 pl-8 md:pl-12 relative" style={{ borderColor: styles.border.replace('border-', '') }}>
                                                <div className={`absolute top-0 -left-6 w-12 h-12 rounded-full ${styles.bg} flex items-center justify-center text-white font-bold text-2xl shadow-lg`}>{index + 1}</div>
                                                <div className="mb-6">
                                                    <p className={`text-2xl font-bold capitalize ${styles.accent}`}>{step.module} &rarr; {step.profile}</p>
                                                    <p className="text-gray-400 mt-1">{step.action}</p>
                                                </div>
                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                                    <div>
                                                        <h4 className="text-lg font-semibold text-gray-300 mb-2">{isFirstStep ? "Input Iniziale" : "Input (dallo Step Precedente)"}</h4>
                                                        <textarea rows={10} readOnly value={previousStepOutput} className="w-full h-full bg-gray-900 text-gray-200 p-4 rounded-lg border border-gray-700 resize-none shadow-inner" />
                                                    </div>
                                                    <div>
                                                        <h4 className="text-lg font-semibold text-gray-300 mb-2">Output Generato in questa fase</h4>
                                                        <div className="w-full h-full rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-inner overflow-y-auto prose prose-invert prose-p:text-gray-300">
                                                            <ReactMarkdown>{step.output || "L'output di questa fase apparirà qui..."}</ReactMarkdown>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        </section>
                    )}
                 </div>
            )}

        </div>
    </main>
);
}