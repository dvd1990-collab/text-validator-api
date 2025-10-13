// src/app/use-cases/page.tsx
"use client";

import { useState } from 'react';
import Link from 'next/link';

export default function UseCasesPage() {
  // Ho rimosso l'importazione di useRouter dato che non la usiamo direttamente in questo file.

  const useCasesData = [
    {
      profile: "Generico",
      title: "Pulizia di Bozze Interne e Normalizzazione del Tono",
      context: "Il team di Ricerca e Sviluppo utilizza un modello AI per riassumere rapidamente i progressi settimanali da condividere con la dirigenza. L'output grezzo è spesso tecnicamente corretto ma ha un tono eccessivamente formale e frasi lunghe che rallentano la lettura e la comprensione. Questo profilo serve a rendere il riassunto immediatamente fruibile e con un tono aziendale neutro e professionale.",
      input: "L'analisi inerente all'architettura modulare ha evidenziato una non-linearità significativa nella fase di implementazione del modulo API. L'ottimizzazione dei parametri algoritmici è stata completata; la performance è stata incrementata del 12.4% rispetto al benchmark stabilito, in conformità con le specifiche di progetto. Le prossime azioni implicano la migrazione dell'infrastruttura di testing.",
      output: "L'analisi dell'architettura modulare ha rilevato una non-linearità significativa durante la fase di implementazione del modulo API. L'ottimizzazione dei parametri algoritmici è conclusa; la performance è aumentata del 12,4% rispetto al benchmark stabilito, in piena conformità con le specifiche di progetto. Le prossime azioni pianificate includono la migrazione dell'infrastruttura di testing.",
      qualityScore: 93,
      reasoning: "Il testo revisionato è un miglioramento incrementale ma solido rispetto all'originale. Sono state effettuate sostituzioni mirate per affinare il tono B2B ('rilevato' al posto di 'evidenziato', 'aumentata' al posto di 'incrementata', 'in piena conformità' al posto di 'in conformità'). La formattazione del numero ('12,4%') è standardizzata. Il markup è pulito (nessun elemento HTML presente). La chiarezza è mantenuta e leggermente migliorata dalla scelta lessicale più precisa. Il punteggio è alto per la coerenza e il mantenimento degli standard B2B."
    },
    {
      profile: "PM - Interpretazione Trascrizioni",
      title: "Sintesi Azionabile da Trascrizione Riunione Tecnica",
      context: "Dopo una riunione strategica di 90 minuti, viene generata una trascrizione automatica verbosa e disorganizzata. Il Project Manager ha bisogno di estrarre rapidamente le decisioni chiave, i next steps e le risorse assegnate, eliminando pause, ripetizioni e digressioni non rilevanti, per produrre un verbale di riunione conciso e focalizzato.",
      input: "Quindi, uhm, sì, abbiamo concordato che, beh, aspetta, devo ricontrollare, ma credo che il *deliverable* A, quello che avevamo in discussione, dovrebbe essere re-prioritizzato. Mario ha detto che non è fattibile entro la fine del Q2. Quindi, la decisione è, forse, spostarlo. Dunque, il *task* per il team di sviluppo è: riallocare le risorse.",
      output: "**OGGETTO:** Riprogrammazione Deliverable A\n\n**DECISIONE:** Il Deliverable A deve essere ri-prioritizzato/spostato a causa di problemi di fattibilità entro la scadenza Q2.\n\n**ATTIVITÀ:** Riassegnazione delle risorse necessarie per la ri-prioritizzazione.\n\n**RESPONSABILE/AZIONE:** Team di Sviluppo - Rilocare le risorse.\n\n**DEADLINE IMPLICITA:** Fine Q2 (Scadenza originale rivista/non confermata per il Deliverable A).",
      qualityScore: 98,
      reasoning: "Il testo revisionato è eccellente sotto il profilo del Project Management. Ha trasformato un discorso vago e incerto ('uhm, sì, abbiamo concordato che, beh, aspetta...') in una comunicazione strutturata e orientata all'azione. La struttura utilizza intestazioni chiare (OGGETTO, DECISIONE, ATTIVITÀ) che facilitano la rapida comprensione dello stato del progetto. La DECISIONE identifica chiaramente il problema (fattibilità Q2) e la soluzione (ri-prioritizzazione). L'ATTIVITÀ e il RESPONSABILE definiscono chi deve fare cosa. L'unico punto migliorabile, sebbene non essenziale per la chiarezza immediata, è la DEADLINE IMPLICITA, che è corretta ma suggerisce che la nuova data non è ancora formalizzata, il che è un dato importante per il PM. Il tono è professionale e orientato all'azione."
    },
    {
      profile: "Copywriter Persuasivo",
      title: "Trasformazione di Specifiche in Vantaggi di Marketing",
      context: "Il team di prodotto ha fornito all'AI le specifiche tecniche di un nuovo software B2B. L'output è una descrizione accurata ma piatta e priva di qualunque elemento persuasivo o emozionale, non adatta a una landing page di marketing che deve convertire (manca l'uso di un linguaggio umano per connettersi con l'utente).",
      input: "Il prodotto X offre un'interfaccia utente configurabile con un *dashboard* analitico integrato. Il suo motore di elaborazione dati asincrono riduce il *latency* del 40%. La compatibilità con i sistemi legacy è garantita tramite il protocollo Z. Implementazione rapida, manutenzione semplificata.",
      output: "**Sblocca la Tua Efficienza Oggi Stesso!** Stanco di interfacce che ti rallentano? Con la nostra **Dashboard Analitica Completamente Personalizzabile**, avrai il controllo totale dei tuoi dati, esattamente come vuoi tu. Dimentica l'attesa: il nostro motore di elaborazione dati rivoluzionario abbatte la latenza del **40%**, regalandoti decisioni istantanee. Nessun incubo di integrazione: ci colleghiamo fluidamente ai tuoi sistemi esistenti grazie al protocollo Z. **Implementazione lampo e manutenzione che scompare**: inizia a dominare il mercato *adesso* e libera tempo prezioso. Sei pronto a vedere la differenza che l'efficienza reale fa sul tuo fatturato?",
      qualityScore: 95,
      reasoning: "Il testo revisionato è significativamente più forte dell'originale, trasformando le specifiche tecniche in benefici tangibili per l'utente. La **Chiarezza dei Benefici** è eccellente: \"controllo totale dei tuoi dati\", \"decisioni istantanee\" e \"manutenzione che scompare\" sono chiari e desiderabili. La **Forza Persuasiva** è potenziata dall'uso di domande retoriche (\"Stanco di interfacce che ti rallentano?\") e un linguaggio energico (\"rivoluzionario\", \"abbatte\", \"dominare il mercato\"). L'uso del grassetto indirizza efficacemente l'attenzione sui punti di forza principali. La **Call to Action** è sia esplicita (\"Sblocca la Tua Efficienza Oggi Stesso!\") che implicita, culminando in una domanda forte e orientata al risultato finale (\"Sei pronto a vedere la differenza che l'efficienza reale fa sul tuo fatturato?\"). Il copy è coinvolgente e orientato all'azione."
    },
    {
      profile: "Revisore Legale/Regolatorio",
      title: "Verifica di Chiarezza e Conformità nel Linguaggio Contrattuale",
      context: "Un avvocato aziendale usa un generatore di AI per produrre la bozza di una nuova clausola di Termini di Servizio. Il testo grezzo, sebbene legalmente valido, è ambiguo in punti critici a causa di una sintassi troppo contorta, aumentando il rischio di contenzioso. Serve un controllo che assicuri precisione lessicale e semplicità regolatoria.",
      input: "La parte sottoscritta si impegna a non divulgare, né direttamente né per interposta persona, informazioni proprietarie, che sono definite in senso lato come comprendenti qualsivoglia dato, informazione o know-how non di pubblico dominio, a meno che e fintanto che un simile trasferimento non sia previamente autorizzato per iscritto dal Licenziante.",
      output: "La Parte Sottoscritta si impegna a non divulgare, direttamente o tramite terzi, Informazioni Proprietarie. Ai fini del presente Accordo, le Informazioni Proprietarie sono definite come qualsiasi dato, informazione o know-how non accessibile al pubblico, salvo preventivo consenso scritto del Licenziante per ogni specifico trasferimento.",
      qualityScore: 92,
      reasoning: "Il testo revisionato migliora significativamente l'originale in termini di chiarezza e formalità, avvicinandosi a uno standard contrattuale più elevato. L'uso di \"tramite terzi\" è più incisivo di \"per interposta persona\". La definizione di \"Informazioni Proprietarie\" è resa più concisa (\"non accessibile al pubblico\" vs \"non di pubblico dominio\"). Tuttavia, l'espressione \"per ogni specifico trasferimento\" introduce una potenziale ambiguità rispetto all'originale che utilizzava \"a meno che e fintanto che un simile trasferimento non sia previamente autorizzato\", suggerendo una necessità di autorizzazione *preventiva* per *qualsiasi* trasferimento, non necessariamente specifica per *ogni singolo* atto. Mantenendo la formulazione legale standard (autorizzazione preventiva generale o specifica), si suggerisce una leggera modifica per mantenere la rigorosità: \"salvo preventivo consenso scritto del Licenziante per il trasferimento.\". La capitalizzazione di \"Informazioni Proprietarie\" e \"Parte Sottoscritta\" è appropriata per un documento legale formale. Nel complesso, la conformità formale e la precisione sono molto alte, necessitano solo di una piccola rifinitura sulla clausola di eccezione."
    },
    {
      profile: "Scrittore di Newsletter",
      title: "Ristrutturazione di Contenuto Long-Form per l'Email Engagement",
      context: "Un articolo di blog di 2000 parole è stato riassunto dall'AI per la newsletter settimanale. Il riassunto è troppo denso, non formattato per la rapida scansione tipica di una casella di posta, e non incoraggia il click-through. Il profilo deve segmentare il testo con bullet point, titoli intermedi e un tono adatto al mezzo.",
      input: "Nel panorama attuale della digitalizzazione B2B, l'adozione di soluzioni cloud-native sta ridefinendo i parametri di efficienza operativa. Le aziende che hanno implementato queste architetture hanno registrato miglioramenti sensibili nella scalabilità e nella resilienza dei sistemi. Tuttavia, la migrazione presenta delle sfide significative che devono essere attentamente valutate, in particolare riguardo la sicurezza dei dati e la formazione del personale preesistente. Questo richiede un piano strategico pluriennale.",
      output: "**Oggetto:** Sbloccare la Vera Efficienza B2B: La Sfida (e l'Opportunità) del Cloud-Native\n\nCiao [Nome Cliente/Azienda],\n\n**Siete ancora bloccati nelle architetture di ieri mentre i vostri competitor volano in Cloud?**\n\nIl panorama digitale B2B si sta ridisegnando velocemente. Non si tratta più solo di \"essere online\", ma di come l'infrastruttura supporta la crescita *aggressiva*. L'adozione delle architetture **cloud-native** non è più un optional; è il nuovo benchmark per l'efficienza operativa.\n\n### La Realtà dei Fatti: I Vantaggi Immediati\n\nLe aziende che hanno fatto il salto stanno vedendo risultati concreti:\n\n*   **Scalabilità Elastica:** Capacità di crescere (o contrarsi) in ore, non mesi, rispondendo in tempo reale alle fluttuazioni del mercato.\n*   **Resilienza Garantita:** Sistemi progettati per resistere a picchi di traffico e fallimenti parziali senza interrompere il business.\n\nMa sappiamo che la strada non è sempre in discesa.\n\n### Il Velo Sottile: Dove si Nascondono i Rischi\n\nLa migrazione non è una semplice \"copia e incolla\". Per trasformare i costi operativi in investimenti strategici, è cruciale affrontare due punti dolenti:\n\n1.  **Sicurezza & Compliance:** Spostare dati critici richiede una revisione radicale delle policy di sicurezza. Non basta estendere il perimetro; bisogna integrarlo nel DNA dell'architettura.\n2.  **Competenze Interne:** Il personale deve evolvere. Un piano di upskilling mirato è fondamentale per sfruttare appieno le nuove piattaforme e non ritrovarsi con strumenti potenti usati a metà regime.\n\n**La Soluzione? Una Roadmap Strategica (Non un Fix Veloce).**\n\nPer ottenere questi benefici senza cadere nelle trappole comuni, serve una strategia pluriennale che allinei la tecnologia alle vostre ambizioni di business.\n\n---\n\n**🚀 NEXT STEP: È il momento di pianificare la trasformazione, non solo di reagire.**\n\nSe state valutando la migrazione cloud-native o state cercando di ottimizzare la vostra architettura attuale, parliamone. Prenota una call di 20 minuti con i nostri architetti per definire le priorità e mappare il percorso più sicuro per la vostra azienda.\n\n[**Clicca Qui per Pianificare la Tua Consulenza Strategica Gratuita**]",
      qualityScore: 95,
      reasoning: "Il TESTO REVISIONATO è significativamente migliorato rispetto all'originale, trasformando un paragrafo informativo in una newsletter coinvolgente. Efficacia del Gancio Iniziale: Eccellente. La domanda in grassetto (\"Siete ancora bloccati nelle architetture di ieri mentre i vostri competitor volano in Cloud?\") è diretta, crea urgenza (FOMO) e stabilisce immediatamente il contesto B2B. Leggibilità: Ottima. L'uso di grassetti, elenchi puntati chiari (Vantaggi Immediati) e liste numerate (Rischi) spezza il muro di testo. Il tono è conversazionale ma mantiene un alto livello di professionalità, appropriato per un tema tecnico come il cloud-native. Chiarezza della Call to Action (CTA): Molto chiara. La sezione \"NEXT STEP\" è ben evidenziata, l'offerta (call di 20 minuti gratuita) è specifica e il bottone finale è orientato all'azione. Il punteggio riflette un'esecuzione di alta qualità per un'email di questo tipo."
    },
    {
      profile: "Social Media Manager B2B",
      title: "Adattamento di Dati Tecnici a Post di LinkedIn Coinvolgenti",
      context: "Il team vuole promuovere un nuovo studio di settore su LinkedIn. L'AI ha generato un riassunto accademico e pieno di gergo (come spesso fa il contenuto AI). Il Social Media Manager ha bisogno di trasformarlo in un post breve, con un hook forte, un tono accessibile e un invito all'azione che massimizzi la portata organica.",
      input: "Il nostro studio quantitativo ha rivelato una correlazione positiva statisticamente significativa (p < 0.01) tra l'ottimizzazione del *data silo* e la riduzione del *churn rate* del cliente. L'analisi regressiva ha confermato l'ipotesi preliminare, fornendo evidenze empiriche per l'efficacia del nostro *framework* proprietario di gestione del ciclo di vita del dato.",
      output: "Dati isolati (Data Silos) stanno impattando il tuo **Customer Churn**? 📉\n\nLa nostra ultima ricerca quantitativa lo conferma: l'ottimizzazione dei Data Silos è direttamente correlata a una significativa riduzione del tasso di abbandono dei clienti (p < 0.01).\n\nAbbiamo validato empiricamente il nostro **Framework proprietario di gestione del ciclo di vita del dato**. Non si tratta solo di teoria; sono risultati concreti che impattano la tua *bottom line*.\n\nSbloccare il valore dei dati è la chiave per fidelizzare i clienti. 🔑\n\nCosa stai facendo oggi per rompere i tuoi silos interni?\n\n#DataStrategy #CustomerRetention #B2BAnalytics #DataManagement #BusinessIntelligence",
      qualityScore: 96,
      reasoning: "Il testo revisionato è eccellente per un contesto B2B sui social media. Concettualmente è breve, d'impatto, e trasforma un risultato tecnico (correlazione p<0.01) in un problema tangibile per il target (\"Dati isolati stanno impattando il tuo Customer Churn?\"). La leggibilità è migliorata grazie all'uso sapiente di grassetti, emoji pertinenti (📉, 🔑) e paragrafi brevi, rendendo la lettura veloce. Gli hashtag sono molto pertinenti e coprono argomenti chiave (Data Strategy, Retention, BI). Infine, lo stimolo all'engagement è forte grazie alla domanda diretta alla fine: \"Cosa stai facendo oggi per rompere i tuoi silos interni?\". Un ottimo equilibrio tra rigore analitico e comunicazione social."
    },
    {
      profile: "Comunicatore di Crisi PR",
      title: "Revisione di Dichiarazione Istituzionale per Empatia e Trasparenza",
      context: "A seguito di una interruzione di servizio imprevista, l'AI ha generato una bozza di dichiarazione eccessivamente tecnica e difensiva, minimizzando l'impatto e mancando di empatia, un errore comune nella comunicazione di crisi. Il profilo deve umanizzare il testo, assicurando chiarezza, trasparenza e un tono che ricostruisca la fiducia degli stakeholder.",
      input: "Si informa la clientela che si è verificata una anomalia operativa non prevedibile nell'infrastruttura di rete. I nostri tecnici stanno procedendo alla diagnostica avanzata e alla successiva risoluzione. L'incidente è sotto controllo. Non appena l'analisi forense sarà completata, verrà fornito un *update*. Ci scusiamo per l'inconveniente tecnico.",
      output: "**COMUNICATO UFFICIALE URGENTE**\n\n**OGGETTO:** Interruzione del Servizio dovuta a Anomalia Tecnica Imprevista\n\nSi comunica alla nostra clientela che, a partire dalle ore [INSERIRE ORA DI INIZIO STIMA, ES. XX:XX], la nostra infrastruttura di rete ha subito un’anomalia operativa di natura tecnica non prevista.\n\nComprendiamo profondamente il disagio e l'impatto che questa interruzione sta causando alle Vostre attività. La nostra priorità assoluta è ripristinare la piena operatività nel minor tempo possibile.\n\n**Azioni Intraprese:**\n\n1.  Immediatamente dopo la rilevazione, i nostri team tecnici specialistici hanno avviato la procedura di diagnostica avanzata per isolare la causa principale dell'anomalia.\n2.  Tutte le risorse necessarie sono state mobilitate per implementare la risoluzione definitiva dell’incidente. La situazione è sotto costante monitoraggio da parte del nostro personale di crisi.\n\n**Prossimi Passi:**\n\nAttualmente, la fase di analisi e intervento è in corso. Ci impegniamo a fornire un aggiornamento formale sullo stato di avanzamento e una stima di ripristino entro le ore [INSERIRE ORA DEL PROSSIMO AGGIORNAMENTO].\n\nCi scusiamo sinceramente per l'interruzione del servizio e ringraziamo per la Vostra pazienza e comprensione mentre lavoriamo per risolvere questa criticità.\n\n[Nome dell'Azienda/Ente]\nServizio Comunicazioni di Crisi",
      qualityScore: 97,
      reasoning: "Il testo revisionato eccelle nell'adempimento dei requisiti di comunicazione di crisi. Il Tono è decisamente più empatico e responsabile rispetto all'originale (uso di espressioni come 'Comprendiamo profondamente il disagio' e 'scusiamo sinceramente'). La Chiarezza del Messaggio è ottima: definisce chiaramente il problema (Interruzione del Servizio), le azioni intraprese (diagnostica, mobilitazione risorse) e i passi futuri (orario del prossimo aggiornamento). L'assenza di ambiguità è rafforzata dalla struttura chiara (OGGETTO, Azioni Intraprese, Prossimi Passi) e dall'uso di placeholder specifici ([INSERIRE ORA]) che indicano la necessità di tempistiche chiare nel rilascio finale. È un comunicato robusto e professionale."
    },
    {
      profile: "Traduttore Tecnico IT",
      title: "Localizzazione Accurata di Manuali Tecnici e Garanzia di Terminologia",
      context: "Un manuale utente per un nuovo hardware è stato tradotto dall'inglese all'italiano usando un traduttore generico. Il risultato è letterale e usa termini tecnici scorretti (es. \"bottone\" invece di \"pulsante\", \"salvare\" al posto di \"memorizzare\"). Questo profilo assicura che il linguaggio IT sia idiomatico e preciso per il mercato locale.",
      input: "Per implementare il *firmware* più aggiornato, l'utente deve accedere alla pagina di configurazione via interfaccia web. Cliccare sulla sezione 'aggiornamento' nel *menu* in basso. Caricare il *file zip* con la versione 4.5. Il sistema poi *re-boottarà* (re-boot) automaticamente per installare.",
      output: "Per aggiornare il \"cervello\" interno del dispositivo (il **firmware**), segui questi semplici passaggi:\n\n1. **Entra nel pannello di controllo:** Accedi all'area di gestione del dispositivo tramite il tuo browser web (come se stessi visitando un sito internet).\n2. **Trova la sezione Aggiornamento:** Cerca il menu, solitamente posizionato in basso, e clicca sulla voce \"Aggiornamento\".\n3. **Carica il nuovo software:** Seleziona il file di aggiornamento (che è un pacchetto compresso, spesso identificato come file .zip) e caricalo nel sistema.\n4. **Attendi il riavvio:** Il dispositivo si spegnerà e si riaccenderà da solo per applicare le novità.\n\n**In sintesi:** Stai semplicemente caricando l'ultima versione del software di base per migliorare le prestazioni e la sicurezza del dispositivo, e questo avverrà automaticamente dopo aver caricato il file.",
      qualityScore: 98,
      reasoning: "Il testo revisionato eccelle nella semplificazione del gergo tecnico per un pubblico business/non-esperto. Termini come 'firmware' sono spiegati come 'cervello' interno, 'interfaccia web' come 'pannello di controllo' o 'area di gestione', e 're-boot' come 'riavvio'. I benefici sono chiari ('migliorare le prestazioni e la sicurezza'). La struttura è estremamente chiara grazie all'uso di elenchi puntati e grassetti. La sintesi finale rafforza il concetto in termini di valore. L'unica piccola nota è che 'pannello di controllo' è un termine generico, ma nel contesto delle istruzioni è perfettamente comprensibile."
    },
    {
      profile: "Specialista Comunicazioni HR",
      title: "Umanizzazione di Annunci Politiche Aziendali Sensibili",
      context: "Il dipartimento HR deve annunciare una modifica importante (e potenzialmente impopolare) nel piano di benefit. L'AI ha prodotto un annuncio troppo clinico, pieno di gergo HR (\"ottimizzazione dei costi\") e privo della sensibilità necessaria per comunicazioni che influenzano i dipendenti. Questo profilo deve umanizzare il testo, assicurando chiarezza, trasparenza e un tono che ricostruisca la fiducia degli stakeholder.",
      input: "A decorrere dal primo del prossimo mese, l'attuale piano di *wellness* subirà una ristrutturazione per l'ottimizzazione delle risorse allocabili. Il nuovo modello mira a una ri-calibrazione della spesa aziendale in conformità con i recenti obiettivi di *budget* di reparto. I dipendenti sono invitati a consultare la documentazione PDF allegata.",
      output: "**Oggetto: Aggiornamento Importante: Evoluzione del Programma di Benessere Aziendale**\n\nGentili Dipendenti,\n\nVi scriviamo per comunicarvi un aggiornamento relativo al nostro attuale Programma di Benessere Aziendale (Wellness Program).\n\nA partire dal **primo giorno del prossimo mese**, implementeremo una riorganizzazione del programma esistente. Questa iniziativa è stata definita in linea con le nostre politiche di ottimizzazione delle risorse aziendali e con gli obiettivi di bilancio stabiliti per il prossimo periodo.\n\nIl nostro obiettivo rimane quello di sostenere il benessere di tutti i nostri collaboratori, adattando al contempo la struttura del programma per garantirne la sostenibilità e massimizzare l'efficacia delle risorse dedicate.\n\nVi invitiamo a prendere visione della documentazione allegata a questa comunicazione, che fornisce un dettaglio completo delle modifiche introdotte, del nuovo modello operativo e delle tempistiche di implementazione.\n\nPer qualsiasi domanda o necessità di chiarimento sul nuovo programma, vi preghiamo di contattare l'Ufficio Risorse Umane.\n\nRingraziamo per la collaborazione e la comprensione.\n\nCordiali saluti,\n\nL'Ufficio Risorse Umane",
      qualityScore: 95,
      reasoning: "Il testo revisionato è significativamente migliore dell'originale. Migliora drasticamente la chiarezza trasformando un linguaggio aziendale denso e ambiguo ('ristrutturazione per l'ottimizzazione delle risorse allocabili') in termini più accessibili ('riorganizzazione del programma esistente', 'ottimizzazione delle risorse aziendali'). Il tono è professionale ma notevolmente più empatico, soprattutto grazie all'affermazione che l'obiettivo rimane 'sostenere il benessere di tutti i nostri collaboratori'. L'uso di grassetto per la data rende l'informazione cruciale immediatamente visibile. È presente un invito chiaro all'azione (consultare la documentazione) e un punto di contatto definito (Ufficio Risorse Umane), eliminando l'ambiguità dell'originale. L'inclusività è gestita attraverso l'uso di 'Gentili Dipendenti' e 'collaboratori'. Non ci sono ambiguità. Il punteggio è alto per l'eccellente miglioramento della comunicazione."
    },
    {
      profile: "Ottimizzatore Email di Vendita",
      title: "Personalizzazione e Ammorbidimento del Tono per Cold Outreach",
      context: "Un addetto alle vendite ha generato una bozza di cold email con l'AI. L'email è generica, troppo formale e si concentra sui \"vantaggi aziendali\" anziché sul problema specifico del potenziale cliente, risultando respingente. Questo profilo aggiunge calore, personalizzazione e rende il Call to Action più amichevole e meno aggressivo.",
      input: "Oggetto: Proposta di Sinergia per Incremento Efficienza Operativa\n\nGentile Sig./Sig.ra [Cognome], La presente per introdurre la nostra soluzione di *Enterprise Resource Planning*. Riteniamo che il nostro prodotto possa fungere da catalizzatore per l'incremento di marginalità della Sua azienda. Desideriamo programmare un incontro di 15 minuti per una dimostrazione. Attendiamo un Suo cortese riscontro.",
      output: "Oggetto: [Nome Azienda Contatto], un modo per ottimizzare [Area Specifica di Dolore/Obiettivo]\n\nCiao [Nome Contatto],\n\nHo notato che [Nome Azienda Contatto] sta facendo un ottimo lavoro nel [menzionare un successo o un'area di interesse].\n\nMolte aziende nel vostro settore faticano a mantenere l'efficienza operativa quando scalano, spesso a causa di sistemi ERP frammentati o obsoleti che rallentano i processi chiave.\n\nLa nostra soluzione aiuta aziende come la vostra a **centralizzare i dati operativi, riducendo i colli di bottiglia del 20%** e liberando risorse preziose.\n\nSarebbe utile parlarne per 10 minuti la prossima settimana per capire se questo può essere rilevante per voi?\n\nCordiali saluti,\n\n[Tuo Nome]\n[Tua Posizione]",
      qualityScore: 92,
      reasoning: "Il testo revisionato è significativamente migliore dell'originale. L'Oggetto è personalizzato e mirato. Il corpo dell'email sposta il focus dal 'noi' al 'loro' (focalizzazione sul cliente), menzionando un successo specifico e identificando un 'dolore' comune nel loro settore (sistemi ERP frammentati). La proposta di valore è quantificabile (\"riducendo i colli di bottiglia del 20%\"). La Call to Action è a basso attrito (\"Sarebbe utile parlarne per 10 minuti\"), richiedendo una piccola frazione di tempo invece di una 'dimostrazione' generica e impegnativa. Migliorabile solo l'introduzione iniziale per essere ancora più specifica sul perché si sta scrivendo, ma la struttura generale è eccellente per un template a freddo."
    },
    {
      profile: "L'Umanizzatore",
      title: "Eliminazione di Pattern Linguistici Robotici per Autenticità",
      context: "Un contenuto web deve superare i rilevatori di AI e connettersi meglio con il lettore. L'AI ha utilizzato frasi prevedibili, strutture passive frequenti e un vocabolario ripetitivo, caratteristiche tipiche del testo generato automaticamente. Questo profilo rompe la monotonia, introducendo variazione stilistica, espressioni idiomatiche e una \"prospettiva personale\".",
      input: "È fondamentale notare che la *blockchain* offre una soluzione per la tracciabilità degli asset. Questa tecnologia permette una verifica trasparente delle transazioni. Di conseguenza, il rischio di frode è mitigato efficacemente. In aggiunta, la decentralizzazione inerente al sistema garantisce un elevato livello di sicurezza e di integrità dei dati.",
      output: "Sai, una delle cose più interessanti della blockchain è come risolve il problema della tracciabilità degli asset. Non è solo una moda, è uno strumento potente. Permette di vedere chiaramente e senza filtri ogni passaggio di una transazione, il che, diciamocelo, riduce drasticamente le possibilità di frode. E c'è di più: proprio perché è decentralizzata, ti dà quella sensazione di sicurezza granitica e ti assicura che i dati rimangano intoccabili e corretti.",
      qualityScore: 95,
      reasoning: "Il testo revisionato è significativamente più umano. L'introduzione con \"Sai\" stabilisce immediatamente un tono colloquiale e confidenziale. L'uso di espressioni come \"Non è solo una moda, è uno strumento potente\", \"diciamocelo\", e la metafora \"sicurezza granitica\" sostituiscono la freddezza del linguaggio tecnico originale con un'esposizione più coinvolgente e relazionale. La sintassi è variata e il tono è caldo ed empatico, riuscendo a trasformare un concetto complesso in una conversazione accessibile."
    }
  ];

  const [selectedUseCase, setSelectedUseCase] = useState(useCasesData[0]);

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-6xl"> {/* Max width aumentata per più spazio */}
        <header className="mb-16 text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold text-blue-400 mb-4 leading-tight">
            I Nostri Use Cases: Dalla Sfida al Successo Certificato
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Scopri come Text Validator AI trasforma i tuoi testi più critici, profilo per profilo, garantendo qualità e impatto.
          </p>
        </header>

        {/* Selettore dei Profili */}
        <section className="mb-12 py-8 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
          <h2 className="text-3xl font-bold text-center text-blue-300 mb-6">Seleziona un Profilo AI per Vedere un Esempio Pratico</h2>
          <div className="flex flex-wrap justify-center gap-4 px-4">
            {useCasesData.map((uc, index) => (
              <button
                key={index}
                onClick={() => setSelectedUseCase(uc)}
                className={`px-6 py-3 rounded-full text-lg font-semibold transition-all duration-200 shadow-md
                  ${selectedUseCase.profile === uc.profile
                    ? 'bg-blue-600 text-white transform scale-105 border-blue-500 border-2'
                    : 'bg-gray-700 text-gray-200 hover:bg-gray-600 border-gray-700 border'
                  }`}
              >
                {uc.profile}
              </button>
            ))}
          </div>
        </section>

        {/* Dettagli dello Use Case Selezionato */}
        {selectedUseCase && (
          <section className="mb-16">
            <h2 className="text-4xl font-bold text-blue-400 mb-6 text-center">
              {selectedUseCase.profile}: {selectedUseCase.title}
            </h2>
            <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
              <h3 className="text-2xl font-semibold text-blue-300 mb-4">Contesto e Problema:</h3>
              <p className="text-gray-300 mb-6 italic">{selectedUseCase.context}</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                <div>
                  <h3 className="text-2xl font-semibold text-white mb-3">Input Grezzo (Prima)</h3>
                  <textarea
                    rows={10}
                    readOnly
                    value={selectedUseCase.input}
                    className="w-full bg-gray-900 text-gray-200 p-4 rounded-lg border border-gray-700 resize-none shadow-inner"
                  ></textarea>
                </div>
                <div>
                  <h3 className="text-2xl font-semibold text-white mb-3">Output Certificato (Dopo)</h3>
                  <textarea
                    rows={10}
                    readOnly
                    value={selectedUseCase.output}
                    className="w-full bg-gray-900 text-gray-200 p-4 rounded-lg border border-gray-700 resize-none shadow-inner"
                  ></textarea>
                </div>
              </div>

              {/* Report di Qualità */}
              <div className="mt-8 p-6 rounded-xl bg-gray-900 border border-gray-700 shadow-md">
                <h3 className="text-2xl font-semibold text-green-400 mb-4 text-center">Report di Qualità AI² Verified™</h3>
                <div className="flex items-center justify-center text-center mb-4">
                  <p className="text-6xl font-bold text-green-400">{selectedUseCase.qualityScore}</p>
                  <p className="ml-2 text-2xl text-gray-400">/ 100</p>
                </div>
                <p className="text-gray-300 italic text-center">
                  {selectedUseCase.reasoning}
                </p>
              </div>
            </div>
          </section>
        )}

      </div>
    </main>
  );
}