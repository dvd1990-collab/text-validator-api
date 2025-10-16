// src/app/use-cases/page.tsx
"use client";

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

// --- DATI PER VALIDATOR (Completi, dal tuo file originale) ---
const validatorUseCases = [
    {
      profile: "Generico",
      title: "Pulizia di Bozze Interne e Normalizzazione del Tono",
      context: "Il team di Ricerca e Sviluppo utilizza un modello AI per riassumere rapidamente i progressi settimanali da condividere con la dirigenza. L'output grezzo è spesso tecnicamente corretto ma ha un tono eccessivamente formale e frasi lunghe che rallentano la lettura e la comprensione. Questo profilo serve a rendere il riassunto immediatamente fruibile e con un tono aziendale neutro e professionale.",
      input: "L'analisi inerente all'architettura modulare ha evidenziato una non-linearità significativa nella fase di implementazione del modulo API. L'ottimizzazione dei parametri algoritmici è stata completata; la performance è stata incrementata del 12.4% rispetto al benchmark stabilito, in conformità con le specifiche di progetto. Le prossime azioni implicano la migrazione dell'infrastruttura di testing.",
      output: "L'analisi dell'architettura modulare ha rilevato una non-linearità significativa durante la fase di implementazione del modulo API. L'ottimizzazione dei parametri algoritmici è conclusa; la performance è aumentata del 12,4% rispetto al benchmark stabilito, in piena conformità con le specifiche di progetto. Le prossime azioni pianificate includono la migrazione dell'infrastruttura di testing.",
      qualityScore: 93,
      reasoning: "Il testo revisionato è un miglioramento incrementale ma solido. La scelta lessicale è più precisa e la formattazione del numero è standardizzata, migliorando la chiarezza per un pubblico B2B."
    },
    {
      profile: "PM - Interpretazione Trascrizioni",
      title: "Sintesi Azionabile da Trascrizione Riunione Tecnica",
      context: "Dopo una riunione strategica, una trascrizione automatica verbosa e disorganizzata deve essere trasformata in un verbale conciso, estraendo decisioni chiave, next steps e risorse assegnate.",
      input: "Quindi, uhm, sì, abbiamo concordato che, beh, aspetta, devo ricontrollare, ma credo che il *deliverable* A... dovrebbe essere re-prioritizzato. Mario ha detto che non è fattibile entro fine Q2. Quindi, la decisione è, forse, spostarlo. Dunque, il *task* per il team di sviluppo è: riallocare le risorse.",
      output: "**OGGETTO:** Riprogrammazione Deliverable A\n\n**DECISIONE:** Il Deliverable A deve essere ri-prioritizzato a causa di problemi di fattibilità entro la scadenza Q2.\n\n**ATTIVITÀ:** Riassegnazione delle risorse.\n\n**RESPONSABILE:** Team di Sviluppo.",
      qualityScore: 98,
      reasoning: "Eccellente trasformazione da un discorso vago a una comunicazione strutturata e orientata all'azione, utilizzando intestazioni chiare che facilitano la rapida comprensione."
    },
    {
      profile: "Copywriter Persuasivo",
      title: "Trasformazione di Specifiche in Vantaggi di Marketing",
      context: "Le specifiche tecniche di un nuovo software B2B devono essere trasformate in un testo persuasivo per una landing page, focalizzandosi sui benefici per il cliente.",
      input: "Il prodotto X offre un'interfaccia utente configurabile con un *dashboard* analitico integrato. Il suo motore asincrono riduce il *latency* del 40%.",
      output: "**Sblocca la Tua Efficienza!** Stanco di interfacce che ti rallentano? Con la nostra **Dashboard Analitica Personalizzabile**, avrai il controllo totale. Dimentica l'attesa: il nostro motore abbatte la latenza del **40%**, regalandoti decisioni istantanee.",
      qualityScore: 95,
      reasoning: "Trasforma efficacemente le specifiche in benefici tangibili, usando un linguaggio energico, domande retoriche e una chiara Call to Action implicita."
    },
    {
      profile: "Revisore Legale/Regolatorio",
      title: "Verifica di Chiarezza nel Linguaggio Contrattuale",
      context: "La bozza di una clausola di Termini di Servizio generata da un AI è legalmente valida ma ambigua e contorta, aumentando il rischio di contenzioso.",
      input: "La parte sottoscritta si impegna a non divulgare, né direttamente né per interposta persona, informazioni proprietarie... a meno che e fintanto che un simile trasferimento non sia previamente autorizzato...",
      output: "La Parte Sottoscritta si impegna a non divulgare, direttamente o tramite terzi, Informazioni Proprietarie, salvo preventivo consenso scritto del Licenziante per ogni specifico trasferimento.",
      qualityScore: 92,
      reasoning: "Migliora notevolmente la chiarezza e la formalità, avvicinandosi a uno standard contrattuale più elevato. La capitalizzazione è appropriata e il linguaggio più incisivo."
    },
    {
      profile: "Scrittore di Newsletter",
      title: "Ristrutturazione di Contenuto per l'Email Engagement",
      context: "Un lungo articolo di blog è stato riassunto da un'AI, ma il testo è troppo denso e non formattato per essere letto rapidamente in una newsletter via email.",
      input: "Nel panorama attuale della digitalizzazione B2B, l'adozione di soluzioni cloud-native sta ridefinendo i parametri di efficienza operativa. Le aziende che hanno implementato queste architetture hanno registrato miglioramenti...",
      output: "**Oggetto:** Stai ancora usando architetture di ieri?\n\nIl panorama digitale B2B si sta ridisegnando. L'adozione di architetture **cloud-native** non è più un optional, è il nuovo benchmark.\n\n**Vantaggi Immediati:**\n* **Scalabilità Elastica:** Cresci in ore, non mesi.\n* **Resilienza Garantita:** Resisti a picchi di traffico senza interruzioni.",
      qualityScore: 95,
      reasoning: "Eccellente trasformazione in una newsletter coinvolgente. Il gancio iniziale crea urgenza, mentre l'uso di grassetti e liste puntate spezza il muro di testo, migliorando la leggibilità."
    },
    {
      profile: "Social Media Manager B2B",
      title: "Adattamento di Dati Tecnici per Post LinkedIn",
      context: "Un riassunto accademico e pieno di gergo di uno studio di settore deve essere trasformato in un post LinkedIn breve, con un hook forte e hashtag pertinenti.",
      input: "Il nostro studio quantitativo ha rivelato una correlazione positiva statisticamente significativa (p < 0.01) tra l'ottimizzazione del *data silo* e la riduzione del *churn rate*...",
      output: "I tuoi Dati sono isolati (Data Silos) e stanno impattando il **Customer Churn**? 📉\n\nLa nostra ultima ricerca lo conferma: ottimizzare i Data Silos riduce significativamente l'abbandono dei clienti.\n\n#DataStrategy #CustomerRetention #B2BAnalytics",
      qualityScore: 96,
      reasoning: "Ottimo adattamento per i social. Trasforma un risultato tecnico in un problema tangibile, usa emoji pertinenti e hashtag mirati per massimizzare l'engagement."
    },
    {
      profile: "Comunicatore di Crisi PR",
      title: "Revisione di Dichiarazione Istituzionale per Empatia",
      context: "A seguito di un'interruzione di servizio, una bozza generata da AI risulta troppo tecnica e difensiva, mancando di empatia e trasparenza.",
      input: "Si informa la clientela che si è verificata una anomalia operativa non prevedibile. I nostri tecnici stanno procedendo alla diagnostica. L'incidente è sotto controllo.",
      output: "**COMUNICATO UFFICIALE URGENTE**\n\nComprendiamo profondamente il disagio e l'impatto che questa interruzione sta causando. La nostra priorità assoluta è ripristinare la piena operatività.\n\n**Azioni Intraprese:**\n1. I nostri team hanno avviato la diagnostica per isolare la causa.\n2. Tutte le risorse sono state mobilitate.\n\nCi impegniamo a fornire un aggiornamento entro le ore [XX:XX].",
      qualityScore: 97,
      reasoning: "Eccellente. Il tono è decisamente più empatico e responsabile. La struttura chiara (Azioni, Prossimi Passi) e l'assenza di ambiguità costruiscono fiducia."
    },
    {
      profile: "Traduttore Tecnico IT",
      title: "Localizzazione Accurata di Manuali Tecnici",
      context: "Un manuale utente tradotto automaticamente dall'inglese all'italiano usa termini tecnici scorretti e letterali, risultando incomprensibile.",
      input: "Per implementare il *firmware* più aggiornato, l'utente deve accedere alla pagina di configurazione... Cliccare 'aggiornamento'... Il sistema poi *re-boottarà*...",
      output: "Per aggiornare il software interno del dispositivo (**firmware**), segui questi passaggi:\n\n1. **Entra nel pannello di controllo** tramite browser.\n2. Clicca sulla voce **'Aggiornamento'**.\n3. Carica il file di aggiornamento.\n4. **Attendi il riavvio automatico** del dispositivo.",
      qualityScore: 98,
      reasoning: "Perfetta semplificazione del gergo tecnico. Spiega termini come 'firmware' e 're-boot' in modo accessibile, rendendo le istruzioni chiare per un pubblico non esperto."
    },
    {
      profile: "Specialista Comunicazioni HR",
      title: "Umanizzazione di Annunci di Politiche Aziendali",
      context: "Un annuncio HR su una modifica ai benefit aziendali è troppo clinico e pieno di gergo, mancando della sensibilità necessaria per comunicazioni ai dipendenti.",
      input: "A decorrere dal primo del prossimo mese, l'attuale piano di *wellness* subirà una ristrutturazione per l'ottimizzazione delle risorse allocabili...",
      output: "**Oggetto: Aggiornamento Importante sul Programma di Benessere Aziendale**\n\nGentili Dipendenti,\nVi scriviamo per comunicarvi una riorganizzazione del programma di benessere a partire dal **primo giorno del prossimo mese**.\n\nIl nostro obiettivo rimane sostenere il benessere di tutti, garantendo al contempo la sostenibilità del programma. Per ogni domanda, contattate l'Ufficio Risorse Umane.",
      qualityScore: 95,
      reasoning: "Migliora drasticamente la chiarezza e il tono, rendendolo più empatico e diretto. L'uso del grassetto e l'invito al contatto eliminano l'ambiguità dell'originale."
    },
    {
      profile: "Ottimizzatore Email di Vendita",
      title: "Personalizzazione di un Template di Cold Email",
      context: "Una bozza di cold email generata da AI è troppo generica e formale, focalizzata sull'azienda e non sul problema del cliente, risultando inefficace.",
      input: "Oggetto: Proposta di Sinergia...\nGentile Sig./Sig.ra [Cognome], La presente per introdurre la nostra soluzione di *Enterprise Resource Planning*...",
      output: "Oggetto: [Nome Azienda Contatto], un'idea per [Area di Dolore]\n\nCiao [Nome Contatto],\nHo notato il vostro lavoro in [settore]. Molte aziende faticano a scalare a causa di sistemi ERP frammentati.\n\nLa nostra soluzione aiuta a **ridurre i colli di bottiglia del 20%**. Sarebbe utile parlarne per 10 minuti?",
      qualityScore: 92,
      reasoning: "Eccellente. Sposta il focus dal 'noi' al 'loro', personalizza l'oggetto, quantifica il valore e propone una Call to Action a basso attrito, aumentando le probabilità di risposta."
    },
    {
      profile: "L'Umanizzatore",
      title: "Eliminazione di Pattern Linguistici Robotici",
      context: "Un testo generato da AI usa frasi prevedibili e un vocabolario ripetitivo, risultando innaturale e facilmente identificabile come artificiale.",
      input: "È fondamentale notare che la *blockchain* offre una soluzione per la tracciabilità... Questa tecnologia permette una verifica trasparente... Di conseguenza, il rischio di frode è mitigato...",
      output: "Sai, una delle cose più interessanti della blockchain è come risolve il problema della tracciabilità. Permette di vedere ogni passaggio di una transazione, il che, diciamocelo, riduce drasticamente le possibilità di frode. E in più, essendo decentralizzata, ti dà quella sensazione di sicurezza granitica.",
      qualityScore: 95,
      reasoning: "Trasformazione eccellente. L'uso di un tono colloquiale ('Sai', 'diciamocelo') e di metafore ('sicurezza granitica') rende il testo più umano, coinvolgente e relazionale."
    }
];

// --- DATI PER INTERPRETER (DAI NUOVI DOCUMENTI) ---
const interpreterUseCases = [
    {
      profile: "Analista Contratto di Vendita",
      title: "Identificare clausole di Responsabilità e Recesso Iniquo",
      context: "Un Sales Manager deve chiudere un contratto SaaS da 800k€, ma teme che le modifiche del cliente nascondano rischi finanziari illimitati.",
      input: "ARTICOLO 14 – LIMITAZIONE DI RESPONSABILITÀ... la responsabilità complessiva del Fornitore... è limitata... ai sei (6) mesi precedenti... Tuttavia, tale limitazione non si applica, e il Fornitore sarà ritenuto responsabile senza tetto massimo (illimitatamente), per perdite derivanti da... (iii) inadempienza grave...",
      output: `### Punti Chiave\n* **Cap Responsabilità:** Limitato ai pagamenti degli ultimi 6 mesi.\n* **Responsabilità Illimitata:** Si applica per violazioni IP, frode e **inadempienza grave**.\n* **Recesso Cliente:** Possibile con 30 giorni di preavviso se l'uptime è < 99.5%.\n\n### 🚩 Punti di Attenzione\n* Il cap di responsabilità è potenzialmente molto basso.\n* La definizione di 'inadempienza grave' è in un altro allegato.`,
      qualityScore: 98,
      reasoning: "L'analisi estrae con precisione i punti commercialmente e legalmente più critici, evidenziando correttamente i rischi per un manager."
    },
    {
      profile: "Revisore Contratto di Acquisto",
      title: "Individuare clausole di tacita proroga",
      context: "Un responsabile Procurement deve firmare un contratto quadro di 10 anni. Il documento di 70 pagine potrebbe nascondere clausole vessatorie come la 'tacita proroga'.",
      input: "ARTICOLO 7 – DURATA E RINNOVO. Il presente Contratto... si rinnoverà tacitamente per periodi aggiuntivi di un (1) anno, a meno che una delle Parti non notifichi... con un preavviso di almeno sei (6) mesi... ARTICOLO 8 – SOSPENSIONE. Il Fornitore si riserva la facoltà di sospendere l'esecuzione... in caso di incremento dei costi... superiore al 15%...",
      output: `### Punti Chiave\n* **Durata e Rinnovo:** Il contratto si **rinnova automaticamente** ogni anno.\n* **Preavviso di Disdetta:** Necessario un preavviso di **6 mesi**.\n\n### 🚩 Punti di Attenzione per l'Acquirente\n*   **Rischio di Blocco:** Il rinnovo tacito con preavviso così lungo è molto vincolante.\n*   **Sospensione Unilaterale:** Il fornitore può interrompere le consegne se i suoi costi aumentano del 15%.\n*   **Foro Competente Svantaggioso:** Le dispute legali si discutono presso la sede del fornitore.`,
      qualityScore: 98,
      reasoning: "L'analisi è fortemente orientata alla protezione dell'acquirente, identificando con precisione i tre rischi principali: il vincolo del rinnovo, la sospensione unilaterale e la giurisdizione sfavorevole."
    },
    {
      profile: "Estrattore P&L Aziendale",
      title: "Calcolo del ROA (Return on Assets) in tempo reale",
      context: "Un analista M&A deve valutare decine di target, filtrando quelli con redditività dell'attivo (ROA) debole, ma i dati sono sparsi in PDF non standardizzati.",
      input: "CONTO ECONOMICO ANNO N: RICAVI NETTI: € 5,120,400. Costo del Venduto: € 2,980,150. UTILE NETTO: € 749,450. Da Stato Patrimoniale: Attività Totali al 31/12/N-1: € 6,550,900, Attività Totali al 31/12/N: € 7,210,100.",
      output: `### KPI di Profitto (Anno N)\n* **Margine Lordo:** € 2,140,250 (41.80%)\n* **Utile Netto:** € 749,450 (14.64%)\n\n### Calcolo ROA\n* **Attività Totali Medie:** (€ 6.550.900 + € 7.210.100) / 2 = € 6.880.500\n* **ROA (Return on Assets):** (€ 749.450 / € 6.880.500) = **10.89%**\n\n### Sintesi Esecutiva\nL'azienda dimostra una solida redditività operativa e un ROA del 10.89%, indicando un uso efficiente delle proprie risorse.`,
      qualityScore: 99,
      reasoning: "Eccellente. Non solo estrae i dati finanziari, ma esegue correttamente il calcolo intermedio (Attività Totali Medie) per arrivare a un ROA preciso, fornendo un insight di alto valore."
    },
    {
      profile: "Analista Bilancio Aziendale",
      title: "Calcolare il Quick Ratio escludendo l'inventario illiquido",
      context: "Un Credit Manager sospetta che il bilancio di un partner sia gonfiato da inventario obsoleto. Ha bisogno del Quick Ratio per valutare la reale capacità di far fronte agli impegni finanziari.",
      input: "ATTIVITÀ CORRENTI (migliaia di €): Cassa: 1,450. Titoli Negoziabili: 320. Crediti: 2,890. Rimanenze: 5,100 (Nota: 45% non fungibili). Totale Attività Correnti: 9,940. PASSIVITÀ CORRENTI: 6,450.",
      output: `### Calcolo Rapporti di Liquidità\n* **Attività Correnti:** 9,940 K€\n* **Passività Correnti:** 6,450 K€\n* **Current Ratio:** 9,940 / 6,450 = **1.54** (Adeguato)\n\n### 🚩 Analisi di Liquidità Immediata\n* **Attività Liquide (senza rimanenze):** 1,450 (Cassa) + 320 (Titoli) + 2,890 (Crediti) = 4,660 K€\n* **Quick Ratio (Acid-Test):** 4,660 / 6,450 = **0.72** (CRITICO)\n\n### Sintesi Esecutiva\nIl Quick Ratio a 0.72 è un segnale di allarme. L'azienda dipende pesantemente dalla vendita del suo inventario (di cui il 45% è poco liquido) per coprire i debiti a breve.`,
      qualityScore: 97,
      reasoning: "L'analisi è precisa e orientata al rischio. Isola correttamente il Quick Ratio come indicatore chiave, evidenziando la dipendenza critica dall'inventario e fornendo una diagnosi chiara del rischio."
    },
    {
      profile: "Sintesi Legale Breve",
      title: "Sintetizzare una Circolare della Banca d'Italia",
      context: "Un Compliance Officer riceve una densa circolare normativa e deve identificare immediatamente l'impatto pratico sulle procedure interne di governance.",
      input: "CIRCOLARE N. XX... Art. 18.2 – Struttura. L'organo con funzione di supervisione strategica... Il presidente... favorisce in modo neutrale la dialettica tra componenti esecutivi e non esecutivi e sollecita la partecipazione attiva dei componenti non esecutivi...",
      output: `### Disposizione Normativa Principale\nL'organo di supervisione deve monitorare attivamente i controlli interni. \n\n### Ruolo Chiave del Presidente\nIl Presidente ha il **dovere attivo** di:\n* Agire come **facilitatore neutrale**.\n* Promuovere una discussione equilibrata, sollecitando l'intervento dei membri **non esecutivi**.\n* Assicurare che le informazioni per le decisioni siano fornite con **largo anticipo**.\n\n### Implicazione Pratica\nLa norma mira a prevenire decisioni dominate dalle posizioni esecutive, rafforzando il ruolo di controllo dei membri indipendenti.`,
      qualityScore: 96,
      reasoning: "L'analisi distilla con successo il gergo burocratico in obblighi pratici e chiari. L'enfasi sul 'dovere attivo' del presidente e sul ruolo dei membri non esecutivi è centrata e utile."
    },
    {
      profile: "Revisore Polizza Assicurativa",
      title: "Mappare le esclusioni di copertura Cyber",
      context: "Un Risk Manager deve sapere se la polizza All Risks copre l'Interruzione di Attività (BI) causata da attacchi ransomware o se è necessaria un'estensione.",
      input: "SEZIONE V – RISCHI ESCLUSI. Questa Polizza non copre... qualsiasi interruzione dell'attività (Business Interruption) che non sia conseguenza diretta di un danno materiale e fisico... (e) Eventi di origine elettronica o cyber... a meno che non sia stata acquistata una specifica Estensione (Rider CYB 3.1).",
      output: `### Analisi Copertura Cyber e BI\n* **Interruzione di Attività (BI):** Coperta **solo** se causata da un **danno fisico diretto** all'immobile o ai macchinari.\n* **Eventi Cyber:** **ESCLUSI** dalla copertura base.\n\n### 🚩 Rischio Critico Identificato\nSenza l'acquisto dell'estensione specifica **Rider CYB 3.1**, l'azienda è **totalmente scoperta** per qualsiasi perdita derivante da attacchi informatici, inclusa l'interruzione dell'attività causata da ransomware.`,
      qualityScore: 99,
      reasoning: "Perfetto. L'analisi va dritta al punto, separando le due condizioni (BI e Cyber) e collegandole per arrivare alla conclusione critica: senza il Rider, non c'è copertura. Estremamente utile."
    },
    {
      profile: "Verificatore Fatture/Bollette",
      title: "Validare la conformità tecnica di una fattura elettronica",
      context: "Il reparto contabilità riceve notifiche di scarto dallo SDI per errori tecnici (firma digitale non valida) che bloccano i pagamenti e creano tensioni con i fornitori.",
      input: "NOTA DI RICEZIONE SDI: Il sistema di controllo ha rilevato il codice errore 00102: 'file non integro o firma digitale non valida/scaduta'. La Certificazione Temporale... (Codice 00104) non è confermata... Il pagamento è Sospeso...",
      output: `### Stato della Fattura\n**Richiede Revisione / NON PAGABILE**\n\n### 🚩 Problemi Tecnici Rilevati\n* **Errore Principale (SDI 00102):** La **firma digitale** apposta sulla fattura è **non valida o scaduta**. Il file è considerato legalmente non integro.\n* **Errore Secondario (SDI 00104):** La marca temporale (Certificazione) proviene da un'autorità non riconosciuta.\n\n### Azione Richiesta\nContattare immediatamente il fornitore e richiedere una **nuova emissione della fattura** con una firma digitale valida.`,
      qualityScore: 97,
      reasoning: "L'analisi traduce correttamente i codici di errore tecnici in problemi di business concreti ('firma non valida') e fornisce un'azione richiesta chiara e inequivocabile."
    },
    {
      profile: "Estrattore Dati Fatti",
      title: "Estrarre dati chiave dai comunicati stampa dei competitor",
      context: "Un analista di Business Development deve monitorare i competitor, estraendo date di lancio, nomine di dirigenti e valori di acquisizione da lunghi comunicati stampa.",
      input: "COMUNICATO STAMPA [18/03/2024] – GammaTech... annuncia... la nomina di Sig.ra Elena Rossi a Chief Operating Officer (COO), a partire dal 15 Aprile 2024... Il lancio ufficiale... del nuovo software 'Alpha Sentinel v4.0'... è fissato per il 1° Luglio 2024... L'espansione è stata finanziata da un round Serie B... che ha valorizzato la società a 125 milioni di Euro.",
      output: `### Timeline Eventi Chiave\n* **Febbraio 2024:** Completato round di investimenti Serie B.\n* **18 Marzo 2024:** Annunciata nomina nuovo COO.\n* **15 Aprile 2024:** Elena Rossi inizia come COO.\n* **1 Luglio 2024:** Lancio del software 'Alpha Sentinel v4.0'.\n\n### Dati Estratti\n* **Nuovo COO:** Elena Rossi\n* **Valutazione Aziendale (Feb 2024):** 125 milioni di Euro`,
      qualityScore: 95,
      reasoning: "L'estrazione è rapida e precisa. Isola i dati quantitativi e nominali più importanti dal testo narrativo, organizzandoli in una timeline e in un elenco di facile consultazione."
    },
    {
      profile: "Analista Debiti/Liquidità",
      title: "Valutare il Debt Service Coverage Ratio (DSCR)",
      context: "Un analista di credito deve valutare se un'azienda genera abbastanza cassa per coprire i suoi debiti. Un DSCR inferiore a 1.25 è un segnale di allarme.",
      input: "Nota 6.4 – Cash Flow e Covenants. Il Flusso di Cassa Operativo Netto... è stato di € 1,950,000... Il Servizio del Debito Totale... ammonta a € 1,800,000... Si evidenzia che il Rischio di Liquidità rimane elevato se il DSCR non supera 1.15 in modo sostenuto.",
      output: `### Calcolo del DSCR\n* **Flusso di Cassa Operativo Netto:** € 1.950.000\n* **Servizio del Debito Totale:** € 1.800.000\n* **DSCR Calcolato:** 1.950.000 / 1.800.000 = **1.08x**\n\n### 🚩 Diagnosi del Rischio di Liquidità\n*   **RISCHIO ELEVATO:** Il DSCR calcolato di **1.08x** è **inferiore** alla soglia di sicurezza di 1.15x menzionata nel documento.\n*   L'azienda genera a malapena la cassa necessaria per coprire le rate del debito, con un margine quasi nullo.`,
      qualityScore: 99,
      reasoning: "Perfetto. L'analisi non si limita a estrarre i dati, ma esegue il calcolo critico (DSCR) e confronta il risultato con la soglia di rischio menzionata nel testo, fornendo una diagnosi immediata e di alto valore."
    },
    {
      profile: "Spiega in Parole Semplici",
      title: "Tradurre il gergo legale di una policy interna per il CEO",
      context: "Il CEO deve approvare una nuova policy sulla protezione dei dati, ma il documento è pieno di riferimenti al GDPR e termini tecnici che ne rendono difficile la comprensione rapida.",
      input: "ARTICOLO 45 – PROCESSI DI GOVERNANCE... L'organizzazione deve garantire... principi di accountability... come specificato nell'Articolo 5(2) del GDPR... la valutazione dell'impatto dei trattamenti (DPIA)... monitorare l'efficacia dei controlli di terzo livello...",
      output: `### Cosa Significa in Pratica?\nQuesto articolo obbliga l'azienda a **documentare e poter dimostrare** ogni scelta fatta sul trattamento dei dati personali. Non basta agire correttamente, bisogna provarlo.\n\n### Perché è Importante?\nServe a prevenire multe e a gestire i rischi. L'azienda deve controllare ogni tre mesi se i dati vengono usati in modo sicuro (tramite una "Valutazione d'Impatto") e che i sistemi di controllo funzionino.\n\n### Cosa Devo Fare come CEO?\n1.  **Assicurati** che esista un **Registro dei Trattamenti** sempre aggiornato.\n2.  **Verifica** che le **valutazioni di impatto** siano programmate regolarmente.\n3.  **Ricorda** che la **responsabilità finale** sulle decisioni ad alto rischio rimane tua, non può essere completamente delegata.`,
      qualityScore: 98,
      reasoning: "La traduzione dal 'legalese' al linguaggio manageriale è eccellente. Le sezioni 'Cosa Significa', 'Perché è Importante' e 'Cosa Devo Fare' sono perfette per un C-Level, trasformando concetti astratti in azioni concrete e responsabilità chiare."
    }
];

const complianceUseCases = [
	{
	profile: "Analizzatore GDPR per Comunicazioni Marketing",
	title: "Il Bundling Illegale: Consenso, Sconti e Multe Miliardarie",
	context: "Un e-commerce propone un'unica checkbox per accettare Termini di Servizio, Privacy Policy e iscrizione alla newsletter in cambio di uno sconto. Questa pratica (bundling) rende il consenso non 'liberamente dato' e non 'specifico', invalidandolo e aprendo la porta a sanzioni GDPR severe.",
	input: 'Spuntando qui, accetti i nostri Termini di Servizio, confermi di aver letto la Privacy Policy e ti iscrivi alla nostra newsletter per ricevere uno sconto del 10% e comunicazioni personalizzate.',
	output: `### Punteggio di Rischio Conformità: ALTO\nRisultati del Controllo:\n❌ Consenso: RISCHIO ALTO. Il testo raggruppa (bundling) consensi per finalità diverse (servizio, marketing, profilazione) in un'unica azione, violando il requisito di consenso specifico e granulare del GDPR. L'incentivo economico (sconto) ne compromette ulteriormente la libera espressione.\n⚠️ Informativa Privacy: ATTENZIONE. Si menziona la Privacy Policy ma non viene fornito un link diretto e immediatamente accessibile.\n❌ Diritto di Revoca: RISCHIO ALTO. Manca qualsiasi informazione su come l'utente possa revocare il consenso o disiscriversi facilmente.\nRaccomandazioni (Non Vincolanti):\nSeparare i consensi: Utilizzare checkbox distinte per i Termini di Servizio (necessari per il contratto), per l'iscrizione alla newsletter (marketing) e per le comunicazioni personalizzate (profilazione).\nRendere lo sconto accessibile indipendentemente dal consenso al marketing per garantire la libertà di scelta.\nDisclaimer: Questo è un controllo automatico e non costituisce una consulenza legale. Si raccomanda di consultare un professionista per una valutazione definitiva.`,
	}, 
	{ 
	profile: "Verificatore Anti-Bias per Annunci di Lavoro", 
	title: "La Fuga dei Talenti: Linguaggio Cifrato che Restringe il Pool di Candidati", 
	context: "Un annuncio di lavoro utilizza un linguaggio con 'codici di genere' e riferimenti all'età che, sebbene non espliciti, possono scoraggiare inconsciamente candidati qualificati, restringendo il bacino di talenti e aumentando il rischio di contenziosi per discriminazione.", 
	input: 'Cerchiamo un ninja del codice, dominante nel team di sviluppo, energico e in grado di sostenere un ambiente competitivo ad alta pressione. I recenti laureati sono incoraggiati a candidarsi.', 
	output:`### Punteggio di Rischio Conformità: ALTO\nRisultati del Controllo:\n⚠️ Linguaggio di Genere: ATTENZIONE. L'uso di termini come "dominante" e "competitivo" è spesso associato a stereotipi maschili e può scoraggiare candidate donne.\n⚠️ Riferimenti all'Età: ATTENZIONE. La menzione esplicita di "recenti laureati" e il termine "energico" possono essere interpretati come un bias a favore di candidati più giovani.\n❌ Requisiti Pertinenti: RISCHIO ALTO. Termini gergali come "ninja del codice" sono legati a una specifica "cultura-fit" e possono escludere professionisti qualificati che non si identificano con tale profilo.\nRaccomandazioni (Non Vincolanti):\nSostituire "dominante" con termini come "collaborativo" o "con forte leadership".\nRimuovere il riferimento a "recenti laureati" per non scoraggiare candidati con maggiore esperienza.\nUsare una descrizione del ruolo più oggettiva al posto di "ninja del codice".\nDisclaimer: Questo è un controllo automatico e non costituisce una consulenza legale. Si raccomanda di consultare un professionista per una valutazione definitiva.`,}, 
	{ 
	profile: "Checker per Disclaimer Finanziari (MiFID II / CONSOB)", 
	title: "Prominenza Obbligatoria: Rischio Nascosto nel Micro-Testo", 
	context: "Una comunicazione promozionale per un fondo di investimento mette in grande risalto un rendimento potenziale elevato, relegando gli avvisi di rischio a un testo piccolo e grigio, violando il requisito di 'prominenza' imposto dalla MiFID II.", 
	input: "Investendo nel nostro fondo 'Future Growth', si anticipa un rendimento potenziale stimato del 15% (i dettagli tecnici sui coefficienti beta sono nel KIID). In caratteri piccoli grigi: La volatilità storica del portafoglio suggerisce la possibilità di perdite totali del capitale investito.", 
	output:`### Punteggio di Rischio Conformità: ALTO\n	Risultati del Controllo:\n❌ Avviso di Rischio Esplicito: RISCHIO ALTO. L'avviso sulla possibilità di perdita totale del capitale è presente ma non è 'prominente'. L'uso di caratteri piccoli e di colore a basso contrasto viola il requisito di presentazione equa e bilanciata dei rischi rispetto ai benefici.\n⚠️ Linguaggio non Ingannevole: ATTENZIONE. L'enorme enfasi sul rendimento potenziale del 15% rispetto alla scarsa visibilità dei rischi può essere considerata una pratica fuorviante.\n⚠️ Chiarezza Informativa: ATTENZIONE. L'uso di gergo tecnico non spiegato come "coefficienti beta" riduce la comprensibilità per un investitore non professionale.\n	Raccomandazioni (Non Vincolanti):\n	Assicurare che il disclaimer di rischio abbia la stessa (o maggiore) prominenza visiva del claim sul rendimento.\n	Aggiungere sempre l'avviso standard: "I rendimenti passati non sono indicativi di quelli futuri".\n	Disclaimer: Questo è un controllo automatico e non costituisce una consulenza legale. Si raccomanda di consultare un professionista per una valutazione definitiva.`,}, 
	{ 
	profile: "Validatore di Claim Pubblicitari (Anti-False Advertising)", 
	title: "L'Illusione del Superlativo: Claim Non Verificabili e UCPD", 
	context: "Un testo pubblicitario utilizza affermazioni assolute e superlative ('indiscutibilmente il più veloce') senza fornire alcuna prova o fonte a supporto, esponendo l'azienda a sanzioni per pubblicità ingannevole secondo la Direttiva sulle Pratiche Commerciali Sleali (UCPD).", 
	input: "La Nostra Soluzione è Indiscutibilmente la Più Veloce sul Mercato, offrendo un tempo di risposta senza pari e un'efficienza straordinaria.", 
	output:`### Punteggio di Rischio Conformità: ALTO\n	Risultati del Controllo:\n❌ Verificabilità dei Claim: RISCHIO ALTO. I claim "Indiscutibilmente la Più Veloce" e "senza pari" sono superlativi assoluti non supportati da alcuna fonte, benchmark o dato verificabile citato nel testo.\n❌ Correttezza Comparativa: RISCHIO ALTO. La comparazione implicita ("la Più Veloce sul Mercato") non è basata su parametri oggettivi e dimostrabili come richiesto dalle normative.\n	Raccomandazioni (Non Vincolanti):\n	Rimuovere i superlativi assoluti o supportarli con prove concrete (es. "Più veloce secondo il test XYZ di Settembre 2025").\n	Sostituire "efficienza straordinaria" con un beneficio quantificabile (es. "riduce i tempi di elaborazione del 40%").\n	Disclaimer: Questo è un controllo automatico e non costituisce una consulenza legale. Si raccomanda di consultare un professionista p\ner una valutazione definitiva.`,},
	{ 
	profile: "Revisore di Clausole per Termini di Servizio Semplificati", 
	title: "Plain Language Compliance: Il Mandato Legale per la Leggibilità dei Contratti", 
	context: "Una clausola in un contratto per consumatori è scritta in un linguaggio legalistico complesso, violando il requisito della Direttiva sull'Accessibilità UE che impone un livello di comprensibilità B2 (CEFR) o inferiore, rischiando la nullità della clausola stessa.", 
	input: "Ferme restando le disposizioni di cui all'Articolo IV.b, il Recipiente prende atto e accetta che qualsiasi controversia derivante o in relazione all'inadempienza contrattuale sarà risolta mediante arbitrato vincolante ai sensi del Regolamento Arbitrale della CCIA, con esclusione della giurisdizione ordinaria.", 
	output:`### Punteggio di Rischio Conformità: MEDIO\n	Risultati del Controllo:\n⚠️ Chiarezza Clausole Onerose: ATTENZIONE. La clausola che impone l'arbitrato vincolante è una clausola onerosa. Anche se legalmente valida, la sua formulazione è complessa.\n❌ Leggibilità (Plain Language): RISCHIO ALTO. Il testo utilizza un linguaggio arcaico ("Ferme restando") e un gergo legale ("inadempienza", "giurisdizione ordinaria") che non rispetta un livello di comprensibilità B2, come richiesto per i contratti con i consumatori.\n	Raccomandazioni (Non Vincolanti):\n	Riformulare la clausola in un linguaggio più semplice, ad esempio: "Qualsiasi disputa su questo contratto sarà risolta tramite un arbitrato e non in un tribunale ordinario."\n	Assicurarsi che le clausole onerose siano approvate specificamente per iscritto, come richiesto dall'Art. 1341 del Codice Civile.\n	Disclaimer: Questo è un controllo automatico e non costituisce una consulenza legale. Si raccomanda di consultare un professionista per una valutazione definitiva.`,},
	{ 
	profile: "Analizzatore di Green Claims (CSRD/Tassonomia UE)", 
	title: "Greenwashing Catastrofico: Verifica Scientifica e Rischio di Sanzioni", 
	context: "Un'azienda si dichiara 'Carbon Neutral' basandosi esclusivamente sull'acquisto di crediti di carbonio e definisce un suo prodotto '100% Eco-Friendly'. Entrambe sono pratiche considerate greenwashing secondo la Direttiva Green Claims, con sanzioni fino al 4-5% del fatturato.", 
	input: "La nostra Azienda è Orgogliosamente Carbon Neutral grazie all'acquisto di crediti di offset certificati. Questo prodotto è 100% Eco-Friendly.", 
	output:`### Punteggio di Rischio Conformità: ALTO	Risultati del Controllo:\n❌ Specificità dei Claim: RISCHIO ALTO. Il claim "100% Eco-Friendly" è generico, vago e non comprovato. È una delle pratiche esplicitamente vietate.\n⚠️ Supporto Scientifico: ATTENZIONE. Dichiararsi "Carbon Neutral" basandosi solo sull'offsetting, senza menzionare sforzi di riduzione diretta delle emissioni, è considerato fuorviante. Manca qualsiasi riferimento a metodologie o certificazioni verificabili.\n	Raccomandazioni (Non Vincolanti):\n	Rimuovere immediatamente il claim "100% Eco-Friendly".\n	Comunicare in modo trasparente la strategia di sostenibilità, dettagliando sia le azioni di riduzione diretta delle emissioni sia, separatamente, le attività di offsetting.\n	Disclaimer: Questo è un controllo automatico e non costituisce una consulenza legale. Si raccomanda di consultare un professionista per una valutazione definitiva.`,},
	{ 
	profile: "Revisore di Comunicazioni Mediche e Farmaceutiche", 
	title: "Chiarezza Clinica: Gergo Medico Inaccessibile per il Pubblico", 
	context: "La descrizione di una reazione avversa a un farmaco, destinata a un foglietto illustrativo per pazienti, utilizza un gergo clinico incomprensibile, violando le linee guida EMA sulla chiarezza e la sicurezza del paziente.", 
	input: "Reazione Avversa Rara: Osservata l'insorgenza di acromegalia con episodi transitori di assenza seizure durante la fase di titolazione del farmaco.", 
	output:`### Punteggio di Rischio Conformità: ALTO\n	Risultati del Controllo:\n❌ Correttezza Terminologica (Pubblico): RISCHIO ALTO. Il testo usa termini clinici specialistici ("acromegalia", "assenza seizure", "titolazione") non comprensibili per un paziente, contravvenendo alle linee guida EMA sulla chiarezza.\n✅ Claim di Efficacia: OK. Il testo non fa alcun claim sull'efficacia del farmaco, ma descrive solo un effetto collaterale.\n	Raccomandazioni (Non Vincolanti):\n	"Tradurre" ogni termine tecnico in un linguaggio semplice, come raccomandato dall'EMA. Esempio: "acromegalia" -> "eccessiva crescita delle ossa di mani, piedi e viso"; "assenza seizure" -> "brevi e improvvisi vuoti di coscienza".\n	Disclaimer: Questo è un controllo automatico e non costituisce una consulenza legale o medica. Si raccomanda di consultare un professionista per una valutazione definitiva.`,},
	{ 
	profile: "Checker di Accessibilità Testuale Digitale (WCAG)", 
	title: "Contrasto Critico: Violazione AA e Rischio di Esclusione Digitale", 
	context: "Le istruzioni di design per un sito web specificano di usare un link di colore blu chiaro su sfondo bianco e di rimuovere la sottolineatura, violando i criteri di contrasto minimo e di identificazione dei link secondo le WCAG 2.1.", 
	input: 'Istruzione al Content/Design Team: "Inserire il link di supporto tecnico in blu chiaro (#3366FF) su sfondo bianco (#FFFFFF). Non utilizzare la sottolineatura di default per mantenere il design pulito."', 
	output:`### Punteggio di Rischio Conformità: ALTO\n	Risultati del Controllo:\n❌ Indipendenza Sensoriale: RISCHIO ALTO. L'istruzione di usare solo il colore per distinguere un link, rimuovendo la sottolineatura, viola il criterio WCAG 1.4.1. Gli utenti con daltonismo potrebbero non essere in grado di identificare il link.\n⚠️ Contrasto Minimo: ATTENZIONE. Un colore blu chiaro (#3366FF) su sfondo bianco (#FFFFFF) ha un rapporto di contrasto di circa 2.8:1, che è inferiore al minimo richiesto di 4.5:1 dal livello AA delle WCAG per il testo normale.\n	Raccomandazioni (Non Vincolanti):\n	Mantenere sempre la sottolineatura per i link all'interno di un corpo di testo.\n	Utilizzare un colore per i link che garantisca un rapporto di contrasto di almeno 4.5:1 con lo sfondo.\n	Disclaimer: Questo è un controllo automatico e non costituisce una valutazione di conformità completa. Si raccomanda un'analisi tecnica e la consultazione di un esperto.`,},
	{ 
	profile: "Verificatore di Comunicazioni KYC/AML Anti-Frodi", 
	title: "Friction Compliance: Istruzioni di Due Diligence Ambigue", 
	context: "Una comunicazione a un cliente B2B ad alto rischio per la procedura di onboarding è vaga sui documenti necessari, creando attrito per il cliente e aumentando il rischio di una due diligence incompleta per l'azienda.", 
	input: "Per completare l'onboarding, potrebbe essere richiesta documentazione supplementare e accertamenti aggiuntivi in base alla nostra valutazione del rischio. Vi informeremo se i documenti che avete caricato sono sufficienti.", 
	output:`### Punteggio di Rischio Conformità: MEDIO\n	Risultati del Controllo:\n⚠️ Trasparenza Transazioni: ATTENZIONE. La comunicazione è reattiva ("Vi informeremo se...") invece che proattiva. Un linguaggio vago come "potrebbe essere richiesta documentazione supplementare" crea incertezza e ritardi.\n✅ Aderenza a Procedure Standard: OK. Il testo menziona correttamente una "valutazione del rischio" e "accertamenti aggiuntivi", che sono in linea con le procedure standard di Enhanced Due Diligence (EDD).\n	Raccomandazioni (Non Vincolanti):\n	Riformulare la comunicazione per essere proattiva e specifica. Esempio: "Dato che la vostra struttura societaria rientra nei profili ad alto rischio, vi chiediamo di fornire fin da ora i seguenti documenti aggiuntivi: [Elenco Documenti]."\n	Disclaimer: Questo è un controllo automatico e non sostituisce un sistema di monitoraggio AML completo. Si raccomanda di consultare un esperto di compliance.`,},
	{ 
	profile: "Analizzatore di Disclaimer e Condizioni d'Uso E-commerce", 
	title: "Il Bottone Sbagliato: Nullità del Contratto di Vendita Online", 
	context: "Il pulsante finale di un carrello e-commerce riporta la dicitura 'Prosegui alla Conferma'. Questa etichetta ambigua non esplicita l'obbligo di pagamento e, secondo la Direttiva sui Diritti dei Consumatori (CRD), può rendere il contratto nullo.", 
	input: 'Testo sul pulsante di acquisto finale nel funnel: "Prosegui alla Conferma dell\'Ordine"', 
	output:`### Punteggio di Rischio Conformità: ALTO\n	Risultati del Controllo:\n❌ Informativa Precontrattuale: RISCHIO ALTO. Il testo del pulsante "Prosegui alla Conferma dell'Ordine" non soddisfa il requisito della Direttiva CRD, che impone un'etichetta inequivocabile sull'obbligo di pagamento (es. "Acquista Ora" o "Paga Adesso").\n✅ Gestione Diritto di Recesso: OK. Il testo del pulsante non riguarda il diritto di recesso.\n✅ Chiarezza sulla Garanzia: OK. Il testo del pulsante non riguarda la garanzia.\n	Raccomandazioni (Non Vincolanti):\n	Sostituire immediatamente il testo del pulsante con una formulazione conforme, come "Acquista Ora", "Paga Adesso" o "Ordina con Obbligo di Pagare".\n	Disclaimer: Questo è un controllo automatico e non costituisce una consulenza legale. Si raccomanda di consultare un professionista per una valutazione definitiva.`,	}
];


export default function UseCasesPage() {
  const [activeTab, setActiveTab] = useState<'validator' | 'interpreter' | 'compliance'>('validator');
  
  type UseCase = (typeof validatorUseCases)[0] | (typeof interpreterUseCases)[0] | (typeof complianceUseCases)[0];

	const [selectedUseCase, setSelectedUseCase] = useState<UseCase | null>(
	  validatorUseCases.length > 0 ? validatorUseCases[0] : null
	)

  const handleTabChange = (tab: 'validator' | 'interpreter' | 'compliance') => {
	  setActiveTab(tab);
	  if (tab === 'validator' && validatorUseCases.length > 0) {
		setSelectedUseCase(validatorUseCases[0]); // Seleziona il primo elemento
	  } else if (tab === 'interpreter' && interpreterUseCases.length > 0) {
		setSelectedUseCase(interpreterUseCases[0]); // Seleziona il primo elemento
	  } else if (tab === 'compliance' && complianceUseCases.length > 0) {
		setSelectedUseCase(complianceUseCases[0]); // Seleziona il primo elemento
	  } else {
		setSelectedUseCase(null);
	  }
	};

  const currentUseCases = activeTab === 'validator' 
    ? validatorUseCases 
    : activeTab === 'interpreter' 
    ? interpreterUseCases 
    : complianceUseCases;

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-6xl">
        <header className="mb-12 text-center">
          <h1 className="text-5xl md:text-6xl font-extrabold text-blue-400 mb-4 leading-tight">
            Use Cases: Dalla Sfida al Successo
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Scopri come i nostri strumenti risolvono problemi reali, garantendo qualità e chiarezza in ogni analisi.
          </p>
        </header>

        {/* Selettore Tab */}
        <div className="mb-12 flex justify-center border-b border-gray-700">
          <button
            onClick={() => handleTabChange('validator')}
            className={`px-8 py-3 text-xl font-semibold transition-colors ${activeTab === 'validator' ? 'border-b-2 border-blue-500 text-white' : 'text-gray-400 hover:text-white'}`}
          >
            Validator
          </button>
          <button
            onClick={() => handleTabChange('interpreter')}
            className={`px-8 py-3 text-xl font-semibold transition-colors ${activeTab === 'interpreter' ? 'border-b-2 border-green-500 text-white' : 'text-gray-400 hover:text-white'}`}
          >
            Interpreter
          </button>
		  <button
			onClick={() => handleTabChange('compliance')}
			className={`px-8 py-3 text-xl font-semibold transition-colors ${activeTab === 'compliance' ? 'border-b-2 border-yellow-500 text-white' : 'text-gray-400 hover:text-white'}`}
          >
			Compliance Checkr
	      </button>
        </div>

        {/* Selettore dei Profili */}
        <section className="mb-12 py-8 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
          <h2 className="text-3xl font-bold text-center text-blue-300 mb-6">Seleziona un Profilo per Vedere un Esempio Pratico</h2>
          <div className="flex flex-wrap justify-center gap-4 px-4">
            {currentUseCases.map((uc, index) => (
              <button
                key={index}
                onClick={() => setSelectedUseCase(uc)}
                className={`px-6 py-3 rounded-full text-lg font-semibold transition-all shadow-md ${
					selectedUseCase?.profile === uc.profile
						? (activeTab === 'validator' ? 'bg-blue-600' : activeTab === 'interpreter' ? 'bg-green-600' : 'bg-yellow-600') + ' text-white transform scale-105'
						: 'bg-gray-700 text-gray-200 hover:bg-gray-600'
				}`}
              >
                {uc.profile}
              </button>
            ))}
          </div>
        </section>

        {/* Dettagli dello Use Case Selezionato */}
        {selectedUseCase && (
          <section className="mb-16 animate-fade-in">
            <h2 className="text-4xl font-bold text-blue-400 mb-6 text-center">
              {selectedUseCase.profile}: {selectedUseCase.title}
            </h2>
            <div className="bg-gray-800 p-8 rounded-xl shadow-lg border border-gray-700">
              <h3 className="text-2xl font-semibold text-blue-300 mb-4">Contesto e Problema (Challenge):</h3>
              <p className="text-gray-300 mb-8 italic">{selectedUseCase.context}</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8 md:flex">
			  {/* Box di Input */}
			  <div className="md:w-1/2 flex flex-col">
				<h3 className="text-2xl font-semibold text-white mb-3">Input Grezzo (Prima)</h3>
				<textarea
				  rows={12}
				  readOnly
				  value={selectedUseCase.input}
				  className="w-full flex-grow bg-gray-900 text-gray-200 p-4 rounded-lg border border-gray-700 resize-none shadow-inner"
				></textarea>
			  </div>
			  
			  {/* Box di Output */}
			  <div className="md:w-1/2 flex flex-col">
				<h3 className="text-2xl font-semibold text-white mb-3">Output Analizzato (Dopo)</h3>
				<div className="w-full flex-grow rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-inner overflow-y-auto prose prose-invert prose-p:text-gray-300 prose-strong:text-white prose-headings:text-blue-400">
				  <ReactMarkdown>
					  {selectedUseCase.output}
				  </ReactMarkdown>
				</div>
			  </div>
			</div>

              {/* Report di Qualità (mostrato solo se la tab non è 'compliance') */}
				{activeTab !== 'compliance' && selectedUseCase && 'qualityScore' in selectedUseCase && (
					<div className="mt-8 p-6 rounded-xl bg-gray-900 border border-gray-700 shadow-md">
						<h3 className="text-2xl font-semibold text-green-400 mb-4 text-center">Controllo di Qualità AI²™</h3>
						<div className="flex items-center justify-center text-center mb-4">
							<p className="text-6xl font-bold text-green-400">{selectedUseCase.qualityScore}</p>
							<p className="ml-2 text-2xl text-gray-400">/ 100</p>
						</div>
						<p className="text-gray-300 italic text-center">
							{selectedUseCase.reasoning}
						</p>
					</div>
				)}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

// Aggiungi questa animazione al tuo file globals.css se non c'è già
/*
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in {
  animation: fadeIn 0.5s ease-out forwards;
}
*/