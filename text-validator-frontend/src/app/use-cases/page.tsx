// src/app/use-cases/page.tsx
"use client";

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const validatorUseCases = [
    {
      profile: "Scrittore di Proposte Commerciali",
      title: "XXXXX",
      context: "Marco, titolare di una piccola agenzia di software, deve inviare una proposta a un potenziale cliente, Rossi Cucine, per la realizzazione di un nuovo sistema di gestione magazzino. Ha raccolto degli appunti disordinati dopo una telefonata e deve trasformarli in una proposta professionale e persuasiva.",
      input: "Cliente: Rossi Cucine, azienda di medie dimensioni. Problema: il loro sistema attuale è vecchio, basato su Excel, lento. Perdono tempo, fanno errori di inventario. Hanno bisogno di qualcosa di moderno. La nostra soluzione: software custom basato su cloud. Accessibile da ovunque. Funzioni da includere: gestione anagrafica prodotti, carico/scarico magazzino con lettore barcode, reportistica inventario in tempo reale, alert per scorte minime. Costi: sviluppo circa 15.000 euro. Poi canone di mantenimento di 250 euro/mese per hosting, backup e supporto. Tempistiche: 8 settimane per lo sviluppo, poi una settimana di formazione in loco. Perché noi: siamo specializzati in soluzioni per PMI, abbiamo già fatto un lavoro simile per Bianchi Arredamenti. Dobbiamo convincerli che è un investimento, non un costo. Parlare del ROI: risparmio di tempo per i magazzinieri (stimato 20%), riduzione errori di inventario (-90%). Prossimi passi: inviare la proposta e fissare una call per la demo.",      
	  output: `# PROPOSTA COMMERCIALE: Ottimizzazione Gestione Magazzino e Inventario per Rossi Cucine

**Data:** [Inserire Data]
**A:** Rossi Cucine
**Oggetto:** Proposta per la Digitalizzazione del Sistema Gestionale e il Miglioramento dell'Efficienza Operativa

## 1. Introduzione e Comprensione del Problema

Comprendiamo perfettamente che Rossi Cucine, operando in un mercato dinamico come quello dell'arredamento, necessiti di strumenti agili per mantenere la propria competitività.

Abbiamo analizzato le sfide operative attuali, in particolare:
*   **Sistema Datato e Inefficiente:** L'affidamento a strumenti legacy basati su fogli di calcolo (Excel) sta rallentando i processi quotidiani.
*   **Rischio di Errore Umano:** La gestione manuale dell'inventario comporta un elevato rischio di discrepanze tra stock fisico e registrato.
*   **Perdita di Produttività:** Il tempo speso per aggiornamenti manuali e la ricerca di dati sottrae risorse preziose al personale operativo.

L'obiettivo primario è sostituire questo ambiente frammentato con una **piattaforma centralizzata, moderna e accessibile** che garantisca precisione e velocità nelle operazioni di magazzino.

## 2. La Nostra Soluzione Proposta: Piattaforma Gestionale Cloud Custom

Proponiamo lo sviluppo di un **Software Gestionale Customizzato, basato su tecnologia Cloud**, progettato specificamente per le esigenze di magazzino e inventario di Rossi Cucine. Questa soluzione sarà accessibile in qualsiasi momento e da qualsiasi dispositivo connesso, garantendo flessibilità operativa.

### Benefici Chiave Attesi (ROI Immediato)

L'adozione di questo sistema si traduce direttamente in un ritorno sull'investimento misurabile:

*   **Aumento dell'Efficienza:** Stimiamo un **risparmio di tempo operativo pari al 20%** per il personale di magazzino grazie alla digitalizzazione dei flussi di lavoro.
*   **Accuratezza dell'Inventario:** Prevediamo una **riduzione degli errori di inventario superiore al 90%**, eliminando le discrepanze critiche.
*   **Accessibilità Totale:** Gestione delle operazioni da remoto grazie all'architettura Cloud.
*   **Decisioni Basate sui Dati:** Disponibilità immediata di dati affidabili per la pianificazione degli acquisti e della produzione.

### Funzionalità Principali Incluse

Il sistema sarà dotato delle seguenti funzionalità essenziali per il vostro contesto:

1.  **Anagrafica Prodotti Centralizzata:** Gestione completa e strutturata di tutti i codici e le specifiche dei prodotti.
2.  **Gestione Movimenti Magazzino Ottimizzata:** Processi di carico e scarico integrati e velocizzati tramite l'utilizzo di **lettori barcode**.
3.  **Inventario in Tempo Reale:** Dashboard e reportistica aggiornata istantaneamente sulla situazione delle scorte.
4.  **Sistema di Allerta Proattivo:** Notifiche automatiche per il raggiungimento delle **scorte minime** (Reorder Point).

## 3. Dettagli dell'Offerta e Piano di Lavoro

La nostra metodologia è focalizzata su una consegna rapida e un'implementazione mirata, garantendo che siate operativi nel minor tempo possibile.

| Fase | Attività Principali | Durata Stimata |
| :--- | :--- | :--- |
| **Fase 1: Analisi e Design** | Definizione dettagliata delle specifiche funzionali e UI/UX. | 1 Settimana |
| **Fase 2: Sviluppo Core** | Implementazione del database, moduli anagrafica e gestione barcode. | 5 Settimane |
| **Fase 3: Testing e Deployment** | Test integrati, migrazione dati (se necessario) e messa in produzione. | 2 Settimane |
| **Fase 4: Formazione e Go-Live** | Formazione intensiva del personale in loco e supporto immediato. | 1 Settimana |
| **Tempo Totale Stimato:** | | **8 Settimane di Sviluppo + 1 Settimana di Formazione** |

## 4. Investimento Proposto

L'investimento è strutturato in due componenti: un costo iniziale per lo sviluppo del software custom e un canone ricorrente per la sua manutenzione e il supporto operativo.

| Descrizione Costo | Dettaglio | Importo (IVA Esclusa) |
| :--- | :--- | :--- |
| **Sviluppo Soluzione Custom** | Progettazione, sviluppo, integrazione delle funzionalità descritte e implementazione iniziale. | **€ 15.000,00** (Una Tantum) |
| **Canone di Mantenimento e Supporto** | Hosting su infrastruttura Cloud sicura, backup giornalieri, aggiornamenti di sicurezza e supporto tecnico H24/7. | **€ 250,00 / Mese** (Ricorrente) |

**Nota sull'Investimento:** Questo progetto rappresenta un investimento strategico mirato a trasformare un centro di costo (la gestione inefficiente degli errori e del tempo) in un fattore di vantaggio competitivo.

## 5. Perché Scegliere Noi e Prossimi Passi

Siamo partner specializzati nell'offrire soluzioni tecnologiche scalabili e concrete per le Piccole e Medie Imprese. La nostra esperienza pregressa, inclusi progetti di successo nel settore dell'arredamento (come dimostrato dal lavoro svolto per **Bianchi Arredamenti**), assicura che comprendiamo le vostre specificità operative e le sfide logistiche uniche.

Il nostro approccio è focalizzato sul **risultato tangibile**, assicurando che il software non sia solo installato, ma effettivamente adottato per generare il ROI promesso.

### Prossimi Passi Consigliati

Per procedere e visualizzare concretamente come il sistema funzionerà all'interno del vostro contesto:

1.  **Accettazione della Proposta:** Conferma formale del nostro accordo.
2.  **Pianificazione Kick-Off:** Definizione del calendario di avvio e pianificazione della prima riunione di analisi dettagliata (Fase 1).
3.  **Sessione di Demo Personalizzata:** Vi invitiamo a fissare una breve call di approfondimento dove potremo mostrarvi un prototipo interattivo delle funzionalità chiave.

Siamo a vostra completa disposizione per discutere questa proposta e rispondere a qualsiasi domanda. Attendiamo il vostro riscontro per iniziare il percorso verso l'efficienza operativa.`,
qualityScore: 98,
reasoning: `La trasformazione è eccellente e trasforma un riassunto di appunti tecnici in una proposta commerciale professionale, persuasiva e ben strutturata. Ha pienamente soddisfatto tutti i criteri richiesti. 1. **Focalizzazione sul Cliente**: Ottima. Il testo inizia immediatamente riconoscendo i problemi del cliente (sistema datato, rischio errori, perdita di produttività) e lega le funzionalità ai benefici misurabili (ROI al 20% di risparmio di tempo, -90% errori), trasformando le caratteristiche in valore per Rossi Cucine. 2. **Struttura Persuasiva**: La struttura è impeccabile: introduzione/problema -> soluzione/benefici (ROI) -> funzionalità -> piano di lavoro -> costi -> prova sociale/next steps. È un percorso logico che guida il lettore. 3. **Chiarezza dell'Offerta e dei Costi**: Molto chiara. I costi sono tabellati in modo trasparente (una tantum vs ricorrente) e le tempistiche sono suddivise in fasi logiche con una chiara distinzione tra sviluppo e formazione. 4. **Professionalità e Call-to-Action**: Il tono è estremamente professionale. La sezione 'Prossimi Passi' è chiara, specifica le azioni richieste (Accettazione, Pianificazione, Demo) e fornisce un invito all'azione diretto.`
    },
	{
      profile: "Comunicatore di Crisi PR",
      title: "XXXXX",
      context: "VerdeBio, una PMI che produce alimenti biologici, ha scoperto una contaminazione accidentale da un allergene non dichiarato (arachidi) in un lotto di barrette energetiche. Devono scrivere un comunicato stampa urgente per informare i clienti e gestire la crisi reputazionale.",
      input: "URGENTE: Dobbiamo comunicare subito il problema con le barrette Energia Pura. Lotto #24-A58. C'è stata una contaminazione incrociata con arachidi. Non è scritto sull'etichetta. È un disastro. Dobbiamo ritirare il prodotto. Scrivi qualcosa da mettere sul sito e da mandare ai giornali. Dobbiamo dire che ci dispiace molto, è la prima volta che succede. La sicurezza dei clienti è la nostra priorità. Chi ha comprato il lotto può riportarlo in negozio per un rimborso completo o una sostituzione. Abbiamo già bloccato la produzione e stiamo investigando internamente per capire come sia potuto succedere. Non usiamo un tono allarmistico ma dobbiamo essere chiari. Metti un numero di telefono del servizio clienti per chi ha domande. Speriamo che la gente capisca che siamo un'azienda seria e che è stato un incidente isolato.",
      output: `Siamo profondamente dispiaciuti e vogliamo affrontare immediatamente la preoccupazione dei nostri consumatori riguardo al prodotto "Energia Pura". La sicurezza e la salute dei nostri clienti sono la nostra priorità assoluta.

Abbiamo identificato un potenziale problema di contaminazione incrociata con tracce di arachidi in un lotto specifico di barrette "Energia Pura", identificato con il codice Lotto #24-A58. Queste tracce non erano dichiarate sull'etichettatura del prodotto. Riconosciamo la gravità di questa omissione e ci scusiamo incondizionatamente per il rischio che questo può aver comportato per i consumatori, in particolare quelli con allergie.

**Azioni Immediate Intraprese:**

1.  **Richiamo Volontario del Prodotto:** Abbiamo immediatamente avviato il ritiro volontario da tutti i canali di distribuzione del Lotto #24-A58.
2.  **Sospensione Produzione:** Abbiamo sospeso la produzione per riesaminare e rafforzare rigorosamente i nostri protocolli di sicurezza e prevenzione della contaminazione incrociata.
3.  **Indagine Approfondita:** È stata avviata un'indagine interna completa per comprendere l'origine esatta di questa anomalia e assicurare che vengano prese misure definitive per evitare che situazioni simili si ripetano in futuro.

**Supporto ai Consumatori:**

Invitiamo tutti i consumatori che hanno acquistato barrette "Energia Pura" appartenenti al Lotto #24-A58 a non consumarle. È possibile restituire il prodotto presso il punto vendita per ricevere un rimborso completo o una sostituzione con un prodotto alternativo.

Per qualsiasi domanda, chiarimento o supporto immediato, è attiva una linea dedicata del nostro Servizio Clienti al numero [INSERIRE NUMERO DI TELEFONO].

Siamo un'azienda impegnata nella qualità e nella fiducia dei nostri clienti. Stiamo gestendo questa situazione con la massima serietà e trasparenza, e ci impegniamo a ripristinare la fiducia attraverso azioni concrete e rigorose revisioni operative.

Per aggiornamenti ufficiali sulla risoluzione della situazione, vi preghiamo di fare riferimento esclusivamente al nostro sito web aziendale.`,
qualityScore: 95,
reasoning: `La trasformazione dal testo originale (un insieme di istruzioni e appunti interni) al testo revisionato è eccellente e trasforma efficacemente la confusione interna in una comunicazione di crisi pubblica professionale. Il testo revisionato eccelle nell'Empatia e Tono, usando espressioni come "profondamente dispiaciuti" e "priorità assoluta". La Trasparenza è massima, chiarendo esattamente il problema (contaminazione da arachidi, omissione sull'etichetta) e il lotto specifico. L'Orientamento all'Azione è chiarissimo, strutturato in punti numerati (richiamo, sospensione produzione, indagine). Infine, l'Efficacia Strategica è elevata: riconosce l'errore, si scusa incondizionatamente, fornisce canali di supporto chiari e riafferma l'impegno per la fiducia, mitigando così i danni reputazionali meglio di quanto non potesse fare il messaggio originale.`
    },
	{
      profile: "Redattore Termini e Condizioni E-commerce",
      title: "XXXXX",
      context: "Giulia ha appena lanciato il suo e-commerce di gioielli artigianali, Gioie di Luna. Ha bisogno di una bozza dei Termini e Condizioni di vendita da inserire sul sito, ma ha solo un elenco di punti che vuole coprire.",
      input: "Nome azienda: Gioie di Luna di Giulia Bianchi, P.IVA 1234567890. Sede: Via Roma 1, Milano. I prodotti sono fatti a mano, quindi possono avere piccole differenze rispetto alle foto. Pagamenti: accettiamo carte di credito, PayPal. Spedizione: spediamo in tutta Italia con corriere espresso. Costa 7 euro, gratis sopra i 100 euro di spesa. Tempi di consegna: 3-5 giorni lavorativi. Resi: i clienti possono rendere i prodotti entro 14 giorni dalla consegna. Il prodotto deve essere integro, non usato, nella sua scatola originale. Le spese di spedizione per il reso sono a carico del cliente. Rimborsiamo i soldi entro 10 giorni da quando riceviamo il reso. Per i prodotti personalizzati non accettiamo resi. Privacy: i dati dei clienti li usiamo solo per gestire l'ordine, non li diamo a nessuno. Linkare alla privacy policy. Legge applicabile: quella italiana. Foro di Milano.",
      output: "",
      qualityScore: "01",
      reasoning: ""
    },
	{
      profile: "Scrittore Testi per Landing Page",
      title: "XXXXX",
      context: "Una startup fintech, PayFast, sta lanciando un nuovo servizio di POS mobile per piccoli commercianti. Devono creare il testo per una landing page che spinga i visitatori a richiedere una demo gratuita.",
      input: "Titolo: qualcosa tipo Accetta pagamenti ovunque con PayFast. Target: piccoli negozianti, artigiani, liberi professionisti. Prodotto: un piccolo lettore di carte che si collega allo smartphone via Bluetooth. Problema che risolviamo: i POS tradizionali sono costosi, hanno canoni fissi e sono ingombranti. La nostra soluzione: nessun canone fisso, paghi solo una piccola commissione (1.5%) quando incassi. Benefici: economico, portatile, facile da usare. Come funziona: ordini il lettore, scarichi l'app, lo colleghi e inizi a incassare. Prova sociale: mettere una citazione di un cliente beta, tipo Finalmente un POS semplice e senza costi fissi! - Marco, Fioraio. Offerta: demo gratuita del servizio. CTA: Richiedi la tua Demo Gratuita. Dobbiamo far capire che è la soluzione perfetta per chi incassa in mobilità o ha volumi bassi. Menzionare che accettiamo tutte le carte: Visa, Mastercard, Amex.",
      output: `# Headline Principale
**Dimentica Canoni Fissi: Accetta Pagamenti con Carta Ovunque, Pagando Solo Quando Incassi!**

# Sottotitolo
Il lettore di carte Bluetooth portatile che trasforma il tuo smartphone nel tuo POS definitivo. Ideale per artigiani, professionisti e piccoli negozi con bassi volumi.

# Introduzione (Il Problema)
Sei stanco di POS ingombranti, contratti vincolanti e canoni fissi mensili che prosciugano i tuoi guadagni anche nei giorni "morti"? Sai che i metodi tradizionali ti costringono a pagare, indipendentemente dal fatto che tu abbia venduto qualcosa o meno? Per chi lavora in mobilità o gestisce un piccolo volume, questa struttura è semplicemente insostenibile.

# La Soluzione
Ti presentiamo **PayFast**, il sistema di pagamento mobile creato su misura per chi vuole flessibilità e trasparenza. Collega il lettore al tuo smartphone e inizia ad accettare pagamenti in pochi secondi, senza sorprese in fattura.

*   **Costi Zero Fissi:** Nessun canone mensile, nessun costo di attivazione nascosto.
*   **Commissione Trasparente:** Paghi solo l'1.5% su ogni singola transazione incassata.
*   **Portabilità Massima:** Piccolo, leggero e si collega via Bluetooth. Porta il tuo business dove sono i tuoi clienti.
*   **Accetta Tutto:** Supportiamo tutte le principali carte di credito e debito: Visa, Mastercard e American Express.
*   **Configurazione Lampo:** Ordini il dispositivo, scarichi l'app gratuita, colleghi e sei operativo in 5 minuti.

# Prova Sociale
"Finalmente un POS semplice, economico e senza costi fissi! Da quando uso PayFast ho incrementato le vendite perché i clienti pagano più volentieri. Un risparmio enorme per la mia piccola attività." - Marco Rossi, Titolare "Fiori di Marco".

# Call-to-Action (CTA) Finale
**Smetti di pagare per non vendere. Inizia oggi a usare solo ciò che guadagni!**

**Richiedi la Tua Demo Gratuita di PayFast Ora**`,
qualityScore: 95,
reasoning: `La trasformazione è eccellente. La Headline è molto più forte, focalizzandosi immediatamente sul beneficio chiave (zero canoni fissi) e sulla soluzione (pagare solo quando incassi). Il testo revisionato segue perfettamente un flusso persuasivo logico (Problema -> Soluzione -> Benefici/Caratteristiche -> Prova Sociale -> CTA). Vengono inclusi tutti gli elementi richiesti: la chiarezza sulla commissione dell'1.5%, l'accettazione di tutte le carte (Visa, MC, Amex) e la CTA è chiara e ben posizionata. La struttura per la conversione è completa e ben ottimizzata per il target di piccoli commercianti e professionisti mobili.`
    },
	{
      profile: "Redattore di Annunci di Lavoro",
      title: "XXXXX",
      context: "DigitalMinds, una piccola agenzia di marketing, sta cercando di assumere un Social Media Manager. L'HR manager, che è anche il responsabile amministrativo, ha buttato giù una bozza di annuncio da pubblicare su LinkedIn.",
      input: "Titolo: Cercasi Social Media Manager Siamo un'agenzia di marketing in crescita a Bologna. Cerchiamo una persona che gestisca i social per i nostri clienti. Deve saper creare contenuti (testi, grafiche base, video brevi). Deve pianificare il calendario editoriale. Deve gestire le campagne pubblicitarie su Meta (Facebook/Instagram) e LinkedIn. Richiesta esperienza di almeno 2-3 anni in un ruolo simile. Deve conoscere bene i tool di social media management (es. Hootsuite, Buffer). Deve saper analizzare i dati e fare reportistica. Offriamo un ambiente giovane e dinamico. Contratto a tempo indeterminato, RAL da definire in base all'esperienza. Mandare CV a info@digitalminds.it. Bello se conosce anche TikTok e Google Analytics.",
      output: `**Titolo del Lavoro**
Specialista in Strategia e Gestione dei Social Media

**Chi Siamo**
Siamo un’agenzia di marketing digitale in forte espansione con sede a Bologna, guidata dalla passione per la creazione di strategie di comunicazione efficaci. Crediamo nel potenziale della collaborazione, dell'innovazione e di un ambiente di lavoro giovane e stimolante, dove ogni contributo individuale è valorizzato per il successo collettivo.

**Le Tue Responsabilità**
*   Sviluppare e implementare strategie di social media efficaci per una varietà di clienti in diversi settori.
*   Pianificare, creare e curare contenuti originali (testi persuasivi, grafiche di base e brevi formati video) ottimizzati per ciascuna piattaforma.
*   Gestire e mantenere aggiornati i calendari editoriali.
*   Ideare, lanciare e monitorare campagne pubblicitarie a pagamento (Paid Social) sulle piattaforme Meta (Facebook/Instagram) e LinkedIn.
*   Analizzare le performance delle attività sui social media, estrarre insight significativi e produrre reportistica chiara per i clienti.
*   Mantenersi costantemente aggiornati sulle ultime tendenze e aggiornamenti degli algoritmi delle piattaforme social.

**Requisiti Fondamentali**
*   Esperienza comprovata di almeno 2-3 anni nella gestione professionale di canali social media o in un ruolo equivalente.
*   Padronanza nell'utilizzo di tool di gestione e scheduling per i social media (es. Hootsuite, Buffer o equivalenti).
*   Capacità dimostrata di ideazione e produzione di contenuti creativi (copywriting e basi di grafica/video editing).
*   Solida competenza nella gestione e nell'ottimizzazione di campagne advertising su Meta e LinkedIn.
*   Ottime capacità analitiche e di reportistica per misurare il ROI delle attività.

**Competenze Apprezzate**
*   Esperienza nella gestione e nella strategia di contenuti per la piattaforma TikTok.
*   Familiarità con i principi e gli strumenti di Google Analytics per la misurazione del traffico web derivante dai social.
*   Conoscenza di software di editing video o grafici più avanzati.

**Cosa Offriamo**
*   Contratto a tempo indeterminato, per una collaborazione stabile e a lungo termine.
*   Retribuzione competitiva, commisurata all'effettiva esperienza e competenza del candidato/a (RAL da definire in fase di colloquio).
*   Un ambiente di lavoro dinamico, collaborativo e orientato alla crescita professionale continua.
*   Opportunità reali di sviluppo di carriera all'interno dell'agenzia.
*   Flessibilità e supporto per la formazione continua.`,
qualityScore: 96,
reasoning: `La trasformazione da 'TESTO ORIGINALE' a 'TESTO REVISIONATO' è estremamente efficace. Il titolo è stato elevato da generico a professionale ('Specialista in Strategia e Gestione dei Social Media'). La struttura è stata completamente riorganizzata utilizzando intestazioni chiare (Chi Siamo, Responsabilità, Requisiti, Offerta), rendendo l'annuncio facilmente scansionabile. La Chiarezza del Ruolo è notevolmente migliorata grazie all'uso di punti elenco e verbi d'azione mirati (es. 'Sviluppare e implementare strategie'). L'Attrattività è potenziata dalla sezione 'Chi Siamo' che comunica valori aziendali (passione, collaborazione, innovazione) e dalla sezione 'Cosa Offriamo' che rende espliciti i benefici (stabilità, crescita). Infine, l'inclusività è migliorata sostituendo la terminologia secca con formulazioni più neutre e aggiungendo '/a' dove appropriato (es. candidato/a) o utilizzando il linguaggio professionale neutro (es. 'Specialista').`
    }
];

const interpreterUseCases = [
    {
      profile: "Revisore Contratto di Acquisto",
      title: "XXXX",
      context: "L'ufficio acquisti di una PMI manifatturiera, MetalTech, ha ricevuto un contratto di fornitura standard da un nuovo fornitore di componenti in acciaio. L'imprenditore vuole una sintesi rapida dei punti chiave e dei potenziali rischi prima di passarlo all'avvocato",
      input: "CONTRATTO DI FORNITURA Tra: Acciaio S.p.A. (Fornitore) e MetalTech S.r.l. (Acquirente). Oggetto: Fornitura di laminati in acciaio tipo S235JR secondo le specifiche dell'Allegato A. Durata: Il presente contratto ha durata di 3 anni dalla data di firma e si rinnoverà tacitamente per ulteriori periodi di 3 anni, salvo disdetta da inviarsi tramite PEC almeno 6 mesi prima della scadenza. Prezzi: I prezzi sono quelli indicati nel listino prezzi del Fornitore, che potrà essere aggiornato unilateralmente con preavviso di 30 giorni. Pagamenti: L'Acquirente si impegna a pagare le fatture a 60 giorni data fattura fine mese. In caso di ritardo, saranno applicati interessi di mora ai sensi del D.Lgs. 231/2002. Consegna: La consegna avverrà entro 15 giorni lavorativi dall'ordine. Eventuali ritardi non daranno diritto a penali se inferiori a 30 giorni. Garanzia: Il Fornitore garantisce che i prodotti sono esenti da vizi per 6 mesi dalla consegna. L'Acquirente deve denunciare i vizi entro 8 giorni dalla scoperta. Limitazione di responsabilità: La responsabilità del Fornitore per eventuali danni non potrà in ogni caso superare il valore della singola fornitura contestata. Esclusiva: L'Acquirente si impegna ad acquistare dal Fornitore in via esclusiva tutti i laminati di tipo S235JR per l'intera durata del contratto.",
      output: "",
      qualityScore: "01",
      reasoning: ""
    },
	{
      profile: "Analista Bilancio Aziendale",
      title: "XXXX",
      context: "L'amministratore di una piccola azienda commerciale riceve dal commercialista il bilancio annuale in un formato testuale complesso. Non essendo un esperto finanziario, ha bisogno di una sintesi chiara che gli spieghi lo stato di salute dell'azienda.",
      input: "Bilancio al 31/12/2024 - CommercioFacile Srl STATO PATRIMONIALE ATTIVO: Immobilizzazioni: 150.000. Rimanenze di magazzino: 80.000. Crediti verso clienti: 120.000. Liquidità immediate: 30.000. Totale Attivo: 380.000. PASSIVO: Patrimonio Netto: 100.000 (Capitale Sociale 20.000, Utile d'esercizio 15.000, Riserve 65.000). Debiti verso banche (a lungo termine): 130.000. Debiti verso fornitori (a breve termine): 90.000. Altri debiti a breve: 60.000. Totale Passivo e Netto: 380.000. CONTO ECONOMICO Valore della Produzione (Ricavi): 500.000. Costi della Produzione: Costi per materie prime: 250.000. Costi per servizi: 80.000. Costi del personale: 100.000. Ammortamenti: 20.000. Totale Costi: 450.000. Differenza tra Valore e Costi (EBIT): 50.000. Proventi e Oneri Finanziari: Interessi passivi: -25.000. Risultato prima delle imposte: 25.000. Imposte sul reddito: -10.000. Utile Netto d'esercizio: 15.000.",
      output: "",
      qualityScore: "01",
      reasoning: ""
    },
	{
      profile: "Analista di Capitolati di Gara e Bandi",
      title: "XXXX",
      context: "Una startup innovativa nel settore agritech, AgriSmart, vuole partecipare a un bando regionale per la digitalizzazione. Il CEO ha trovato il testo del bando e ha bisogno di estrarre rapidamente le informazioni chiave per decidere se procedere con la candidatura.",
      input: "AVVISO PUBBLICO DIGITALE PMI LAZIO 2026 - REGIONE LAZIO Art. 1 - Oggetto: Il presente bando, gestito da Lazio Innova, intende sostenere progetti di innovazione digitale e tecnologica delle PMI del territorio. La dotazione finanziaria è di 5 milioni di euro. Art. 2 - Soggetti Beneficiari: Possono presentare domanda le micro, piccole e medie imprese (PMI) che abbiano sede operativa nel Lazio e siano regolarmente iscritte alla CCIAA. Sono escluse le imprese in difficoltà. Art. 3 - Spese Ammissibili: Sono ammissibili le spese per l'acquisto di software, hardware e servizi di consulenza specialistica finalizzati all'introduzione di tecnologie 4.0. Le spese devono essere sostenute dopo la data di presentazione della domanda. Non sono ammissibili le spese di funzionamento. Art. 4 - Agevolazione: L'agevolazione è concessa nella forma di contributo a fondo perduto, pari al 50% delle spese ammissibili, fino a un massimo di 50.000 euro per progetto. Art. 5 - Criteri di Valutazione: I progetti saranno valutati sulla base dei seguenti criteri: carattere innovativo del progetto (40 punti), impatto economico e occupazionale (30 punti), sostenibilità del progetto (20 punti), qualità del team proponente (10 punti). Art. 6 - Modalità e Termini: Le domande devono essere presentate esclusivamente tramite la piattaforma GeCoWEB Plus a partire dalle ore 12:00 del 15 novembre 2025 ed entro le ore 17:00 del 15 dicembre 2025. Art. 7 - Documentazione: Alla domanda dovranno essere allegati: business plan del progetto, preventivi di spesa dettagliati, visura camerale aggiornata e dichiarazione sugli aiuti de minimis.",
      output: "",
      qualityScore: "01",
      reasoning: ""
    },
	{
      profile: "Analizzatore di Feedback dei Clienti",
      title: "XXXX",
      context: "Il product manager di un software SaaS per la gestione di progetti, TaskFlow, ha raccolto una serie di recensioni e feedback da email e sondaggi. Vuole analizzarli per capire i punti di forza e di debolezza del prodotto e decidere le prossime priorità di sviluppo.",
      input: "Recensione 1 (Sondaggio): Adoro la semplicità dell'interfaccia, è molto intuitiva. Però il prezzo mi sembra un po' alto rispetto alla concorrenza. E mi piacerebbe tanto avere un'integrazione con Google Calendar. Email 1 (Supporto): Buongiorno, il vostro software è fantastico per organizzare i task, ma l'app mobile è lentissima e crasha spesso. È quasi inutilizzabile. Potete risolvere? Recensione 2 (Sito di review): TaskFlow ha cambiato il modo in cui il mio team lavora. La vista Kanban è perfetta. Il supporto clienti è stato super veloce e disponibile quando ho avuto un problema. Cinque stelle! Recensione 3 (Sondaggio): Buon prodotto, ma mancano funzionalità avanzate. Non posso creare dipendenze tra i task, e la reportistica è troppo basilare per le mie esigenze. Sto valutando di passare a un altro strumento. Email 2 (Feedback): Sarebbe possibile aggiungere una modalità scura? Lavoro spesso di sera e l'interfaccia bianca mi affatica la vista. Per il resto, ottimo lavoro! Recensione 4 (Sito di review): L'ho provato ma l'ho trovato confusionario. L'onboarding iniziale non mi ha aiutato a capire come impostare il mio primo progetto. Inoltre, non si integra con Slack, che per noi è fondamentale. Email 3 (Supporto): Il vostro supporto è eccezionale, mi avete risolto un problema in 10 minuti. Grazie mille!",
      output: "",
      qualityScore: "01",
      reasoning: ""
    },
	{
      profile: "Sintetizzatore di Meeting e Trascrizion",
      title: "XXXX",
      context: "Dopo una riunione strategica sul lancio di un nuovo prodotto, il project manager ha la trascrizione della call. Ha bisogno di un riassunto operativo da condividere con il team, con decisioni chiare e compiti assegnati.",
      input: "Ok team, parliamo del lancio del nuovo 'Prodotto X'. Allora, la data di lancio. Io direi fine novembre. Che ne pensate?' 'Anna: Fine novembre è rischioso con il Black Friday. Forse meglio la seconda settimana di gennaio? Così abbiamo tempo per preparare meglio la campagna.' 'Marco: Concordo con Anna. Gennaio è più tranquillo. Quindi decidiamo per il 15 gennaio 2026.' 'Ok, deciso per il 15 gennaio. Ora, la campagna marketing. Anna, tu e il tuo team dovete preparare il piano di comunicazione. Mi serve una bozza entro fine mese, diciamo il 30 ottobre.' 'Anna: Ricevuto. Lavoreremo su social, newsletter e un paio di articoli sul blog.' 'Perfetto. Luca, per la parte tecnica. Il sito è pronto? La pagina prodotto?' 'Luca: La pagina è quasi pronta. Manca il video tutorial. Ho bisogno che Marco mi dia il girato entro la prossima settimana, altrimenti non ce la faccio.' 'Marco: Ok, ti mando il video entro venerdì prossimo. Poi, per il prezzo. Io pensavo a 49 euro. È in linea con i competitor.' 'Anna: Forse potremmo fare un'offerta di lancio a 39 euro per il primo mese per incentivare le vendite?' 'Ottima idea. Facciamo così. Prezzo di lancio 39 euro. Luca, aggiorna la pagina di conseguenza. C'è altro? Ah, sì, il supporto. Dobbiamo preparare le FAQ. Sara, te ne occupi tu? Scadenza fine novembre.' 'Sara: Certo, nessun problema. Preparo una bozza.' 'Bene, mi sembra tutto. Ricapitolando: lancio 15 gennaio, piano marketing da Anna entro 30/10, video da Marco a Luca entro settimana prossima, FAQ da Sara per fine novembre. Ottimo lavoro.",
      output: "",
      qualityScore: "01",
      reasoning: ""
    },
];

const complianceUseCases = [
	{
	profile: "Validatore Formale Domanda di Bando",
	title: "xxxx",
	context: "Una PMI sta per inviare la domanda per un bando Invitalia. L'amministratore ha preparato una descrizione riassuntiva di tutti i documenti che sta per allegare e vuole un ultimo controllo per assicurarsi di non aver dimenticato nulla di formale e cruciale.",
	input: "Stiamo per inviare la domanda per il bando 'Imprese ON'. Abbiamo compilato tutto il modulo online. Alleghiamo il nostro business plan dettagliato, che include le proiezioni finanziarie per i prossimi 3 anni. Ci sono anche i preventivi per i macchinari che vogliamo acquistare, firmati dai fornitori. Ho firmato digitalmente la dichiarazione che attesta che siamo una PMI e che non siamo un'impresa in difficoltà. Ho anche allegato la mia carta d'identità come legale rappresentante. La visura camerale è stata scaricata ieri, quindi è aggiornata. Per gli aiuti di stato, abbiamo compilato la dichiarazione 'de minimis' con gli incentivi ricevuti negli ultimi tre anni. Il commercialista dice che il DURC è regolare, ma non l'ho allegato perché di solito lo verificano loro online. Non credo serva altro, mi sembra tutto completo. Il progetto è solido e speriamo bene.",
	output: "",
	}, 
	{
	profile: "Checker Adeguata Verifica Cliente (KYC)",
	title: "xxxx",
	context: "Un'agenzia immobiliare sta registrando un nuovo cliente per l'acquisto di un immobile di lusso. L'agente ha raccolto alcune informazioni preliminari e le inserisce nel sistema per una prima valutazione del profilo di rischio KYC/AML.",
	input: "Nuovo cliente: Sig. Mario Rossi. Operazione: Acquisto immobile residenziale a Milano, valore 2.5 milioni di euro. Documenti: Mi ha mandato via email la scansione della sua carta d'identità, ma è scaduta il mese scorso. Dice che la sta rinnovando. Residenza: Dichiara di essere residente a Lugano, in Svizzera. Professione: Imprenditore nel settore delle criptovalute. Modalità di pagamento: Intende pagare tramite bonifico da un conto presso una banca di Singapore. Titolare Effettivo: Dichiara di agire per proprio conto, quindi è lui il titolare effettivo. Scopo del rapporto: Acquisto di una seconda casa per investimento. Non mi ha fornito altre informazioni, sembra un po' reticente a dare dettagli sulla sua attività. Ha fretta di concludere l'operazione.",
	output: "",
	}, 
	{
	profile: "Revisore Clausole Termini di Servizio",
	title: "xxxx",
	context: "Una startup che offre un servizio online in abbonamento ha preparato una bozza delle clausole più importanti dei suoi Termini di Servizio. Prima di pubblicarli, vuole un'analisi per identificare eventuali clausole che potrebbero essere considerate problematiche o vessatorie per i consumatori.",
	input: "Benvenuto sul nostro servizio. Utilizzando la piattaforma, accetti i seguenti termini.' 'Prezzi e Pagamenti: L'abbonamento si rinnova automaticamente ogni mese. Ci riserviamo il diritto di modificare i prezzi in qualsiasi momento, con le modifiche che avranno effetto dal rinnovo successivo.' 'Cancellazione: Puoi cancellare il tuo abbonamento in qualsiasi momento. La cancellazione avrà effetto dalla fine del periodo di fatturazione in corso. Non sono previsti rimborsi per i periodi parzialmente utilizzati.' 'Limitazione di Responsabilità: Il nostro servizio è fornito 'così com'è'. In nessun caso la nostra responsabilità totale per qualsiasi danno potrà superare l'importo pagato da te negli ultimi tre mesi.' 'Modifiche al Servizio: Possiamo modificare o interrompere il servizio, o qualsiasi sua parte, in qualsiasi momento e senza preavviso.' 'Contenuti dell'Utente: Concedi a noi una licenza mondiale, perpetua e irrevocabile per utilizzare, riprodurre e modificare qualsiasi contenuto che carichi sulla piattaforma.' 'Legge Applicabile: Qualsiasi controversia sarà regolata dalla legge dello Stato del Delaware, USA, e il foro competente esclusivo sarà quello di Wilmington, Delaware.",
	output: "",
	}, 
	{
	profile: "Analizzatore GDPR Marketing",
	title: "xxxx",
	context: "Un'azienda di e-commerce sta progettando il modulo di iscrizione alla newsletter da inserire nel footer del proprio sito. Il marketing manager ha scritto il testo e vuole assicurarsi che sia conforme al GDPR prima di mandarlo online.",
	input: "Rimani Aggiornato! Iscriviti alla nostra newsletter per ricevere offerte esclusive e novità sui nostri prodotti. [Campo per inserire l'indirizzo email] Spuntando la casella, accetti di ricevere le nostre comunicazioni di marketing e la nostra privacy policy. (Checkbox preselezionata) Acconsento al trattamento dei miei dati per finalità di marketing, come descritto nella privacy policy. Potrai disiscriverti in qualsiasi momento. Vogliamo anche usare i tuoi dati per creare un profilo delle tue preferenze e mandarti offerte super personalizzate. I tuoi dati saranno condivisi con i nostri partner commerciali per le loro campagne marketing.",
	output: "",
	}, 
	{
	profile: "Verificatore Anti-Bias Annunci Lavoro",
	title: "xxxx",
	context: "Un'azienda manifatturiera sta cercando un nuovo responsabile per il reparto produttivo. Il direttore del personale ha scritto un annuncio di lavoro e lo passa al suo assistente per una revisione prima della pubblicazione.",
	input: "Cercasi brillante neolaureato in ingegneria per posizione di Responsabile di Produzione. Siamo un'azienda leader nel settore metalmeccanico e cerchiamo un giovane dinamico e ambizioso per guidare il nostro team di operai. Il candidato ideale è un uomo di polso, capace di farsi rispettare e di gestire con fermezza il personale. Deve avere una forte resistenza allo stress e essere disposto a fare straordinari quando necessario. Richiesta una presenza impeccabile e ottime doti relazionali. È un'ottima opportunità di carriera per un giovane che vuole crescere e dimostrare il suo valore. Il candidato dovrà coordinare il lavoro degli uomini in produzione. Richiesta madrelingua italiana. Inviare CV con foto a... Astenersi perditempo.",
	output: "",
	}, 
	{
	profile: "Revisore Comunicazioni Mediche",
	title: "xxxx",
	context: "Un'azienda che produce integratori alimentari ha preparato il testo per la pagina prodotto di un nuovo integratore per il sistema immunitario. Il testo deve essere controllato per assicurarsi che non violi le normative sulla comunicazione di prodotti salutistici.",
	input: "Immuno-Shield: la cura definitiva per i malanni di stagione! Stanco di ammalarti ogni inverno? Con il nostro nuovo e rivoluzionario integratore Immuno-Shield, puoi dire addio a raffreddore e influenza per sempre. La nostra formula esclusiva è una potente miscela di erbe e vitamine che garantisce di potenziare il tuo sistema immunitario del 300%. Studi scientifici dimostrano che l'assunzione di Immuno-Shield previene qualsiasi infezione virale e batterica. È un vero e proprio vaccino naturale, senza effetti collaterali. Migliaia di medici lo raccomandano ai loro pazienti. Non lasciare che i germi rovinino le tue giornate. Immuno-Shield è la soluzione garantita per una salute di ferro. Acquistalo ora e non ti ammalerai mai più! Risultati visibili e garantiti in soli 3 giorni.",
	output: "",
	}, 
	{
	profile: "Validatore di Green Claims (CSRD)",
	title: "xxxx",
	context: "Un brand di moda sta lanciando una nuova linea di abbigliamento e ha preparato un testo di marketing per il sito web per comunicarne le caratteristiche di sostenibilità. Vuole verificare che le affermazioni non siano a rischio di greenwashing.",
	input: "Scopri la nostra nuova collezione 'Terra Pura': 100% eco-friendly. Abbiamo a cuore il pianeta, per questo abbiamo creato una linea di abbigliamento completamente sostenibile e amica dell'ambiente. Ogni capo è realizzato con materiali naturali e a impatto zero. I nostri tessuti sono ecologici e biodegradabili, per un futuro più verde. Scegliendo 'Terra Pura', contribuisci a salvare il pianeta. La nostra produzione è etica e rispetta la natura. Siamo un'azienda green, impegnata a ridurre l'inquinamento. Questi vestiti sono buoni per te e buoni per il mondo. Unisciti al nostro movimento per una moda consapevole. Il nostro packaging è completamente riciclabile.",
	output: "",
	},
	{
	profile: "Checker Accessibilità Testuale (WCAG)",
	title: "xxxx",
	context: "Un ente pubblico sta scrivendo il testo per una pagina del proprio sito istituzionale che spiega come accedere a un servizio. Devono assicurarsi che il testo sia accessibile a tutti i cittadini, inclusi quelli con disabilità.",
	input: "Per usufruire del servizio, è necessario seguire la procedura descritta di seguito. Innanzitutto, l'utente deve procurarsi il modulo apposito. Per scaricare il documento, clicca qui. Una volta compilato in ogni sua parte, il suddetto modulo deve essere inviato tramite il portale. L'interfaccia del portale è molto complessa, quindi fate attenzione. Abbiamo anche preparato un documento con tutte le informazioni, è un PDF di 50 pagine. L'immagine qui sotto mostra uno screenshot del portale. Se avete problemi, potete trovare ulteriori informazioni in questa pagina. Ricordatevi che la scadenza è perentoria. Il linguaggio utilizzato nel bando è molto tecnico e pieno di acronimi, ma è necessario per legge. Non usare il browser Explorer perché non funziona bene.",
	output: "",
	},
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