import logging # <-- AGGIUNGI QUESTA RIGA
import time
import requests
import os
import uvicorn
import redis
from fastapi import Request, FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field
from datetime import date
from fastapi import Header
from dotenv import load_dotenv
load_dotenv()
from svix.webhooks import Webhook, WebhookVerificationError # <-- MODIFICA QUESTA RIG
from supabase import create_client, Client
# --- NUOVE IMPORTAZIONI PER IL CORS ---
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import ai_core
# --- Aggiungi l'importazione per la verifica dei JWT RS256 ---
from jose import jwt, jwk # pip install python-jose

# --- Aggiungi l'importazione per la verifica dei JWT RS256 ---
from jose import jwt, jwk # pip install python-jose

# --- INIZIO: NUOVI CONTROLLI VARIABILI D'AMBIENTE CRITICHE ---

# Variabili Clerk
CLERK_WEBHOOK_SECRET = os.getenv("CLERK_WEBHOOK_SECRET")
if not CLERK_WEBHOOK_SECRET:
    logging.error("CLERK_WEBHOOK_SECRET non configurato.")
    raise ValueError("CLERK_WEBHOOK_SECRET non trovata nel file .env o nelle variabili d'ambiente.")

CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
if not CLERK_JWKS_URL:
    logging.error("CLERK_JWKS_URL non configurato.")
    raise ValueError("CLERK_JWKS_URL non trovata nel file .env o nelle variabili d'ambiente.")

# Variabili Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
if not SUPABASE_URL:
    logging.error("SUPABASE_URL non configurato.")
    raise ValueError("SUPABASE_URL non trovata nel file .env o nelle variabili d'ambiente.")

SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_SERVICE_KEY:
    logging.error("SUPABASE_SERVICE_KEY non configurato.")
    raise ValueError("SUPABASE_SERVICE_KEY non trovata nel file .env o nelle variabili d'ambiente.")

# Variabili Google AI (Gemini)
# ai_core.py ha già un controllo, ma lo duplichiamo qui per un fail-fast a livello di app principale.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY non configurato.")
    raise ValueError("GOOGLE_API_KEY non trovata nel file .env o nelle variabili d'ambiente.")

# --- FINE: NUOVI CONTROLLI VARIABILI D'AMBIENTE CRITICHE ---

# Inizializzazione Supabase con le variabili verificate
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
if not CLERK_JWKS_URL:
    raise ValueError("CLERK_JWKS_URL non trovata nel file .env")

# --- Pydantic Models ---
class TextInput(BaseModel):
    text: str = Field(..., min_length=10)
    profile_name: str = Field("Generico", description="Nome del profilo AI da utilizzare per la validazione.") 

class QualityReport(BaseModel):
    reasoning: str
    human_quality_score: int

class UsageInfo(BaseModel):
    count: int
    limit: int

class ValidationResponse(BaseModel):
    normalized_text: str
    quality_report: QualityReport
    usage: UsageInfo # Ora questa riga funzionerà

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Text Validator API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CONFIGURAZIONE CORS ---
# Definiamo da quali "origini" (domini) il nostro backend accetterà richieste.
# Per lo sviluppo, ci basta accettare richieste dal nostro server frontend.
origins = [
    "http://localhost:3000", # Per lo sviluppo locale
    "https://text-validator-frontend.vercel.app", # L'URL di produzione
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permetti tutti i metodi (GET, POST, etc.)
    allow_headers=["*"],  # Permetti tutti gli header
)
# --- FINE CONFIGURAZIONE CORS ---


@app.get("/health", tags=["Monitoring"])
async def read_health():
    return {"status": "ok"}


@app.post("/validate", 
          response_model=ValidationResponse, 
          tags=["Core Logic"])
@limiter.limit("5/minute")
async def validate_text(request: Request, payload: TextInput, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token di autenticazione mancante.")

    # --- Renamed 'jwt_string' to avoid collision with jose.jwt library ---
    clerk_jwt_token_string = authorization.split(" ")[1]
    user_id = None

    try:
        # Scarica le chiavi pubbliche di Clerk (in un'app di produzione, useresti una cache)
        jwks_response = requests.get(CLERK_JWKS_URL)
        jwks_response.raise_for_status() # Lancia un'eccezione per errori HTTP
        jwks_data = jwks_response.json()

        # Estrai l'header del JWT per trovare l'algoritmo e il 'kid' (Key ID)
        header = jwt.get_unverified_header(clerk_jwt_token_string)
        
        # Trova la chiave pubblica corrispondente dal JWKS usando il 'kid'
        public_key = None
        for key_data in jwks_data["keys"]:
            if key_data["kid"] == header["kid"]:
                public_key = jwk.construct(key_data)
                break
        
        if not public_key:
            raise Exception("Chiave pubblica corrispondente (kid) non trovata in Clerk JWKS. Controlla il CLERK_JWKS_URL e la configurazione di Clerk.")

        # Verifica e decodifica il JWT usando la chiave pubblica
        decoded_token = jwt.decode(
            clerk_jwt_token_string, # Usa la variabile stringa del token
            public_key,             # Usa l'oggetto chiave pubblica
            algorithms=["RS256"],   # Specifica l'algoritmo di firma
            audience=None,          # Clerk di solito non usa un audience specifico per questi token
            options={"verify_signature": True, "verify_aud": False, "verify_iss": False} # Verifichiamo solo la firma
        )
        
        user_id = decoded_token.get("sub") # Clerk usa 'sub' per l'ID utente
        if not user_id:
            raise Exception("ID utente (sub) non trovato nel token decodificato. Il token potrebbe essere malformato.")
        
    except requests.exceptions.RequestException as e:
        logging.error(f"!!! ERRORE SCARICAMENTO JWKS: Impossibile scaricare le chiavi JWKS da Clerk. Controlla CLERK_JWKS_URL o la connessione di rete. Errore: {e}")
        raise HTTPException(status_code=503, detail=f"Errore di configurazione autenticazione: Impossibile raggiungere i servizi Clerk.")
    except Exception as e:
        logging.error(f"!!! ERRORE VALIDAZIONE TOKEN: {str(e)}")
        raise HTTPException(status_code=401, detail=f"La validazione del token è fallita: {str(e)}")

    # --- 2. RECUPERO PROFILO "PAZIENTE" (La soluzione alla Race Condition) ---
    profile = None
    # Tentiamo di recuperare il profilo fino a 3 volte (per circa 1.5 secondi totali)
    for attempt in range(3):
        profile_res = supabase.table('profiles').select('*').eq('id', user_id).execute()
        if profile_res.data:
            profile = profile_res.data[0]
            logging.info(f"Profilo trovato al tentativo #{attempt + 1} per utente {user_id}.")
            break # Profilo trovato, usciamo dal ciclo
        
        logging.warning(f"Tentativo #{attempt + 1}: Profilo per utente {user_id} non ancora visibile, attendo 500ms...")
        time.sleep(0.5) # Pausa di mezzo secondo

    if not profile:
        logging.critical(f"CRITICO: Profilo utente {user_id} non trovato dopo multipli tentativi. Il trigger del database potrebbe essere fallito.")
        raise HTTPException(status_code=500, detail="CRITICO: Profilo utente non trovato dopo multipli tentativi. Contatta il supporto.")

    # --- 3. IL RESTO DELLA FUNZIONE (Logica Limiti e AI) ---
    DAILY_LIMIT = 10
    if profile.get('role') != 'admin':
        today = str(date.today())
        last_used = profile.get('last_used_date')
        current_count = profile.get('usage_count', 0)

        if last_used != today:
            current_count = 0
            supabase.table('profiles').update({'last_used_date': today, 'usage_count': 0}).eq('id', user_id).execute()

        if current_count >= DAILY_LIMIT:
            raise HTTPException(status_code=429, detail=f"Hai superato il limite giornaliero di {DAILY_LIMIT} chiamate.")
    
    # --- FASE 1: Normalizzazione del testo ---
    try:
        normalized_text = await ai_core.normalize_text(payload.text, profile_name=payload.profile_name)
        if not isinstance(normalized_text, str):
            raise HTTPException(status_code=500, detail="Errore interno: La Fase 1 non ha restituito testo valido.")
    except Exception as e:
        error_detail = f"Errore imprevisto durante la Fase 1: {str(e)}"
        logging.error(f"Errore AI Fase 1: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

    # --- FASE 2: Calcolo del punteggio di qualità ---
    try:
        quality_report_data = await ai_core.get_quality_score(
            original_text=payload.text,
            normalized_text=normalized_text,
            profile_name=payload.profile_name
        )
        if "error" in quality_report_data:
            logging.error(f"Errore LLM nella Fase 2: {quality_report_data.get('details', 'Dettagli non disponibili')}")
            raise HTTPException(status_code=503, detail=quality_report_data.get("details", "Errore LLM."))
        
        quality_report = QualityReport(**quality_report_data)
    except Exception as e:
        error_detail = f"Errore imprevisto durante la Fase 2: {str(e)}"
        logging.error(f"Errore AI Fase 2: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

    # --- AGGIORNAMENTO CONTEGGIO ---
    new_count = profile.get('usage_count', 0)
    if profile.get('role') != 'admin':
        new_count += 1
        supabase.table('profiles').update({'usage_count': new_count}).eq('id', user_id).execute()

    # --- Costruzione e invio della risposta finale ---
    return ValidationResponse(
        normalized_text=normalized_text.strip(),
        quality_report=quality_report,
        usage=UsageInfo(count=new_count, limit=DAILY_LIMIT if profile.get('role') != 'admin' else -1)
    )

@app.post("/webhooks/new-user", include_in_schema=False) # Nascosto dalla documentazione pubblica
async def handle_new_user_webhook(request: Request, payload: dict, x_webhook_secret: str = Header(None)):
    """
    Endpoint per ricevere il webhook da Supabase alla creazione di un nuovo utente.
    """
    # 1. Sicurezza: Controlla il segreto del webhook
    expected_secret = os.environ.get("SUPABASE_WEBHOOK_SECRET")
    if not expected_secret or x_webhook_secret != expected_secret:
        print("!!! ERRORE WEBHOOK: Segreto non valido o mancante.")
        raise HTTPException(status_code=401, detail="Segreto del webhook non valido.")

    # 2. Estrai i dati dell'utente dal payload del webhook
    try:
        event_type = payload.get("type")
        user_record = payload.get("record")
        
        if event_type != "INSERT" or not user_record:
            print(f"Webhook ricevuto per un evento non gestito: {event_type}")
            return {"status": "evento ignorato"}

        user_id = user_record.get("id")
        user_email = user_record.get("email")

        if not user_id:
            raise ValueError("ID utente mancante nel payload del webhook.")

    except Exception as e:
        print(f"!!! ERRORE WEBHOOK: Payload non valido. {str(e)}")
        raise HTTPException(status_code=422, detail=f"Payload del webhook non valido: {str(e)}")

    # 3. Logica di creazione del profilo
    print(f"Webhook ricevuto: Tentativo di creazione profilo per utente {user_id}...")
    try:
        # Usiamo il client Supabase (con service_key) per creare il profilo
        insert_res = supabase.table('profiles').insert({
            'id': user_id,
            'email': user_email 
        }).execute()
        
        # Controllo di sicurezza: verifichiamo che l'inserimento sia andato a buon fine
        if not insert_res.data:
             raise Exception(f"L'inserimento del profilo per {user_id} non ha restituito dati.")

    except Exception as e:
        print(f"!!! ERRORE WEBHOOK: Impossibile creare il profilo per {user_id}. Errore: {str(e)}")
        # Se la creazione del profilo fallisce, restituiamo un 500
        # in modo che Supabase possa ritentare l'invio del webhook.
        raise HTTPException(status_code=500, detail=f"Impossibile creare il profilo: {str(e)}")

    print(f"Profilo creato con successo per l'utente {user_id} via webhook.")
    return {"status": "profilo creato con successo"}

@app.post("/api/webhook/clerk/", status_code=status.HTTP_200_OK)
async def clerk_webhook_handler(request: Request, response: Response):
    # 1. Recupero del payload grezzo (CRITICO per la verifica della firma)
    try:
        payload = await request.body()
    except Exception as e:
        logging.error(f"Could not read request body for webhook: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not read request body.")

    # 2. Recupero degli headers Svix
    headers = request.headers
    
    # 3. Verifica della Firma Svix
    try:
        wh = Webhook(CLERK_WEBHOOK_SECRET)
        # wh.verify accetta payload (grezzo) e headers, e verifica Timestamp/Signature.
        event_message = wh.verify(payload, headers)
        
    except WebhookVerificationError as e:
        logging.warning(f"Clerk Webhook verification failed: {e}. Payload: {payload.decode()}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signature verification failed.")
        
    except Exception as e:
        logging.error(f"Errore inatteso durante la verifica del webhook: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected webhook verification error.")

    # 4. Logica di Business Post-Verifica (Payload sicuro)
    event_type = event_message.get('type')
    user_data = event_message.get('data')

    if not user_data:
        logging.warning(f"Webhook event of type {event_type} received, but 'data' field is missing.")
        return {"message": "Webhook event acknowledged, but 'data' field is missing."}

    user_id = user_data.get('id')
    user_email = user_data.get('email_addresses')[0]['email_address'] if user_data.get('email_addresses') else None

    if not user_id:
        logging.error(f"User ID mancante nel payload del webhook per evento {event_type}.")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="User ID missing in webhook payload.")

    if event_type == "user.created":
        logging.info(f"Webhook user.created: Tentativo di creazione profilo per utente {user_id} ({user_email})...")
        try:
            # Usiamo il client Supabase (inizializzato con service_key) per creare il profilo
            # La service_key bypassa la RLS.
            data_to_insert = {
            "id": user_id,  # Clerk user ID è una stringa, non un UUID!
            "email": user_email,
            "usage_count": 0,
            "role": "user"
        }
            insert_res = supabase.table('profiles').insert(data_to_insert).execute()
            
            if not insert_res.data:
                raise Exception(f"L'inserimento del profilo per {user_id} non ha restituito dati.")

            logging.info(f"Profilo creato con successo per l'utente {user_id} via webhook.")
            return {"message": f"User {user_id} provisioned successfully."}

        except Exception as e:
            logging.error(f"ERRORE CRITICO WEBHOOK: Impossibile creare il profilo per {user_id}. Errore: {str(e)}")
            # RESTITUIRE 500 per fare in modo che Svix ritenti la chiamata!
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Provisioning failed: {e}")

    elif event_type == "user.deleted":
        logging.info(f"Webhook user.deleted: Tentativo di eliminazione/anonimizzazione profilo per utente {user_id}.")
        try:
            # Logica per eliminare o anonimizzare i dati in public.profiles.
            # Esempio: supabase.table('profiles').delete().eq('id', user_id).execute()
            # O aggiornare: supabase.table('profiles').update({'email': null, 'usage_count': 0}).eq('id', user_id).execute()
            delete_res = supabase.table('profiles').delete().eq('id', user_id).execute()
            if not delete_res.data:
                logging.warning(f"Nessun profilo trovato o eliminato per l'utente {user_id} (potrebbe essere già stato cancellato).")
            logging.info(f"Profilo eliminato/anonimizzato per user {user_id}.")
            return {"message": f"User {user_id} deletion acknowledged."}
        except Exception as e:
            logging.error(f"ERRORE CRITICO WEBHOOK: Impossibile eliminare/anonimizzare profilo per {user_id}. Errore: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Deletion failed: {e}")

    else:
        logging.info(f"Webhook: Evento di tipo {event_type} ricevuto e ignorato.")
        return {"message": f"Event type {event_type} acknowledged, no action taken."}

@app.get("/user-status", response_model=UsageInfo, tags=["User Management"])
@limiter.limit("50/minute") # Limite più generoso per le chiamate di stato
async def get_user_status(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token di autenticazione mancante.")

    clerk_jwt_token_string = authorization.split(" ")[1]
    user_id = None

    try:
        # Re-using JWT verification logic
        jwks_response = requests.get(CLERK_JWKS_URL)
        jwks_response.raise_for_status()
        jwks_data = jwks_response.json()
        header = jwt.get_unverified_header(clerk_jwt_token_string)
        public_key = None
        for key_data in jwks_data["keys"]:
            if key_data["kid"] == header["kid"]:
                public_key = jwk.construct(key_data)
                break
        if not public_key:
            raise Exception("Chiave pubblica corrispondente (kid) non trovata in Clerk JWKS.")
        decoded_token = jwt.decode(
            clerk_jwt_token_string,
            public_key,
            algorithms=["RS256"],
            audience=None,
            options={"verify_signature": True, "verify_aud": False, "verify_iss": False}
        )
        user_id = decoded_token.get("sub")
        if not user_id:
            raise Exception("ID utente (sub) non trovato nel token decodificato.")
            
    except requests.exceptions.RequestException as e:
        logging.error(f"!!! ERRORE SCARICAMENTO JWKS per /user-status: {e}")
        raise HTTPException(status_code=503, detail=f"Errore di configurazione autenticazione: Impossibile raggiungere i servizi Clerk.")
    except Exception as e:
        logging.error(f"!!! ERRORE VALIDAZIONE TOKEN per /user-status: {str(e)}")
        raise HTTPException(status_code=401, detail=f"La validazione del token è fallita: {str(e)}")

    # --- Recupero Profilo e Limiti ---
    profile = None
    # Non facciamo la logica di "pazienza" qui, se il profilo non c'è, è un problema.
    profile_res = supabase.table('profiles').select('*').eq('id', user_id).execute()
    if profile_res.data:
        profile = profile_res.data[0]

    if not profile:
        logging.critical(f"CRITICO: Profilo utente {user_id} non trovato per /user-status.")
        raise HTTPException(status_code=500, detail="CRITICO: Profilo utente non trovato. Contatta il supporto.")

    DAILY_LIMIT = 10 # Deve essere consistente con validate_text

    current_count = profile.get('usage_count', 0)
    # Calcola il limite e il conteggio basato sul ruolo admin
    if profile.get('role') == 'admin':
        return UsageInfo(count=current_count, limit=-1) # -1 per illimitato
    else:
        # Se è un nuovo giorno, il contatore viene resettato nella validate_text,
        # ma qui mostriamo lo stato attuale prima di ogni validazione
        return UsageInfo(count=current_count, limit=DAILY_LIMIT)

if __name__ == "__main__":
    # Legge la porta dalla variabile d'ambiente PORT fornita da Cloud Run
    # Se non la trova, usa 8000 come default (per lo sviluppo locale)
    port = int(os.environ.get("PORT", 8000))
    
    # Avvia il server uvicorn programmaticamente
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)