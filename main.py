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
from typing import List, Optional
# --- Aggiungi l'importazione per la verifica dei JWT RS256 ---
from jose import jwt, jwk # pip install python-jose

# --- Aggiungi l'importazione per la verifica dei JWT RS256 ---
from jose import jwt, jwk # pip install python-jose

PLANS = {
    "free": {
        "shared_limit": 5,
        "max_input_length": 1500,
        "validator": {
            "allowed_profiles": ["Generico", "L'Umanizzatore", "Social Media Manager B2B", "Ottimizzatore Email di Vendita"],
            "quality_check": False
        },
        "interpreter": {
            "allowed_profiles": ["Spiega in Parole Semplici"],
            "quality_check": False
        },
        "compliance_checkr": {
            "enabled": False,
            "allowed_profiles": []
        },
        "strategist": {
            "enabled": False,
            "allowed_profiles": []
        },
        "ctov": {
            "enabled": False,
            "max_profiles": 0
        }
    },
    "starter": {
        "shared_limit": 20,
        "max_input_length": 15000,
        "validator": {
            "allowed_profiles": [
                # Profili Free
                "Generico", "L'Umanizzatore", "Social Media Manager B2B", "Ottimizzatore Email di Vendita",
                # Profili Aggiuntivi Starter
                "Copywriter Persuasivo", "Scrittore di Newsletter", "Generatore Descrizioni Prodotto E-commerce", 
                "Scrittore Testi per Landing Page", "Redattore di Annunci di Lavoro", "Scrittore di Proposte Commerciali",
                "Redattore di Sezioni di Business Plan", "Comunicatore di Crisi PR"
            ],
            "quality_check": True
        },
        "interpreter": {
            "allowed_profiles": [
                # Profilo Free
                "Spiega in Parole Semplici",
                # Profili Aggiuntivi Starter
                "Sintetizzare di Meeting e Trascrizioni", "Analizzatore di Feedback dei Clienti", "Estrattore di Dati Strutturati"
            ],
            "quality_check": True
        },
        "compliance_checkr": {
            "enabled": True, # Abilitato per l'assaggio
            "allowed_profiles": ["Verificatore Anti-Bias Annunci Lavoro", "Analizzatore Disclaimer E-commerce"]
        },
        "strategist": {
            "enabled": True, # Abilitato per l'assaggio
            "allowed_profiles": ["Sviluppatore di Buyer Persona", "Ideatore di Pillar Page e Content Cluster", "Generatore di Brief Creativo per Campagne"]
        },
        "ctov": {
            "enabled": True,
            "max_profiles": 2
        }
    },
    "pro": {
        "shared_limit": 150,
        "max_input_length": 100000,
        "validator": {
            "allowed_profiles": "all",
            "quality_check": True
        },
        "interpreter": {
            "allowed_profiles": "all",
            "quality_check": True
        },
        "compliance_checkr": {
            "enabled": True,
            "allowed_profiles": "all"
        },
        "strategist": {
            "enabled": True,
            "allowed_profiles": "all"
        },
        "ctov": {
            "enabled": True,
            "max_profiles": 10
        }
    },
    "business": { # NUOVO PIANO
        "shared_limit": -1, # Illimitato o gestito a livello di team
        "max_input_length": None,
        "validator": { "allowed_profiles": "all", "quality_check": True },
        "interpreter": { "allowed_profiles": "all", "quality_check": True },
        "compliance_checkr": { "enabled": True, "allowed_profiles": "all" },
        "strategist": { "enabled": True, "allowed_profiles": "all" },
        "ctov": { "enabled": True, "max_profiles": -1 } # Illimitato
    },
    "admin": {
        "shared_limit": -1,
        "max_input_length": None,
        "validator": { "allowed_profiles": "all", "quality_check": True },
        "interpreter": { "allowed_profiles": "all", "quality_check": True },
        "compliance_checkr": { "enabled": True, "allowed_profiles": "all" },
        "strategist": { "enabled": True, "allowed_profiles": "all" },
        "ctov": { "enabled": True, "max_profiles": -1 }
    }
}

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
    ctov_profile_id: Optional[str] = None 

class QualityReport(BaseModel):
    reasoning: str
    human_quality_score: int

class UsageInfo(BaseModel):
    count: int
    limit: int

class CTOVProfileBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    mission: Optional[str] = Field(None, max_length=250)
    archetype: Optional[str] = None
    tone_traits: Optional[List[str]] = []
    banned_terms: Optional[List[str]] = []

class CTOVProfileCreate(CTOVProfileBase):
    pass

class CTOVProfileResponse(CTOVProfileBase):
    id: str # UUID sarà convertito in stringa
# === INIZIO BLOCCO DA AGGIUNGERE ===
# 1. DEFINISCI IL NUOVO MODELLO DI RISPOSTA PER LO STATO UTENTE
class UserStatusResponse(BaseModel):
    usage: UsageInfo
    tier: str
    validator_profiles: list[str] | str
    interpreter_profiles: list[str] | str
    compliance_access: bool
    strategist_access: bool # NUOVO
    ctov_access: bool               
    ctov_max_profiles: int          
    ctov_profiles: List[CTOVProfileResponse] 
# === FINE BLOCCO DA AGGIUNGERE ===

class ValidationResponse(BaseModel):
    normalized_text: str
    quality_report: QualityReport | None = None # <-- MODIFICA QUI
    usage: UsageInfo
    
class InterpretationResponse(BaseModel):
    interpreted_text: str
    quality_report: QualityReport | None = None
    usage: UsageInfo

class ComplianceResponse(BaseModel):
    compliance_report: str
    usage: UsageInfo
    
class StrategyResponse(BaseModel):
    strategy_text: str
    usage: UsageInfo

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


# ==============================================================================
# === NUOVO ENDPOINT: STRATEGIST ===============================================
# ==============================================================================
@app.post("/strategist", response_model=StrategyResponse, tags=["Strategist"])
@limiter.limit("5/minute")
async def create_strategy(request: Request, payload: TextInput, authorization: str = Header(None)):
    user_id, profile = await get_user_profile_from_token(authorization)
    
    # --- LOGICA DI GESTIONE PIANI PER STRATEGIST ---
    user_tier_name = profile.get('subscription_tier', 'free')
    user_role = profile.get('role', 'user')
    plan = PLANS.get("admin") if user_role == 'admin' else PLANS.get(user_tier_name, PLANS["free"])
    
    if not plan["strategist"]["enabled"]:
        raise HTTPException(status_code=403, detail="Lo Strategist non è incluso nel tuo piano. Esegui l'upgrade al piano Pro.")

    # Verifica lunghezza massima dell'input
    max_length = plan.get("max_input_length")
    if max_length is not None and len(payload.text) > max_length:
        raise HTTPException(
            status_code=413,
            detail=f"Il testo inserito supera il limite di {max_length} caratteri per il tuo piano."
        )
    
    # Verifica limite di chiamate condiviso
    shared_limit = plan["shared_limit"]
    current_count = profile.get('usage_count', 0)
    if shared_limit != -1:
        today = str(date.today())
        if profile.get('last_used_date') != today:
            current_count = 0
            supabase.table('profiles').update({'last_used_date': today, 'usage_count': 0}).eq('id', user_id).execute()
        if current_count >= shared_limit:
            raise HTTPException(status_code=429, detail=f"Hai superato il limite giornaliero condiviso di {shared_limit} chiamate.")
            
    try:
        strategy_text = await ai_core.generate_strategy(payload.text, profile_name=payload.profile_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la generazione della strategia: {str(e)}")
    
    # Aggiornamento conteggio
    new_count = current_count + 1
    if shared_limit != -1:
        supabase.table('profiles').update({'usage_count': new_count}).eq('id', user_id).execute()

    return StrategyResponse(
            strategy_text=strategy_text.strip(),
            usage=UsageInfo(count=new_count, limit=shared_limit)
        )

@app.get("/health", tags=["Monitoring"])
async def read_health():
    return {"status": "ok"}


@app.post("/validate", response_model=ValidationResponse, tags=["Validator"])
@limiter.limit("5/minute")
async def validate_text(request: Request, payload: TextInput, authorization: str = Header(None)):
    user_id, profile = await get_user_profile_from_token(authorization)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token di autenticazione mancante.")

    # ... la logica di validazione del token e recupero profilo rimane IDENTICA ...
    # ... fino a dopo il recupero del 'profile' ...
    
    clerk_jwt_token_string = authorization.split(" ")[1]
    user_id = None
    try:
        # Logica di validazione token...
        jwks_response = requests.get(CLERK_JWKS_URL)
        jwks_response.raise_for_status()
        jwks_data = jwks_response.json()
        header = jwt.get_unverified_header(clerk_jwt_token_string)
        public_key = None
        for key_data in jwks_data["keys"]:
            if key_data["kid"] == header["kid"]:
                public_key = jwk.construct(key_data)
                break
        if not public_key: raise Exception("Chiave pubblica non trovata.")
        options = {
            "verify_signature": True, 
            "verify_aud": False, 
            "verify_iss": False,
            "leeway": 5
        }
        decoded_token = jwt.decode(clerk_jwt_token_string, public_key, algorithms=["RS256"], options=options)
        user_id = decoded_token.get("sub")
        if not user_id: raise Exception("ID utente non trovato.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Validazione token fallita: {str(e)}")

    profile_res = supabase.table('profiles').select('*').eq('id', user_id).execute()
    if not profile_res.data:
        raise HTTPException(status_code=500, detail="Profilo utente non trovato.")
    profile = profile_res.data[0]
    
    # --- NUOVA LOGICA DI GESTIONE PIANI PER VALIDATOR ---
    user_tier_name = profile.get('subscription_tier', 'free')
    user_role = profile.get('role', 'user')
    plan = PLANS.get("admin") if user_role == 'admin' else PLANS.get(user_tier_name, PLANS["free"])
    # === INIZIO BLOCCO DA AGGIUNGERE ===
    # 0. Verifica lunghezza massima dell'input
    max_length = plan.get("max_input_length")
    if max_length is not None and len(payload.text) > max_length:
        raise HTTPException(
            status_code=413, # 413 Payload Too Large
            detail=f"Il testo inserito ({len(payload.text)} caratteri) supera il limite di {max_length} caratteri consentito per il tuo piano. Esegui l'upgrade per analizzare documenti più lunghi."
        )
    # === FINE BLOCCO DA AGGIUNGERE ===
    validator_plan = plan["validator"]
    model_to_use = ai_core.VALIDATOR_MODEL_NAME
    # 1. Verifica profilo consentito
    if validator_plan["allowed_profiles"] != "all" and payload.profile_name not in validator_plan["allowed_profiles"]:
        raise HTTPException(status_code=403, detail=f"Il profilo Validator '{payload.profile_name}' non è incluso nel tuo piano.")

    # 2. Verifica limite di chiamate condiviso
    shared_limit = plan["shared_limit"]
    current_count = profile.get('usage_count', 0)
    if shared_limit != -1:
        today = str(date.today())
        if profile.get('last_used_date') != today:
            current_count = 0
            supabase.table('profiles').update({'last_used_date': today, 'usage_count': 0}).eq('id', user_id).execute()
        if current_count >= shared_limit:
            raise HTTPException(status_code=429, detail=f"Hai superato il limite giornaliero condiviso di {shared_limit} chiamate.")
    
    ctov_data = None
    if payload.ctov_profile_id:
        # Se viene richiesto un profilo CTOV, recuperalo
        ctov_res = supabase.table('ctov_profiles').select('*').eq('id', payload.ctov_profile_id).eq('user_id', user_id).single().execute()
        if not ctov_res.data:
            raise HTTPException(status_code=404, detail="Profilo Custom Tone of Voice non trovato o non autorizzato.")
        ctov_data = ctov_res.data
        
    # --- ELABORAZIONE AI ---
    try:
        normalized_text = await ai_core.normalize_text(payload.text, profile_name=payload.profile_name, model_name=model_to_use, ctov_data=ctov_data)
        
        quality_report_obj = None
        if validator_plan["quality_check"]:
            quality_report_data = await ai_core.get_quality_score(original_text=payload.text, normalized_text=normalized_text, profile_name=payload.profile_name, model_name=model_to_use)
            if "error" not in quality_report_data and "human_quality_score" in quality_report_data:
                score = quality_report_data.get("human_quality_score", 0)
                quality_report_data["human_quality_score"] = round(score)
                quality_report_obj = QualityReport(**quality_report_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'elaborazione AI: {str(e)}")

    # --- AGGIORNAMENTO CONTEGGIO ---
    new_count = current_count + 1
    if shared_limit != -1:
        supabase.table('profiles').update({'usage_count': new_count}).eq('id', user_id).execute()

    return ValidationResponse(
        normalized_text=normalized_text.strip(),
        quality_report=quality_report_obj,
        usage=UsageInfo(count=new_count, limit=shared_limit)
    )

@app.post("/interpret", response_model=InterpretationResponse, tags=["Interpreter"])
@limiter.limit("5/minute")
async def interpret_document(request: Request, payload: TextInput, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token di autenticazione mancante.")

    # ... la logica di validazione del token e recupero profilo è IDENTICA a /validate ...
    clerk_jwt_token_string = authorization.split(" ")[1]
    user_id = None
    try:
        # Logica di validazione token...
        jwks_response = requests.get(CLERK_JWKS_URL)
        jwks_response.raise_for_status()
        jwks_data = jwks_response.json()
        header = jwt.get_unverified_header(clerk_jwt_token_string)
        public_key = None
        for key_data in jwks_data["keys"]:
            if key_data["kid"] == header["kid"]:
                public_key = jwk.construct(key_data)
                break
        if not public_key: raise Exception("Chiave pubblica non trovata.")
        decoded_token = jwt.decode(clerk_jwt_token_string, public_key, algorithms=["RS256"], options={"verify_signature": True, "verify_aud": False, "verify_iss": False})
        user_id = decoded_token.get("sub")
        if not user_id: raise Exception("ID utente non trovato.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Validazione token fallita: {str(e)}")

    profile_res = supabase.table('profiles').select('*').eq('id', user_id).execute()
    if not profile_res.data:
        raise HTTPException(status_code=500, detail="Profilo utente non trovato.")
    profile = profile_res.data[0]

    # --- NUOVA LOGICA DI GESTIONE PIANI PER INTERPRETER ---
    user_tier_name = profile.get('subscription_tier', 'free')
    user_role = profile.get('role', 'user')
    plan = PLANS.get("admin") if user_role == 'admin' else PLANS.get(user_tier_name, PLANS["free"])
    # === INIZIO BLOCCO DA AGGIUNGERE ===
    # 0. Verifica lunghezza massima dell'input
    max_length = plan.get("max_input_length")
    if max_length is not None and len(payload.text) > max_length:
        raise HTTPException(
            status_code=413, # 413 Payload Too Large
            detail=f"Il documento inserito ({len(payload.text)} caratteri) supera il limite di {max_length} caratteri consentito per il tuo piano. Esegui l'upgrade per analizzare documenti più lunghi."
        )
    # === FINE BLOCCO DA AGGIUNGERE ===
    interpreter_plan = plan["interpreter"]
    if user_tier_name == 'free':
        model_to_use = ai_core.VALIDATOR_MODEL_NAME # Modello economico
    else:
        model_to_use = ai_core.INTERPRETER_MODEL_NAME # Modello potente
    # 1. Verifica profilo consentito
    if interpreter_plan["allowed_profiles"] != "all" and payload.profile_name not in interpreter_plan["allowed_profiles"]:
        raise HTTPException(status_code=403, detail=f"Il profilo Interpreter '{payload.profile_name}' non è incluso nel tuo piano.")

    # 2. Verifica limite di chiamate condiviso (identica a /validate)
    shared_limit = plan["shared_limit"]
    current_count = profile.get('usage_count', 0)
    if shared_limit != -1:
        today = str(date.today())
        if profile.get('last_used_date') != today:
            current_count = 0
            supabase.table('profiles').update({'last_used_date': today, 'usage_count': 0}).eq('id', user_id).execute()
        if current_count >= shared_limit:
            raise HTTPException(status_code=429, detail=f"Hai superato il limite giornaliero condiviso di {shared_limit} chiamate.")

    # --- ELABORAZIONE AI con le nuove funzioni di ai_core ---
    try:
        interpreted_text = await ai_core.interpret_text(payload.text, profile_name=payload.profile_name, model_name=model_to_use)
        
        quality_report_obj = None
        if interpreter_plan["quality_check"]:
            quality_report_data = await ai_core.get_interpreter_quality_score(original_text=payload.text, interpreted_text=interpreted_text, profile_name=payload.profile_name, model_name=model_to_use)
            if "error" not in quality_report_data and "human_quality_score" in quality_report_data:
                # ARROTONDA IL PUNTEGGIO ALL'INTERO PIÙ VICINO
                score = quality_report_data.get("human_quality_score", 0)
                quality_report_data["human_quality_score"] = round(score)
                quality_report_obj = QualityReport(**quality_report_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'elaborazione AI: {str(e)}")

    # --- AGGIORNAMENTO CONTEGGIO ---
    new_count = current_count + 1
    if shared_limit != -1:
        supabase.table('profiles').update({'usage_count': new_count}).eq('id', user_id).execute()

    return InterpretationResponse(
        interpreted_text=interpreted_text.strip(),
        quality_report=quality_report_obj,
        usage=UsageInfo(count=new_count, limit=shared_limit)
    )


@app.post("/compliance-check", response_model=ComplianceResponse, tags=["Compliance Checkr"])
@limiter.limit("5/minute")
async def compliance_check(request: Request, payload: TextInput, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token di autenticazione mancante.")

    # ... la logica di validazione del token e recupero profilo è IDENTICA a /validate ...
    clerk_jwt_token_string = authorization.split(" ")[1]
    user_id = None
    try:
        # Logica di validazione token...
        jwks_response = requests.get(CLERK_JWKS_URL)
        jwks_response.raise_for_status()
        jwks_data = jwks_response.json()
        header = jwt.get_unverified_header(clerk_jwt_token_string)
        public_key = None
        for key_data in jwks_data["keys"]:
            if key_data["kid"] == header["kid"]:
                public_key = jwk.construct(key_data)
                break
        if not public_key: raise Exception("Chiave pubblica non trovata.")
        decoded_token = jwt.decode(clerk_jwt_token_string, public_key, algorithms=["RS256"], options={"verify_signature": True, "verify_aud": False, "verify_iss": False})
        user_id = decoded_token.get("sub")
        if not user_id: raise Exception("ID utente non trovato.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Validazione token fallita: {str(e)}")

    profile_res = supabase.table('profiles').select('*').eq('id', user_id).execute()
    if not profile_res.data:
        raise HTTPException(status_code=500, detail="Profilo utente non trovato.")
    profile = profile_res.data[0]
    
    # --- LOGICA DI GESTIONE PIANI PER COMPLIANCE CHECKR ---
    user_tier_name = profile.get('subscription_tier', 'free')
    user_role = profile.get('role', 'user')
    plan = PLANS.get("admin") if user_role == 'admin' else PLANS.get(user_tier_name, PLANS["free"])
    
    if not plan["compliance_checkr"]["enabled"]:
        raise HTTPException(status_code=403, detail="Il Compliance Checkr non è incluso nel tuo piano.")
    # 0. Verifica lunghezza massima dell'input
    max_length = plan.get("max_input_length")
    if max_length is not None and len(payload.text) > max_length:
        raise HTTPException(
            status_code=413, # 413 Payload Too Large
            detail=f"Il documento inserito ({len(payload.text)} caratteri) supera il limite di {max_length} caratteri consentito per il tuo piano. Esegui l'upgrade per analizzare documenti più lunghi."
        )
    # === FINE BLOCCO DA AGGIUNGERE ===
    # 2. Verifica limite di chiamate condiviso (identica a /validate)
    shared_limit = plan["shared_limit"]
    current_count = profile.get('usage_count', 0)
    if shared_limit != -1:
        today = str(date.today())
        if profile.get('last_used_date') != today:
            current_count = 0
            supabase.table('profiles').update({'last_used_date': today, 'usage_count': 0}).eq('id', user_id).execute()
        if current_count >= shared_limit:
            raise HTTPException(status_code=429, detail=f"Hai superato il limite giornaliero condiviso di {shared_limit} chiamate.")
    try:
        compliance_report_text = await ai_core.check_compliance(payload.text, profile_name=payload.profile_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'analisi di conformità: {str(e)}")
    
    # --- AGGIORNAMENTO CONTEGGIO ---
    new_count = current_count + 1
    if shared_limit != -1:
        supabase.table('profiles').update({'usage_count': new_count}).eq('id', user_id).execute()

    return ComplianceResponse(
            compliance_report=compliance_report_text.strip(),
            usage=UsageInfo(count=new_count, limit=shared_limit)
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

@app.post("/ctov-profiles", response_model=CTOVProfileResponse, tags=["Custom Tone of Voice"])
async def create_ctov_profile(payload: CTOVProfileCreate, authorization: str = Header(None)):
    # 1. Autenticazione e recupero profilo/piano (codice standard)
    user_id, profile = await get_user_profile_from_token(authorization)
    user_tier_name = profile.get('subscription_tier', 'free')
    user_role = profile.get('role', 'user')
    plan = PLANS.get("admin") if user_role == 'admin' else PLANS.get(user_tier_name, PLANS["free"])

    # 2. Verifica se la funzionalità è abilitata per il piano
    ctov_plan = plan["ctov"]
    if not ctov_plan["enabled"]:
        raise HTTPException(status_code=403, detail="La creazione di Voci Personalizzate non è inclusa nel tuo piano.")

    # 3. Verifica il limite massimo di profili
    max_profiles = ctov_plan["max_profiles"]
    if max_profiles != -1:
        count_res = supabase.table('ctov_profiles').select('id', count='exact').eq('user_id', user_id).execute()
        if count_res.count is not None and count_res.count >= max_profiles:
            raise HTTPException(status_code=403, detail=f"Hai raggiunto il limite di {max_profiles} Voci Personalizzate per il tuo piano.")

    # 4. Inserimento nel database
    try:
        insert_data = payload.dict()
        insert_data['user_id'] = user_id
        res = supabase.table('ctov_profiles').insert(insert_data).execute()
        if not res.data:
            raise Exception("Creazione profilo CTOV fallita")
        # Converte l'UUID in stringa per la risposta
        created_profile = res.data[0]
        created_profile['id'] = str(created_profile['id'])
        return CTOVProfileResponse(**created_profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno durante la creazione del profilo: {str(e)}")


@app.get("/ctov-profiles", response_model=List[CTOVProfileResponse], tags=["Custom Tone of Voice"])
async def get_ctov_profiles(authorization: str = Header(None)):
    user_id, _ = await get_user_profile_from_token(authorization)
    res = supabase.table('ctov_profiles').select('*').eq('user_id', user_id).order('created_at').execute()
    return [CTOVProfileResponse(id=str(p['id']), **p) for p in res.data]

@app.put("/ctov-profiles/{profile_id}", response_model=CTOVProfileResponse, tags=["Custom Tone of Voice"])
async def update_ctov_profile(profile_id: str, payload: CTOVProfileCreate, authorization: str = Header(None)):
    user_id, _ = await get_user_profile_from_token(authorization)
    
    try:
        # L'update su Supabase include un .eq('user_id', user_id) per sicurezza:
        # l'utente può modificare solo un profilo che gli appartiene.
        update_data = payload.dict(exclude_unset=True)
        res = supabase.table('ctov_profiles').update(update_data).eq('id', profile_id).eq('user_id', user_id).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Profilo non trovato o non autorizzato.")
            
        updated_profile = res.data[0]
        updated_profile['id'] = str(updated_profile['id'])
        return CTOVProfileResponse(**updated_profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno durante l'aggiornamento: {str(e)}")


@app.delete("/ctov-profiles/{profile_id}", status_code=204, tags=["Custom Tone of Voice"])
async def delete_ctov_profile(profile_id: str, authorization: str = Header(None)):
    user_id, _ = await get_user_profile_from_token(authorization)
    
    try:
        # Anche il delete include il controllo su user_id.
        res = supabase.table('ctov_profiles').delete().eq('id', profile_id).eq('user_id', user_id).execute()
        
        if not res.data:
            # Se nessun dato viene restituito, significa che il record non esisteva o l'utente non aveva i permessi.
            raise HTTPException(status_code=404, detail="Profilo non trovato o non autorizzato.")
        
        return None # Ritorna una risposta 204 No Content in caso di successo
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno durante l'eliminazione: {str(e)}")

# === FINE BLOCCO ENDPOINT CTOV ===


# Funzione helper da aggiungere per non ripetere il codice di autenticazione
async def get_user_profile_from_token(authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token di autenticazione mancante.")
    
    clerk_jwt_token_string = authorization.split(" ")[1]
    user_id = None
    try:
        # Logica di validazione token...
        jwks_response = requests.get(CLERK_JWKS_URL)
        jwks_response.raise_for_status()
        jwks_data = jwks_response.json()
        header = jwt.get_unverified_header(clerk_jwt_token_string)
        public_key = None
        for key_data in jwks_data["keys"]:
            if key_data["kid"] == header["kid"]:
                public_key = jwk.construct(key_data)
                break
        if not public_key: raise Exception("Chiave pubblica non trovata.")
        decoded_token = jwt.decode(clerk_jwt_token_string, public_key, algorithms=["RS256"], options={"verify_signature": True, "verify_aud": False, "verify_iss": False})
        user_id = decoded_token.get("sub")
        if not user_id: raise Exception("ID utente non trovato.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Validazione token fallita: {str(e)}")

    profile_res = supabase.table('profiles').select('*').eq('id', user_id).execute()
    if not profile_res.data:
        raise HTTPException(status_code=500, detail="Profilo utente non trovato.")
    profile = profile_res.data[0]
    
    return user_id, profile

@app.get("/user-status", response_model=UserStatusResponse, tags=["User Management"])
@limiter.limit("50/minute")
async def get_user_status(request: Request, authorization: str = Header(None)):
    # ... la logica di validazione del token e recupero profilo rimane IDENTICA ...
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token di autenticazione mancante.")
    
    clerk_jwt_token_string = authorization.split(" ")[1]
    user_id = None
    try:
        # Logica di validazione token...
        jwks_response = requests.get(CLERK_JWKS_URL)
        jwks_response.raise_for_status()
        jwks_data = jwks_response.json()
        header = jwt.get_unverified_header(clerk_jwt_token_string)
        public_key = None
        for key_data in jwks_data["keys"]:
            if key_data["kid"] == header["kid"]:
                public_key = jwk.construct(key_data)
                break
        if not public_key: raise Exception("Chiave pubblica non trovata.")
        decoded_token = jwt.decode(clerk_jwt_token_string, public_key, algorithms=["RS256"], options={"verify_signature": True, "verify_aud": False, "verify_iss": False})
        user_id = decoded_token.get("sub")
        if not user_id: raise Exception("ID utente non trovato.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Validazione token fallita: {str(e)}")

    profile_res = supabase.table('profiles').select('*').eq('id', user_id).execute()
    if not profile_res.data:
        raise HTTPException(status_code=500, detail="Profilo utente non trovato.")
    profile = profile_res.data[0]

    # --- LOGICA AGGIORNATA PER RESTITUIRE I PERMESSI DETTAGLIATI ---
    user_tier_name = profile.get('subscription_tier', 'free')
    user_role = profile.get('role', 'user')
    plan = PLANS.get("admin") if user_role == 'admin' else PLANS.get(user_tier_name, PLANS["free"])
    ctov_res = supabase.table('ctov_profiles').select('*').eq('user_id', user_id).execute()
    #ctov_profiles_data = [CTOVProfileResponse(**p, id=str(p['id'])) for p in ctov_res.data]
    
    ctov_profiles_data = []
    for p in ctov_res.data:
        profile_data = p.copy()  # Crea una copia per sicurezza
        profile_data['id'] = str(profile_data['id'])  # Sovrascrive l'id con la sua versione stringa
        ctov_profiles_data.append(CTOVProfileResponse(**profile_data))
    
    return UserStatusResponse(
        usage=UsageInfo(count=profile.get('usage_count', 0), limit=plan["shared_limit"]),
        tier=user_tier_name if user_role != 'admin' else 'admin',
        validator_profiles=plan["validator"]["allowed_profiles"],
        interpreter_profiles=plan["interpreter"]["allowed_profiles"],
        compliance_access=plan["compliance_checkr"]["enabled"],
        strategist_access=plan["strategist"]["enabled"],
        ctov_access=plan["ctov"]["enabled"],
        ctov_max_profiles=plan["ctov"]["max_profiles"],
        ctov_profiles=ctov_profiles_data
    )


if __name__ == "__main__":
    # Legge la porta dalla variabile d'ambiente PORT fornita da Cloud Run
    # Se non la trova, usa 8000 come default (per lo sviluppo locale)
    port = int(os.environ.get("PORT", 8000))
    
    # Avvia il server uvicorn programmaticamente
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)