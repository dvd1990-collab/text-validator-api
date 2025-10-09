import os
import uvicorn
import redis
from fastapi import Request, FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- NUOVA IMPORTAZIONE PER IL CORS ---
from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
 
import ai_core

# --- Pydantic Models ---
class TextInput(BaseModel):
    text: str = Field(..., min_length=10)

class QualityReport(BaseModel):
    reasoning: str
    human_quality_score: int

class ValidationResponse(BaseModel):
    normalized_text: str
    quality_report: QualityReport

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Text Validator API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CONFIGURAZIONE CORS ---
# Definiamo da quali "origini" (domini) il nostro backend accetterà richieste.
# Per lo sviluppo, ci basta accettare richieste dal nostro server frontend.
origins = [
    "http://localhost:3000", # Per lo sviluppo locale
    "https://text-validator-frontend-ne0m6kwvq-davides-projects-97f15092.vercel.app", # L'URL di produzione
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
async def validate_text(request: Request, payload: TextInput):
    """
    Questo endpoint accetta un testo grezzo ed esegue il workflow completo di validazione.
    """
    
    normalized_text = await ai_core.normalize_text(payload.text)
    if not isinstance(normalized_text, str):
        raise HTTPException(status_code=500, detail="Errore interno: La Fase 1 non ha restituito testo valido.")

    quality_report_data = await ai_core.get_quality_score(
        original_text=payload.text,
        normalized_text=normalized_text
    )
    if not isinstance(quality_report_data, dict) or "reasoning" not in quality_report_data:
        raise HTTPException(status_code=500, detail="Errore interno: La Fase 2 non ha restituito un report valido.")

    return ValidationResponse(
        normalized_text=normalized_text.strip(),
        quality_report=QualityReport(**quality_report_data)
    )

if __name__ == "__main__":
    # Legge la porta dalla variabile d'ambiente PORT fornita da Cloud Run
    # Se non la trova, usa 8000 come default (per lo sviluppo locale)
    port = int(os.environ.get("PORT", 8000))
    
    # Avvia il server uvicorn programmaticamente
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)