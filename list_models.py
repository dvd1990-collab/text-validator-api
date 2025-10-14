# list_models.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carica le variabili d'ambiente (per prendere la GOOGLE_API_KEY)
load_dotenv()

# Configura l'API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY non trovata nel file .env")

genai.configure(api_key=API_KEY)

print("--- Elenco dei Modelli Gemini che supportano 'generateContent' ---\n")

for m in genai.list_models():
  # Controlla se il metodo 'generateContent' è tra i metodi supportati dal modello
  if 'generateContent' in m.supported_generation_methods:
    print(f"Nome Modello: {m.name}")
    print(f"  Descrizione: {m.description}")
    print(f"  Limite Token Input: {m.input_token_limit}")
    print("-" * 20)

print("\n--- Fine Elenco ---")
print("Copia il 'Nome Modello' esatto (es. 'gemini-1.5-flash-latest') e usalo nel file ai_core.py")