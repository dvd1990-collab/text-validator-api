import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente per prendere GOOGLE_API_KEY dal file .env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERRORE: La chiave API di Google non è stata trovata nel file .env")
    exit()

try:
    genai.configure(api_key=API_KEY)

    print("--- Sto cercando i modelli disponibili che supportano 'generateContent' ---")
    
    found_models = False
    # Itera su tutti i modelli che l'API mette a disposizione
    for m in genai.list_models():
        # Controlliamo se il modello è in grado di generare contenuto testuale
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
            found_models = True

    if not found_models:
        print("\nATTENZIONE: Nessun modello trovato che supporti 'generateContent'.")
        print("Possibili cause:")
        print("1. La tua chiave API potrebbe non essere abilitata per la 'Generative Language API' o 'Vertex AI API' nel tuo progetto Google Cloud.")
        print("2. Potrebbero esserci restrizioni geografiche associate al tuo account Google Cloud.")

    print("-------------------------------------------------------------------------")

except Exception as e:
    print(f"\nSi è verificato un errore imprevisto durante la comunicazione con l'API di Google: {e}")