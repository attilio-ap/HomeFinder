import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carica le variabili dal file .env
load_dotenv()

# Recupera la API Key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Errore: GEMINI_API_KEY non trovata nel file .env")
else:
    # Configura la libreria con la tua chiave
    genai.configure(api_key=api_key)

    print("--- Modelli Gemini Disponibili ---\n")
    try:
        # Cicla tra i modelli disponibili
        for m in genai.list_models():
            # Filtra per mostrare solo quelli che supportano la generazione di contenuti
            if 'generateContent' in m.supported_generation_methods:
                print(f"ID: {m.name}")
                print(f"Descrizione: {m.description}\n")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")