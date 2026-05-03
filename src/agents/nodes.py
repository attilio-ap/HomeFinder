import os
import requests
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.core.state import (
    CommuteData,
    OsintData,
    PropertyState,
    StructuralParameters,
)


def data_extractor_node(state: PropertyState) -> dict:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
        structured_llm = llm.with_structured_output(StructuralParameters)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sei un analista immobiliare. Estrai i dati in modo oggettivo dal testo dell'annuncio. Converti i piani in numeri (es. piano terra = 0)."),
            ("user", "{text}")
        ])

        chain = prompt | structured_llm
        
        raw_listing_text = state.get("raw_listing_text", "") if isinstance(state, dict) else getattr(state, "raw_listing_text", "")

        extracted_data = chain.invoke({"text": raw_listing_text})

        if getattr(extracted_data, "price", None) is not None and extracted_data.price > 300000:
            is_go = False
        else:
            is_go = True

        return {
            "extracted_parameters": extracted_data,
            "hard_constraints_met": is_go
        }
    except Exception as e:
        print(f"   ❌ ERRORE CRITICO: {e}") 
        return {"hard_constraints_met": False}


def commuter_node(state: PropertyState) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("Errore: GOOGLE_MAPS_API_KEY non trovata. Restituisco dati mockati.")
        return {
            "commute_data": CommuteData(
                transit_time_mins=35,
                distance_km=12.5,
            )
        }

    origin = "Quartiere Isola, Milano"
    destination = "Piazza del Duomo, Milano"
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "key": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        element = data["rows"][0]["elements"][0]
        transit_time_mins = int(element["duration"]["value"] / 60)
        distance_km = float(element["distance"]["value"] / 1000)

        return {
            "commute_data": CommuteData(
                transit_time_mins=transit_time_mins,
                distance_km=distance_km,
            )
        }
    except Exception as e:
        print(f"Errore durante la richiesta a Google Maps: {e}. Restituisco dati mockati.")
        return {
            "commute_data": CommuteData(
                transit_time_mins=35,
                distance_km=12.5,
            )
        }


def osint_node(state: PropertyState) -> Dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Errore: TAVILY_API_KEY non trovata. Restituisco dati mockati.")
        return {
            "osint_data": OsintData(
                broadband_type="FTTH",
                safety_score=0.8,
                poi_count=5,
            )
        }

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": "Quartiere Isola Milano sicurezza, criminalità, fibra ottica FTTH",
        "search_depth": "basic",
        "max_results": 3
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        combined_text = " ".join([result.get("content", "") for result in results]).lower()

        safety_score = 0.5
        if "criminalità" in combined_text or "furt" in combined_text:
            safety_score = -0.5
        elif "sicur" in combined_text or "tranquill" in combined_text:
            safety_score = 0.8
            
        broadband_type = "Misto"
        if "ftth" in combined_text or "fibra" in combined_text:
            broadband_type = "FTTH"
            
        return {
            "osint_data": OsintData(
                broadband_type=broadband_type,
                safety_score=safety_score,
                poi_count=15
            )
        }
    except Exception as e:
        print(f"Errore durante la richiesta a Tavily: {e}. Restituisco dati mockati.")
        return {
            "osint_data": OsintData(
                broadband_type="FTTH",
                safety_score=0.8,
                poi_count=5,
            )
        }


def evaluator_node(state: PropertyState) -> Dict[str, Any]:
    """
    Valuta la proprietà complessiva in base ai dati raccolti.
    Per ora, restituisce un punteggio e un report mockati.
    """
    report = """# Report Valutazione Immobile (Mock)

## Riepilogo
L'immobile soddisfa i requisiti minimi e si trova in un'ottima zona.

- **Prezzo:** 250.000 €
- **Superficie:** 85 mq
- **Distanza lavoro:** 35 minuti (12.5 km)
- **Connettività:** FTTH

**Punteggio finale:** 85.5 / 100
"""
    return {
        "final_score": 85.5,
        "evaluation_report": report,
    }
