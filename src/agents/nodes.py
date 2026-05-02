import os
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
    """
    Calcola i tempi di percorrenza e la distanza per il pendolarismo.
    Per ora, restituisce dati mockati.
    """
    return {
        "commute": CommuteData(
            transit_time_mins=35,
            distance_km=12.5,
        )
    }


def osint_node(state: PropertyState) -> Dict[str, Any]:
    """
    Recupera dati sulla zona (internet, sicurezza, punti di interesse).
    Per ora, restituisce dati mockati.
    """
    return {
        "osint": OsintData(
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
