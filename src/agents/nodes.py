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
    print("⏳ [Data Extractor] Analisi dell'annuncio in corso...")
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

        print("✅ [Data Extractor] Dati estratti con successo!")
        return {
            "extracted_parameters": extracted_data,
            "hard_constraints_met": is_go
        }
    except Exception as e:
        print(f"   ❌ ERRORE CRITICO: {e}") 
        print("❌ [Data Extractor] Errore API, uso dati di default.")
        return {"hard_constraints_met": False}


def commuter_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [Commuter Agent] Calcolo del percorso su Google Maps...")
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

        print(f"✅ [Commuter Agent] Percorso calcolato: {transit_time_mins} min, {distance_km} km")
        return {
            "commute_data": CommuteData(
                transit_time_mins=transit_time_mins,
                distance_km=distance_km,
            )
        }
    except Exception as e:
        print(f"Errore durante la richiesta a Google Maps: {e}. Restituisco dati mockati.")
        print("❌ [Commuter Agent] Errore API, uso dati di default.")
        return {
            "commute_data": CommuteData(
                transit_time_mins=35,
                distance_km=12.5,
            )
        }


def osint_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [OSINT Agent] Ricerca informazioni sul quartiere tramite Tavily...")
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
            
        print(f"✅ [OSINT Agent] Dati quartiere raccolti: Fibra {broadband_type}, Sicurezza {safety_score}")
        return {
            "osint_data": OsintData(
                broadband_type=broadband_type,
                safety_score=safety_score,
                poi_count=15
            )
        }
    except Exception as e:
        print(f"Errore durante la richiesta a Tavily: {e}. Restituisco dati mockati.")
        print("❌ [OSINT Agent] Errore API, uso dati di default.")
        return {
            "osint_data": OsintData(
                broadband_type="FTTH",
                safety_score=0.8,
                poi_count=5,
            )
        }


def evaluator_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [Evaluator Agent] Elaborazione del punteggio e stesura del report finale...")
    try:
        params = state.get('extracted_parameters')
        commute = state.get('commute_data')
        osint = state.get('osint_data')

        # STEP A: Algoritmo WSM (Weighted Sum Model)
        score_price = (300000 / params.price) * 100 if params and getattr(params, 'price', None) else 0
        if score_price > 100:
            score_price = 100
            
        score_sqm = (params.sqm / 80) * 100 if params and getattr(params, 'sqm', None) else 0
        if score_sqm > 100:
            score_sqm = 100

        transit_time_mins = commute.transit_time_mins if commute else 999
        if transit_time_mins < 30:
            score_commute = 100
        elif transit_time_mins < 45:
            score_commute = 70
        else:
            score_commute = 40

        broadband_type = osint.broadband_type if osint else ""
        safety_score = osint.safety_score if osint else 0
        score_osint = (50 if broadband_type == 'FTTH' else 0) + (safety_score * 50)

        final_score = round(
            (score_price * 0.40) +
            (score_sqm * 0.20) +
            (score_commute * 0.25) +
            (score_osint * 0.15),
            1
        )

        # STEP B: LLM Executive Summary
        llm_pro = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sei un analista immobiliare spietato e pragmatico. Scrivi un Executive Summary in Markdown (max 250 parole) valutando un immobile. Sii diretto, evidenzia i pro (es. fibra, vicinanza ufficio) e i contro (es. no ascensore, criminalità, prezzo). Non essere inutilmente entusiasta. Usa elenchi puntati per la leggibilità."),
            ("user", "Dati estratti: {params}\nDati tragitto: {commute}\nDati OSINT: {osint}\nPunteggio finale calcolato: {final_score}")
        ])

        chain = prompt | llm_pro
        response = chain.invoke({
            "params": params,
            "commute": commute,
            "osint": osint,
            "final_score": final_score
        })

        print("✅ [Evaluator Agent] Report generato con successo!")
        # STEP C: Return
        return {
            "final_score": final_score,
            "evaluation_report": response.content
        }

    except Exception as e:
        print("❌ [Evaluator Agent] Errore API, uso dati di default.")
        return {
            "final_score": 0.0,
            "evaluation_report": f"Errore durante la valutazione: {e}"
        }
