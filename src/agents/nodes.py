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


def scraper_node(state: PropertyState) -> dict:
    print("⏳ [Scraper Agent] Recupero testo dell'annuncio dal web tramite Jina Reader...")
    url = state.get('target_url')
    if not url:
        print("❌ [Scraper Agent] target_url mancante nello state.")
        return {"raw_listing_text": "ERROR:SCRAPING_BLOCKED"}
    try:
        jina_url = f"https://r.jina.ai/{url}"
        response = requests.get(jina_url)
        response.raise_for_status()
        text = response.text
        print("✅ [Scraper Agent] Testo estratto con successo dalla pagina!")
        return {"raw_listing_text": text}
    except Exception as e:
        print(f"❌ [Scraper Agent] Errore durante la GET request: {e}.")
        return {"raw_listing_text": "ERROR:SCRAPING_BLOCKED"}


def data_extractor_node(state: PropertyState) -> dict:
    print("⏳ [Data Extractor] Analisi dell'annuncio in corso...")
    try:
        raw_listing_text = state.get("raw_listing_text", "") if isinstance(state, dict) else getattr(state, "raw_listing_text", "")
        
        if "ERROR:SCRAPING_BLOCKED" in raw_listing_text or "403" in raw_listing_text or "Forbidden" in raw_listing_text or len(raw_listing_text) < 50:
            print("❌ [Data Extractor] Testo invalido o bloccato (ERROR/403/Forbidden). Interruzione estrazione.")
            return {"extracted_parameters": None, "hard_constraints_met": False}

        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
        structured_llm = llm.with_structured_output(StructuralParameters)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sei un analista immobiliare. Estrai i dati in modo oggettivo dal testo dell'annuncio. Assicurati di estrarre l'indirizzo esatto o la via dell'immobile in Milano (property_address). Converti i piani in numeri (es. piano terra = 0)."),
            ("user", "{text}")
        ])

        chain = prompt | structured_llm

        extracted_data = chain.invoke({"text": raw_listing_text})

        budget = state.get('max_budget')
        if budget is not None and getattr(extracted_data, "price", None) is not None and extracted_data.price <= budget:
            is_go = True
        else:
            is_go = False

        print("✅ [Data Extractor] Dati estratti con successo!")
        return {
            "extracted_parameters": extracted_data,
            "hard_constraints_met": is_go
        }
    except Exception as e:
        print(f"   ❌ ERRORE CRITICO: {e}") 
        print("❌ [Data Extractor] Errore API, restituisco None.")
        return {"extracted_parameters": None, "hard_constraints_met": False}


def commuter_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [Commuter Agent] Calcolo del percorso su Google Maps...")
    extracted = state.get('extracted_parameters')
    if extracted is None:
        print("❌ [Commuter Agent] extracted_parameters mancante. Restituisco None.")
        return {
            "commute_data": None
        }

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("❌ [Commuter Agent] Errore: GOOGLE_MAPS_API_KEY non trovata. Restituisco None.")
        return {
            "commute_data": None
        }

    origin = getattr(extracted, 'property_address', 'Milano') if extracted else 'Milano'
    destination = state.get('user_office_address', 'Milano')

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
        print(f"❌ [Commuter Agent] Errore durante la richiesta a Google Maps: {e}. Restituisco None.")
        return {
            "commute_data": None
        }


def osint_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [OSINT Agent] Ricerca informazioni sul quartiere tramite Tavily...")
    extracted = state.get('extracted_parameters')
    if extracted is None:
        print("❌ [OSINT Agent] extracted_parameters mancante. Restituisco None.")
        return {
            "osint_data": None
        }

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ [OSINT Agent] Errore: TAVILY_API_KEY non trovata. Restituisco None.")
        return {
            "osint_data": None
        }

    address = getattr(extracted, 'property_address', 'Milano') if extracted else 'Milano'
    query = f"{address} Milano sicurezza, criminalità, fibra ottica FTTH"

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
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
        print(f"❌ [OSINT Agent] Errore durante la richiesta a Tavily: {e}. Restituisco None.")
        return {
            "osint_data": None
        }


def evaluator_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [Evaluator Agent] Elaborazione del punteggio e stesura del report finale...")
    try:
        params = state.get('extracted_parameters')
        if params is None:
            return {
                "final_score": 0.0,
                "evaluation_report": "⚠️ Analisi Fallita: Non è stato possibile recuperare i dati dell'annuncio. Per procedere, utilizza il campo di Fallback nella UI e incolla il testo dell'annuncio manualmente."
            }

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
        print(f"❌ [Evaluator Agent] Errore API: {e}")
        return {
            "final_score": 0.0,
            "evaluation_report": f"⚠️ Errore durante la valutazione: {e}"
        }


def financial_node(state: PropertyState) -> dict:
    print("⏳ [Financial Agent] Calcolo ammortamento e rata...")
    extracted = state.get('extracted_parameters')
    if extracted is None:
        return {}
    
    price = getattr(extracted, 'price', None)
    if price is None:
        return {}

    down_payment = state.get('down_payment', 0.0)
    interest_rate = state.get('interest_rate', 0.0)
    loan_term_years = state.get('loan_term_years', 0)

    P = price - down_payment
    r = interest_rate / 12
    n = loan_term_years * 12

    def calculate_installment(principal, rate, months):
        if principal <= 0 or months <= 0:
            return 0.0
        if rate == 0:
            return principal / months
        return principal * (rate * (1 + rate)**months) / (((1 + rate)**months) - 1)

    M1 = calculate_installment(P, r, n)
    
    discounted_price = price * 0.88
    P2 = discounted_price - down_payment
    M2 = calculate_installment(P2, r, n)

    print("✅ [Financial Agent] Calcoli completati!")
    return {
        "financial_data": {
            "original_price": price,
            "original_installment": M1,
            "discounted_price": discounted_price,
            "discounted_installment": M2,
            "discount_percentage": 12
        }
    }


def negotiator_node(state: PropertyState) -> dict:
    print("⏳ [Negotiator Agent] Generazione email di negoziazione...")
    evaluation_report = state.get('evaluation_report')
    financial_data = state.get('financial_data')
    
    if not evaluation_report or not financial_data:
        return state
        
    raw_listing_text = state.get('raw_listing_text', '')
    
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.4)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un negoziatore immobiliare esperto."),
        ("user", "Testo dell'annuncio: {listing}\n\nReport di valutazione: {report}\n\nDati finanziari: {finance}\n\nScrivi un'email formale, cortese ma decisa, all'agenzia immobiliare. L'email deve proporre come offerta il discounted_price indicato nei dati finanziari, giustificando la richiesta di sconto facendo leva esclusivamente sui difetti reali emersi dal report (es. ristrutturazione necessaria, assenza ascensore, zona, ecc.).")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({
            "listing": raw_listing_text,
            "report": evaluation_report,
            "finance": financial_data
        })
        print("✅ [Negotiator Agent] Email generata con successo!")
        return {"negotiation_email": response.content}
    except Exception as e:
        print(f"❌ [Negotiator Agent] Errore API: {e}")
        return state
