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
    print("⏳ [Scraper Agent] Fetching listing text from the web via Jina Reader...")
    url = state.get('target_url')
    if not url:
        print("❌ [Scraper Agent] target_url missing in state.")
        return {"raw_listing_text": "ERROR:SCRAPING_BLOCKED"}
    try:
        jina_url = f"https://r.jina.ai/{url}"
        response = requests.get(jina_url)
        response.raise_for_status()
        text = response.text
        print("✅ [Scraper Agent] Text successfully extracted from page!")
        return {"raw_listing_text": text}
    except Exception as e:
        print(f"❌ [Scraper Agent] Error during GET request: {e}.")
        return {"raw_listing_text": "ERROR:SCRAPING_BLOCKED"}


def data_extractor_node(state: PropertyState) -> dict:
    print("⏳ [Data Extractor] Analyzing listing...")
    try:
        raw_listing_text = state.get("raw_listing_text", "") if isinstance(state, dict) else getattr(state, "raw_listing_text", "")
        
        if "ERROR:SCRAPING_BLOCKED" in raw_listing_text or "403" in raw_listing_text or "Forbidden" in raw_listing_text or len(raw_listing_text) < 50:
            print("❌ [Data Extractor] Invalid or blocked text (ERROR/403/Forbidden). Aborting extraction.")
            return {"extracted_parameters": None, "hard_constraints_met": False}

        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
        structured_llm = llm.with_structured_output(StructuralParameters)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a real estate analyst. Extract data objectively from the listing text. Make sure to extract the exact address or street of the property in Milan (property_address). Convert floors to numbers (e.g., ground floor = 0)."),
            ("user", "{text}")
        ])

        chain = prompt | structured_llm

        extracted_data = chain.invoke({"text": raw_listing_text})

        budget = state.get('max_budget')
        if budget is not None and getattr(extracted_data, "price", None) is not None and extracted_data.price <= budget:
            is_go = True
        else:
            is_go = False

        print("✅ [Data Extractor] Data successfully extracted!")
        return {
            "extracted_parameters": extracted_data,
            "hard_constraints_met": is_go
        }
    except Exception as e:
        print(f"   ❌ CRITICAL ERROR: {e}") 
        print("❌ [Data Extractor] API Error, returning None.")
        return {"extracted_parameters": None, "hard_constraints_met": False}


def commuter_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [Commuter Agent] Calculating route on Google Maps...")
    extracted = state.get('extracted_parameters')
    if extracted is None:
        print("❌ [Commuter Agent] extracted_parameters missing. Returning None.")
        return {
            "commute_data": None
        }

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("❌ [Commuter Agent] Error: GOOGLE_MAPS_API_KEY not found. Returning None.")
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

        print(f"✅ [Commuter Agent] Route calculated: {transit_time_mins} min, {distance_km} km")
        return {
            "commute_data": CommuteData(
                transit_time_mins=transit_time_mins,
                distance_km=distance_km,
            )
        }
    except Exception as e:
        print(f"❌ [Commuter Agent] Error during Google Maps request: {e}. Returning None.")
        return {
            "commute_data": None
        }


def osint_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [OSINT Agent] Researching neighborhood information via Tavily...")
    extracted = state.get('extracted_parameters')
    if extracted is None:
        print("❌ [OSINT Agent] extracted_parameters missing. Returning None.")
        return {
            "osint_data": None
        }

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ [OSINT Agent] Error: TAVILY_API_KEY not found. Returning None.")
        return {
            "osint_data": None
        }

    address = getattr(extracted, 'property_address', 'Milano') if extracted else 'Milano'
    query = f"{address} Milan safety, crime, FTTH optical fiber"

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
        if "criminalità" in combined_text or "crime" in combined_text or "furt" in combined_text or "theft" in combined_text:
            safety_score = -0.5
        elif "sicur" in combined_text or "safe" in combined_text or "tranquill" in combined_text or "quiet" in combined_text:
            safety_score = 0.8
            
        broadband_type = "Mixed"
        if "ftth" in combined_text or "fibra" in combined_text or "fiber" in combined_text:
            broadband_type = "FTTH"
            
        print(f"✅ [OSINT Agent] Neighborhood data collected: Fiber {broadband_type}, Safety {safety_score}")
        return {
            "osint_data": OsintData(
                broadband_type=broadband_type,
                safety_score=safety_score,
                poi_count=15
            )
        }
    except Exception as e:
        print(f"❌ [OSINT Agent] Error during Tavily request: {e}. Returning None.")
        return {
            "osint_data": None
        }


def evaluator_node(state: PropertyState) -> Dict[str, Any]:
    print("⏳ [Evaluator Agent] Processing score and drafting final report...")
    try:
        params = state.get('extracted_parameters')
        if params is None:
            return {
                "final_score": 0.0,
                "evaluation_report": "⚠️ Analysis Failed: Could not retrieve listing data. To proceed, use the Fallback field in the UI and paste the listing text manually."
            }

        commute = state.get('commute_data')
        osint = state.get('osint_data')

        # STEP A: WSM Algorithm (Weighted Sum Model)
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
            ("system", "You are a ruthless and pragmatic real estate analyst. Write an Executive Summary in Markdown (max 250 words) evaluating a property. Be direct, highlight pros (e.g., fiber, proximity to office) and cons (e.g., no elevator, crime, price). Do not be unnecessarily enthusiastic. Use bullet points for readability."),
            ("user", "Extracted data: {params}\nCommute data: {commute}\nOSINT data: {osint}\nCalculated final score: {final_score}")
        ])

        chain = prompt | llm_pro
        response = chain.invoke({
            "params": params,
            "commute": commute,
            "osint": osint,
            "final_score": final_score
        })

        print("✅ [Evaluator Agent] Report generated successfully!")
        # STEP C: Return
        return {
            "final_score": final_score,
            "evaluation_report": response.content
        }

    except Exception as e:
        print(f"❌ [Evaluator Agent] API Error: {e}")
        return {
            "final_score": 0.0,
            "evaluation_report": f"⚠️ Error during evaluation: {e}"
        }


def financial_node(state: PropertyState) -> dict:
    print("⏳ [Financial Agent] Calculating mortgage amortization and installment...")
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

    print("✅ [Financial Agent] Calculations completed!")
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
    print("⏳ [Negotiator Agent] Generating negotiation email...")
    evaluation_report = state.get('evaluation_report')
    financial_data = state.get('financial_data')
    
    if not evaluation_report or not financial_data:
        return state
        
    raw_listing_text = state.get('raw_listing_text', '')
    
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.4)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert real estate negotiator."),
        ("user", "Listing text: {listing}\n\nEvaluation report: {report}\n\nFinancial data: {finance}\n\nWrite a formal, polite but firm email to the real estate agency. The email must propose the discounted_price indicated in the financial data as an offer, justifying the request for a discount by leveraging exclusively the real flaws highlighted in the report (e.g., needed renovation, lack of elevator, neighborhood, etc.).")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({
            "listing": raw_listing_text,
            "report": evaluation_report,
            "finance": financial_data
        })
        print("✅ [Negotiator Agent] Email generated successfully!")
        return {"negotiation_email": response.content}
    except Exception as e:
        print(f"❌ [Negotiator Agent] API Error: {e}")
        return state
