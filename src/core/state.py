from typing import TypedDict, Optional
from pydantic import BaseModel, Field

# ==========================================
# 1. PYDANTIC MODELS (Data Validation)
# ==========================================

class StructuralParameters(BaseModel):
    """Hard constraints estratti dall'annuncio."""
    price: Optional[float] = None
    price_per_sqm: Optional[float] = None
    sqm: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floor: Optional[str] = None
    has_elevator: Optional[bool] = None
    has_architectural_barriers: Optional[bool] = None

class CommuteData(BaseModel):
    """Dati logistici calcolati dall'agente mappe."""
    transit_time_mins: Optional[int] = None
    distance_km: Optional[float] = None

class OsintData(BaseModel):
    """Dati ambientali e di infrastruttura."""
    broadband_type: Optional[str] = None  # es. FTTH, FTTC, 5G
    safety_score: Optional[float] = Field(None, ge=-1.0, le=1.0) # Normalizzato tra -1 e 1
    poi_count: Optional[int] = None

# ==========================================
# 2. LANGGRAPH STATE (The Global State)
# ==========================================

class PropertyState(TypedDict):
    """
    Questo è lo stato globale che viaggerà di nodo in nodo
    durante l'esecuzione del nostro grafo ciclico.
    """
    property_url: str
    raw_listing_text: str
    
    # Dati strutturati usando i modelli Pydantic
    extracted_parameters: StructuralParameters
    commute_data: CommuteData
    osint_data: OsintData
    
    # Variabili di controllo e output
    hard_constraints_met: bool  # Il nostro interruttore Go/No-Go
    final_score: float
    evaluation_report: str