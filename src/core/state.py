from typing import TypedDict, Optional
from pydantic import BaseModel, Field

# ==========================================
# 1. PYDANTIC MODELS (Data Validation)
# ==========================================

class StructuralParameters(BaseModel):
    """Hard constraints extracted from the listing."""
    property_address: str = Field(description='Exact address or street of the property extracted from the text')
    price: Optional[float] = None
    price_per_sqm: Optional[float] = None
    sqm: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floor: Optional[str] = None
    energy_class: Optional[str] = Field(None, description='Energy class from A to G')
    has_elevator: Optional[bool] = None
    has_architectural_barriers: Optional[bool] = None

class CommuteData(BaseModel):
    """Logistical data calculated by the maps agent."""
    transit_time_mins: Optional[int] = None
    distance_km: Optional[float] = None

class OsintData(BaseModel):
    """Environmental and infrastructure data."""
    broadband_type: Optional[str] = None  # e.g., FTTH, FTTC, 5G
    safety_score: Optional[float] = Field(None, ge=-1.0, le=1.0) # Normalized between -1 and 1
    noise_level: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Noise level: -1.0 (very noisy) to 1.0 (very quiet)")
    public_transport_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Proximity to public transport: 0.0 to 1.0")
    amenities_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Proximity to services/amenities: 0.0 to 1.0")
    poi_count: Optional[int] = None

# ==========================================
# 2. LANGGRAPH STATE (The Global State)
# ==========================================

class PropertyState(TypedDict):
    """
    This is the global state that will travel from node to node
    during the execution of our cyclic graph.
    """
    target_url: str
    user_office_address: str
    max_budget: float
    property_url: str
    raw_listing_text: str
    
    # Financial inputs
    down_payment: float
    interest_rate: float
    loan_term_years: int
    
    # Structured data using Pydantic models
    extracted_parameters: StructuralParameters
    commute_data: CommuteData
    osint_data: OsintData
    
    # Control variables and outputs
    hard_constraints_met: bool  # Our Go/No-Go switch
    final_score: float
    evaluation_report: str
    
    # Financial and negotiation outputs
    financial_data: dict
    negotiation_email: str
