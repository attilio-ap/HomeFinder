from typing import Optional, TypedDict

from pydantic import BaseModel, Field

# ==========================================
# 1. PYDANTIC MODELS (Data Validation)
# ==========================================


class StructuralParameters(BaseModel):
    """Hard constraints extracted from the property listing.

    Attributes:
        property_address (str): Exact address or street of the property.
        price (Optional[float]): Asking price of the property.
        price_per_sqm (Optional[float]): Calculated price per square meter.
        sqm (Optional[int]): Square footage of the property.
        bedrooms (Optional[int]): Number of bedrooms.
        bathrooms (Optional[int]): Number of bathrooms.
        floor (Optional[str]): Floor level of the property.
        energy_class (Optional[str]): Energy class from A to G.
        has_elevator (Optional[bool]): Whether the building has an elevator.
        has_architectural_barriers (Optional[bool]): Presence of architectural barriers.
    """

    property_address: str = Field(
        description="Exact address or street of the property extracted from the text"
    )
    price: Optional[float] = None
    price_per_sqm: Optional[float] = None
    sqm: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floor: Optional[str] = None
    energy_class: Optional[str] = Field(None, description="Energy class from A to G")
    has_elevator: Optional[bool] = None
    has_architectural_barriers: Optional[bool] = None


class CommuteData(BaseModel):
    """Logistical data calculated by the maps agent.

    Attributes:
        transit_time_mins (Optional[int]): Estimated commute time in minutes.
        distance_km (Optional[float]): Distance in kilometers.
    """

    transit_time_mins: Optional[int] = None
    distance_km: Optional[float] = None


class OsintData(BaseModel):
    """Environmental and infrastructure data for the neighborhood.

    Attributes:
        broadband_type (Optional[str]): Predominant connection type (e.g., FTTH, FTTC).
        safety_score (Optional[float]): Normalized safety score between -1.0 and 1.0.
        noise_level (Optional[float]): Normalized noise level between -1.0 and 1.0.
        public_transport_score (Optional[float]): Accessibility score between 0.0 and 1.0.
        amenities_score (Optional[float]): Local amenities score between 0.0 and 1.0.
        poi_count (Optional[int]): Count of nearby points of interest.
    """

    broadband_type: Optional[str] = None  # e.g., FTTH, FTTC, 5G
    safety_score: Optional[float] = Field(None, ge=-1.0, le=1.0)  # Normalized between -1 and 1
    noise_level: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Noise level: -1.0 (very noisy) to 1.0 (very quiet)"
    )
    public_transport_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Proximity to public transport: 0.0 to 1.0"
    )
    amenities_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Proximity to services/amenities: 0.0 to 1.0"
    )
    poi_count: Optional[int] = None


# ==========================================
# 2. LANGGRAPH STATE (The Global State)
# ==========================================


class PropertyState(TypedDict):
    """Global state for the LangGraph execution flow.

    Attributes:
        target_url (str): The URL of the property listing to analyze.
        user_office_address (str): The user's office address for commute calculations.
        max_budget (float): Maximum budget for the property purchase.
        property_url (str): Duplicate of target_url for internal use.
        raw_listing_text (str): Raw text extracted from the listing.
        down_payment (float): Cash down payment for mortgage calculations.
        interest_rate (float): Annual interest rate for the loan.
        loan_term_years (int): Duration of the loan in years.
        extracted_parameters (StructuralParameters): Validated property features.
        commute_data (CommuteData): Calculated logistical information.
        osint_data (OsintData): Neighborhood and infrastructure insights.
        hard_constraints_met (bool): Flag indicating if basic criteria are satisfied.
        final_score (float): Calculated overall score for the property.
        evaluation_report (str): Summarized analysis of the property.
        financial_data (dict): Detailed mortgage and financial breakdown.
        negotiation_email (str): Drafted negotiation email for the user.
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
