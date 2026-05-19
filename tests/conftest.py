from typing import Any

import pytest


@pytest.fixture
def base_state() -> dict[str, Any]:
    return {
        "target_url": "https://www.immobiliare.it/annunci/123456789/",
        "user_office_address": "Piazza del Duomo, Milan",
        "max_budget": 350000.0,
        "down_payment": 50000.0,
        "interest_rate": 0.035,
        "loan_term_years": 30,
        "negotiation_language": "Italian",
    }
