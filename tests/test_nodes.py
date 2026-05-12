import os
from typing import Any, Dict, cast
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.nodes import (
    OsintAnalysis,
    commuter_node,
    osint_node,
    scraper_node,
)
from src.core.state import PropertyState, StructuralParameters


@pytest.mark.asyncio
async def test_scraper_node_success(mocker: Any, base_state: Dict[str, Any]) -> None:
    # Arrange
    mock_fetch = mocker.patch("src.agents.nodes.fetch_jina_reader", new_callable=AsyncMock)
    mock_fetch.return_value = "Apartment for sale in Milan, Via Roma 1, 100sqm, 300000 euros"

    # Act
    result = await scraper_node(cast(PropertyState, base_state))

    # Assert
    assert (
        result["raw_listing_text"]
        == "Apartment for sale in Milan, Via Roma 1, 100sqm, 300000 euros"
    )
    mock_fetch.assert_called_once_with(base_state["target_url"])


@pytest.mark.asyncio
async def test_scraper_node_missing_url(base_state: Dict[str, Any]) -> None:
    # Arrange
    base_state.pop("target_url")

    # Act
    result = await scraper_node(cast(PropertyState, base_state))

    # Assert
    assert result["raw_listing_text"] == "ERROR:SCRAPING_BLOCKED"


@pytest.mark.asyncio
async def test_commuter_node_success(mocker: Any, base_state: Dict[str, Any]) -> None:
    # Arrange
    mock_fetch = mocker.patch("src.agents.nodes.fetch_google_maps", new_callable=AsyncMock)
    mock_fetch.return_value = {
        "status": "OK",
        "rows": [
            {
                "elements": [
                    {"status": "OK", "duration": {"value": 1800}, "distance": {"value": 5000}}
                ]
            }
        ],
    }

    # Provide extracted parameters in state
    state = base_state.copy()
    state["extracted_parameters"] = StructuralParameters(
        price=300000,
        sqm=100,
        property_address="Via Roma 1",
        floor="2",
        bedrooms=3,
        energy_class="A",
    )

    with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "fake_key"}):
        # Act
        result = await commuter_node(cast(PropertyState, state))

        # Assert
        assert result["commute_data"] is not None
        assert result["commute_data"].transit_time_mins == 30
        assert result["commute_data"].distance_km == 5.0


@pytest.mark.asyncio
async def test_osint_node_success(mocker: Any, base_state: Dict[str, Any]) -> None:
    # Arrange
    mock_fetch = mocker.patch("src.agents.nodes.fetch_tavily", new_callable=AsyncMock)
    mock_fetch.return_value = {
        "results": [{"content": "Very safe neighborhood, well connected with FTTH fiber."}]
    }

    from langchain_core.runnables import RunnableLambda

    mock_llm_chain = mocker.patch("src.agents.nodes.get_llm")
    mock_llm_chain.return_value.with_structured_output.return_value = RunnableLambda(
        lambda x: OsintAnalysis(
            broadband_type="FTTH",
            safety_score=0.8,
            noise_level=0.5,
            public_transport_score=0.9,
            amenities_score=0.8,
        )
    )

    state = base_state.copy()
    state["extracted_parameters"] = StructuralParameters(
        price=300000, sqm=100, property_address="Via Roma 1", energy_class="A"
    )

    with patch.dict(os.environ, {"TAVILY_API_KEY": "fake_key"}):
        # Act
        result = await osint_node(cast(PropertyState, state))

        # Assert
        assert result["osint_data"] is not None
        assert result["osint_data"].broadband_type == "FTTH"
        assert result["osint_data"].safety_score == 0.8
