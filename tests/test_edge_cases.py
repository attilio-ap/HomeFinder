import os
from unittest.mock import AsyncMock, patch
from typing import Any, Dict, cast

import httpx
import pytest

from src.agents.nodes import commuter_node, scraper_node
from src.core.state import StructuralParameters, PropertyState


@pytest.mark.asyncio
async def test_scraper_node_api_timeout(mocker: Any, base_state: Dict[str, Any]) -> None:
    # Arrange
    mock_fetch = mocker.patch("src.agents.nodes.fetch_jina_reader", new_callable=AsyncMock)
    mock_fetch.side_effect = httpx.RequestError("Timeout")

    # Act
    result = await scraper_node(cast(PropertyState, base_state))

    # Assert
    assert result["raw_listing_text"] == "ERROR:SCRAPING_BLOCKED"


@pytest.mark.asyncio
async def test_commuter_node_invalid_address(mocker: Any, base_state: Dict[str, Any]) -> None:
    # Arrange
    mock_fetch = mocker.patch("src.agents.nodes.fetch_google_maps", new_callable=AsyncMock)
    mock_fetch.return_value = {"status": "ZERO_RESULTS"}

    state = base_state.copy()
    state["extracted_parameters"] = StructuralParameters(
        price=300000, sqm=100, property_address="Fake Address Non Existent 123", energy_class="A"
    )

    with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "fake_key"}):
        # Act
        result = await commuter_node(cast(PropertyState, state))

        # Assert
        assert result["commute_data"] is None


@pytest.mark.asyncio
async def test_commuter_node_timeout(mocker: Any, base_state: Dict[str, Any]) -> None:
    # Arrange
    mock_fetch = mocker.patch("src.agents.nodes.fetch_google_maps", new_callable=AsyncMock)
    mock_fetch.side_effect = httpx.RequestError("Timeout")

    state = base_state.copy()
    state["extracted_parameters"] = StructuralParameters(
        price=300000, sqm=100, property_address="Via Roma 1", energy_class="A"
    )

    with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "fake_key"}):
        # Act
        result = await commuter_node(cast(PropertyState, state))

        # Assert
        assert result["commute_data"] is None