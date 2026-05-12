import os
from typing import Any, Dict, cast
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from src.agents.nodes import OsintAnalysis
from src.core.graph import app
from src.core.state import StructuralParameters


@pytest.mark.asyncio
async def test_graph_happy_path(mocker: Any, base_state: Dict[str, Any]) -> None:
    # Mock Scraper
    mocker.patch(
        "src.agents.nodes.fetch_jina_reader",
        new_callable=AsyncMock,
        return_value="Great appt in Milan. " * 10,
    )

    # Mock Commuter
    mocker.patch(
        "src.agents.nodes.fetch_google_maps",
        new_callable=AsyncMock,
        return_value={
            "status": "OK",
            "rows": [
                {
                    "elements": [
                        {"status": "OK", "duration": {"value": 1800}, "distance": {"value": 5000}}
                    ]
                }
            ],
        },
    )

    # Mock OSINT
    mocker.patch(
        "src.agents.nodes.fetch_tavily", new_callable=AsyncMock, return_value={"results": []}
    )

    # Create a versatile Mock LLM
    class MockLLM(RunnableLambda):
        def __init__(self) -> None:
            super().__init__(self._call_llm)

        def _call_llm(self, *args: Any, **kwargs: Any) -> Any:
            return mocker.Mock(content="Mocked LLM Response")

        def with_structured_output(self, schema: Any, *args: Any, **kwargs: Any) -> Any:
            if schema.__name__ == "StructuralParameters":
                return RunnableLambda(
                    lambda x: StructuralParameters(
                        price=300000,
                        sqm=100,
                        property_address="Via Roma 1",
                        floor="2",
                        bedrooms=3,
                        has_elevator=True,
                        energy_class="A",
                    )
                )
            elif schema.__name__ == "OsintAnalysis":
                return RunnableLambda(
                    lambda x: OsintAnalysis(
                        broadband_type="FTTH",
                        safety_score=0.8,
                        noise_level=0.5,
                        public_transport_score=0.9,
                        amenities_score=0.8,
                    )
                )
            return RunnableLambda(lambda x: mocker.Mock())

    mocker.patch("src.agents.nodes.get_llm", return_value=MockLLM())

    with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "fake", "TAVILY_API_KEY": "fake"}):
        # Act
        final_state = await app.ainvoke(cast(Any, base_state))

        # Assert routing worked
        assert "raw_listing_text" in final_state
        assert "extracted_parameters" in final_state
        assert final_state["hard_constraints_met"] is True
        assert "commute_data" in final_state
        assert "osint_data" in final_state
        assert "financial_data" in final_state
        assert "final_score" in final_state
        assert "negotiation_email" in final_state


@pytest.mark.asyncio
async def test_graph_extraction_failed(mocker: Any, base_state: Dict[str, Any]) -> None:
    # Mock Scraper
    mocker.patch(
        "src.agents.nodes.fetch_jina_reader",
        new_callable=AsyncMock,
        return_value="ERROR:SCRAPING_BLOCKED",
    )

    final_state = await app.ainvoke(cast(Any, base_state))

    # Assert graph ends early
    assert final_state["raw_listing_text"] == "ERROR:SCRAPING_BLOCKED"
    assert final_state.get("extracted_parameters") is None
    assert final_state.get("hard_constraints_met") is False
    assert "final_score" not in final_state  # Evaluator not reached
