import pprint
from typing import List, Union

from langgraph.graph import StateGraph, END

from src.core.state import PropertyState
from src.agents.nodes import (
    scraper_node,
    data_extractor_node,
    commuter_node,
    osint_node,
    evaluator_node,
    financial_node,
    negotiator_node,
)


def route_after_extraction(state: PropertyState) -> Union[List[str], str]:
    """
    Reads the state after extraction.
    If hard constraints are not met (False), terminates the graph.
    If they are met (True), returns the list of nodes to launch in parallel.
    """
    if not state.get('hard_constraints_met'):
        return END
    
    return ["commuter_node", "osint_node", "financial_node"]


# Initialize the StateGraph passing PropertyState
graph = StateGraph(PropertyState)

# Add nodes to the graph
graph.add_node("scraper_node", scraper_node)
graph.add_node("data_extractor_node", data_extractor_node)
graph.add_node("commuter_node", commuter_node)
graph.add_node("osint_node", osint_node)
graph.add_node("financial_node", financial_node)
graph.add_node("evaluator_node", evaluator_node)
graph.add_node("negotiator_node", negotiator_node)

# Set the entry point
graph.set_entry_point("scraper_node")

# Add a linear edge from the scraper to the data extractor
graph.add_edge("scraper_node", "data_extractor_node")

# Add a conditional edge outgoing from "data_extractor_node"
graph.add_conditional_edges(
    "data_extractor_node",
    route_after_extraction
)

# Add standard edges for Fan-in to the evaluator node
# (Parallel execution converges here)
graph.add_edge("commuter_node", "evaluator_node")
graph.add_edge("osint_node", "evaluator_node")
graph.add_edge("financial_node", "evaluator_node")

# After evaluator_node, the flow goes to the negotiator
graph.add_edge("evaluator_node", "negotiator_node")

# The negotiator closes the graph
graph.add_edge("negotiator_node", END)

# Compile the graph into a variable called app
app = graph.compile()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create an initial test state
    initial_state = {
        "target_url": "https://www.immobiliare.it/annunci/124904795/",
        "user_office_address": "Piazza del Duomo, Milano",
        "max_budget": 350000.0,
        "down_payment": 50000.0,
        "interest_rate": 0.035,
        "loan_term_years": 30
    }
    
    print("Starting LangGraph processing...\n")
    
    # Launch graph execution
    final_output = app.invoke(initial_state)
    
    # Print the final output dictionary to the screen using pprint
    print("\n--- Final Output ---")
    pprint.pprint(final_output)
