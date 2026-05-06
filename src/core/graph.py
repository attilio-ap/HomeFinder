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
    Legge lo stato dopo l'estrazione.
    Se i vincoli hard non sono soddisfatti (False), termina il grafo.
    Se sono soddisfatti (True), restituisce la lista di nodi da lanciare in parallelo.
    """
    if not state.get('hard_constraints_met'):
        return END
    
    return ["commuter_node", "osint_node", "financial_node"]


# Inizializza il StateGraph passando PropertyState
graph = StateGraph(PropertyState)

# Aggiungi i nodi al grafo
graph.add_node("scraper_node", scraper_node)
graph.add_node("data_extractor_node", data_extractor_node)
graph.add_node("commuter_node", commuter_node)
graph.add_node("osint_node", osint_node)
graph.add_node("financial_node", financial_node)
graph.add_node("evaluator_node", evaluator_node)
graph.add_node("negotiator_node", negotiator_node)

# Imposta l'entry point
graph.set_entry_point("scraper_node")

# Aggiungi un arco lineare dallo scraper all'estrattore dati
graph.add_edge("scraper_node", "data_extractor_node")

# Aggiungi un conditional edge in uscita da "data_extractor_node"
graph.add_conditional_edges(
    "data_extractor_node",
    route_after_extraction
)

# Aggiungi i normali edge per il Fan-in al nodo valutatore
# (L'esecuzione parallela confluisce qui)
graph.add_edge("commuter_node", "evaluator_node")
graph.add_edge("osint_node", "evaluator_node")
graph.add_edge("financial_node", "evaluator_node")

# Dopo evaluator_node, il flusso va al negoziatore
graph.add_edge("evaluator_node", "negotiator_node")

# Il negoziatore chiude il grafo
graph.add_edge("negotiator_node", END)

# Compila il grafo in una variabile chiamata app
app = graph.compile()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    # Crea uno stato iniziale di test
    initial_state = {
        "target_url": "https://www.immobiliare.it/annunci/124904795/",
        "user_office_address": "Piazza del Duomo, Milano",
        "max_budget": 350000.0,
        "down_payment": 50000.0,
        "interest_rate": 0.035,
        "loan_term_years": 30
    }
    
    print("Avvio elaborazione LangGraph...\n")
    
    # Lancia l'esecuzione del grafo
    final_output = app.invoke(initial_state)
    
    # Stampa a schermo il dizionario di output finale utilizzando pprint
    print("\n--- Output Finale ---")
    pprint.pprint(final_output)
