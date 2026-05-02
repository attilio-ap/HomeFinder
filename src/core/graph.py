import pprint
from typing import List, Union

from langgraph.graph import StateGraph, END

from src.core.state import PropertyState
from src.agents.nodes import (
    data_extractor_node,
    commuter_node,
    osint_node,
    evaluator_node,
)


def route_after_extraction(state: PropertyState) -> Union[List[str], str]:
    """
    Legge lo stato dopo l'estrazione.
    Se i vincoli hard non sono soddisfatti (False), termina il grafo.
    Se sono soddisfatti (True), restituisce la lista di nodi da lanciare in parallelo.
    """
    if not state.get('hard_constraints_met'):
        return END
    
    return ["commuter_node", "osint_node"]


# Inizializza il StateGraph passando PropertyState
graph = StateGraph(PropertyState)

# Aggiungi i 4 nodi al grafo
graph.add_node("data_extractor_node", data_extractor_node)
graph.add_node("commuter_node", commuter_node)
graph.add_node("osint_node", osint_node)
graph.add_node("evaluator_node", evaluator_node)

# Imposta l'entry point
graph.set_entry_point("data_extractor_node")

# Aggiungi un conditional edge in uscita da "data_extractor_node"
graph.add_conditional_edges(
    "data_extractor_node",
    route_after_extraction
)

# Aggiungi i normali edge per il Fan-in al nodo valutatore
graph.add_edge("commuter_node", "evaluator_node")
graph.add_edge("osint_node", "evaluator_node")

# Collega "evaluator_node" a END
graph.add_edge("evaluator_node", END)

# Compila il grafo in una variabile chiamata app
app = graph.compile()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    # Crea uno stato iniziale di test
    initial_state = {
        "property_url": "https://www.immobiliare.it/annunci/12345678/",
        "raw_listing_text": "VENDESI SPLENDIDO TRILOCALE IN ZONA ISOLA! Nel cuore pulsante di Milano, proponiamo in vendita luminoso appartamento di 92 metri quadri sito al quarto piano di uno stabile signorile anni 60. Purtroppo lo stabile è sprovvisto di ascensore. L'immobile si compone di ingresso, due ampie camere da letto, un bagno finestrato e soggiorno con cucina a vista. Richiesta: Euro 285.000 trattabili. Ottimo investimento!"
    }
    
    print("Avvio elaborazione LangGraph...\n")
    
    # Lancia l'esecuzione del grafo
    final_output = app.invoke(initial_state)
    
    # Stampa a schermo il dizionario di output finale utilizzando pprint
    print("\n--- Output Finale ---")
    pprint.pprint(final_output)
