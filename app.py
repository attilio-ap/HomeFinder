import streamlit as st
import os
from src.core.graph import app as graph_app
from src.agents.nodes import data_extractor_node, commuter_node, osint_node, evaluator_node

# Configurazione della pagina
st.set_page_config(page_title="AI Property Finder", page_icon="🏢", layout="wide")

# CSS Personalizzato
st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
    margin-bottom: 20px;
}
.metric-card h3 {
    margin-top: 0;
    color: #555;
    font-size: 1.1em;
}
.metric-card h2 {
    color: #1f77b4;
    margin-bottom: 0;
    font-size: 1.8em;
}
.report-box {
    background-color: #eef2f5;
    border-left: 5px solid #1f77b4;
    border-radius: 5px;
    padding: 25px;
    margin-top: 10px;
    font-size: 1.05em;
    line-height: 1.6;
    color: #333;
}
/* Adattamento per tema scuro di Streamlit */
@media (prefers-color-scheme: dark) {
    .metric-card {
        background-color: #262730;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card h3 {
        color: #ccc;
    }
    .metric-card h2 {
        color: #4fa8f7;
    }
    .report-box {
        background-color: #1e1e1e;
        color: #ddd;
    }
}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.title("🏢 AI Property Finder")
st.markdown("#### L'Intelligenza Artificiale che valuta il tuo prossimo investimento immobiliare")
st.write("") # Spaziatura

# Input Utente nel Main Layout (Niente Sidebar)
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    target_url = st.text_input("URL dell'annuncio Immobiliare", placeholder="Es. https://www.immobiliare.it/annunci/...")
with col2:
    user_office_address = st.text_input("Indirizzo Ufficio", value="Piazza del Duomo, Milano")
with col3:
    max_budget = st.number_input("Budget Massimo (€)", min_value=50000.0, value=350000.0, step=10000.0)

with st.expander("🛠️ Non riesci a incollare l'URL? Usa il testo manuale"):
    manual_text = st.text_area("Incolla qui il testo dell'annuncio se il sito blocca lo scraper")

st.write("") # Spaziatura

# Pulsante Esecuzione al centro
start_analysis = st.button("Avvia Analisi Completa", type="primary", use_container_width=True)

if start_analysis:
    if not target_url and not manual_text:
        st.warning("Inserisci un URL o incolla il testo dell'annuncio per procedere.")
    else:
        with st.spinner("L'Agente sta lavorando..."):
            if manual_text:
                # Salta il nodo scraper ed esegue i nodi successivi manualmente usando il testo fornito
                state = {
                    "raw_listing_text": manual_text,
                    "user_office_address": user_office_address,
                    "max_budget": max_budget
                }
                
                # Node 1: Estrazione
                state.update(data_extractor_node(state))
                
                # Se i vincoli hard sono soddisfatti, prosegue in parallelo (simulato) e poi valuta
                if state.get("hard_constraints_met"):
                    state.update(commuter_node(state))
                    state.update(osint_node(state))
                    state.update(evaluator_node(state))
                    
                final_output = state
            else:
                # Avvia il grafo normalmente
                initial_state = {
                    "target_url": target_url,
                    "user_office_address": user_office_address,
                    "max_budget": max_budget
                }
                final_output = graph_app.invoke(initial_state)

            # Sezione Visualizzazione Risultati
            st.divider()
            st.header("📊 Risultati dell'Analisi")
            
            params = final_output.get("extracted_parameters")
            
            if params is None:
                err_msg = final_output.get("evaluation_report", "❌ Impossibile analizzare l'immobile. Il testo potrebbe essere bloccato (403) o invalido.")
                st.error(err_msg)
            elif not final_output.get("hard_constraints_met"):
                st.warning("⚠️ L'immobile non soddisfa i vincoli hard (es. supera il budget massimo). Analisi interrotta.")
                if params:
                    price = getattr(params, "price", "N/A")
                    price_str = f"€ {price:,.2f}".replace(",", ".") if price != "N/A" and price is not None else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Prezzo Estratto</h3>
                        <h2>{price_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ Analisi completata con successo!")
                
                # Crea una riga di 4 colonne per le metriche chiave
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                # 1. Punteggio
                score = final_output.get("final_score", 0)
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Punteggio</h3>
                        <h2>{score} / 100</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # 2. Prezzo
                price = getattr(params, "price", "N/A")
                price_str = f"€ {price:,.2f}".replace(",", ".") if price != "N/A" and price is not None else "N/A"
                with m_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Prezzo</h3>
                        <h2>{price_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # 3. Distanza (Commute)
                commute = final_output.get("commute_data")
                dist_str = f"{commute.transit_time_mins} min" if commute and getattr(commute, 'transit_time_mins', None) is not None else "N/A"
                with m_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Distanza Ufficio</h3>
                        <h2>{dist_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # 4. Sicurezza/Fibra (OSINT)
                osint = final_output.get("osint_data")
                # Estraiamo un dato riassuntivo o di sicurezza/fibra
                osint_str = osint.broadband_type if osint and getattr(osint, 'broadband_type', None) else "N/A"
                with m_col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Connettività</h3>
                        <h2>{osint_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Executive Summary
                st.subheader("📝 Executive Summary")
                report = final_output.get("evaluation_report", "")
                
                # Utilizzo della report-box
                st.markdown(f"""
                <div class="report-box">
                    {report}
                </div>
                """, unsafe_allow_html=True)
