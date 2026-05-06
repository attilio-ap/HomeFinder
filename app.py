import streamlit as st
import os
from src.core.graph import app as graph_app
from src.agents.nodes import data_extractor_node, commuter_node, osint_node, financial_node, evaluator_node, negotiator_node

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

# Input Finanziari
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    down_payment = st.number_input("Anticipo Disponibile (€)", value=50000.0, step=5000.0)
with f_col2:
    interest_rate = st.number_input("Tasso Interesse Stimato (%)", value=3.5, step=0.1)
with f_col3:
    loan_term_years = st.number_input("Durata Mutuo (Anni)", value=30, step=1)


with st.expander("🛠️ Non riesci a incollare l'URL? Usa il testo manuale"):
    manual_text = st.text_area("Incolla qui il testo dell'annuncio se il sito blocca lo scraper")

st.write("") # Spaziatura

# Pulsante Esecuzione al centro
start_analysis = st.button("Avvia Analisi Completa", type="primary", use_container_width=True)

if start_analysis:
    if not target_url and not manual_text:
        st.warning("Inserisci un URL o incolla il testo dell'annuncio per procedere.")
    else:
        with st.status("Avvio dell'agente LangGraph...", expanded=True) as status:
            if manual_text:
                # Salta il nodo scraper ed esegue i nodi successivi manualmente usando il testo fornito
                state = {
                    "raw_listing_text": manual_text,
                    "user_office_address": user_office_address,
                    "max_budget": max_budget,
                    "down_payment": down_payment,
                    "interest_rate": interest_rate / 100.0, # converti in decimale
                    "loan_term_years": loan_term_years
                }
                
                st.write("✅ **Scraper:** Bypassato (testo manuale fornito).")
                
                # Node 1: Estrazione
                state.update(data_extractor_node(state))
                st.write("🧠 **AI Extractor:** Parametri e indirizzo strutturati.")
                
                # Se i vincoli hard sono soddisfatti, prosegue in parallelo (simulato) e poi valuta
                if state.get("hard_constraints_met"):
                    state.update(commuter_node(state))
                    st.write("🚗 **Google Maps:** Calcolo del pendolarismo completato.")
                    
                    state.update(osint_node(state))
                    st.write("🌍 **Tavily OSINT:** Analisi del quartiere completata.")
                    
                    state.update(financial_node(state))
                    st.write("💰 **Financial:** Simulazione mutuo e target di prezzo calcolati.")

                    st.write("✍️ **AI Evaluator:** Stesura del report finale in corso...")
                    state.update(evaluator_node(state))

                    st.write("🤝 **Negotiator:** Stesura email di negoziazione in corso...")
                    state.update(negotiator_node(state))
                    
                final_output = state
                status.update(label="Analisi completata con successo!", state="complete", expanded=False)
            else:
                # Avvia il grafo normalmente
                initial_state = {
                    "target_url": target_url,
                    "user_office_address": user_office_address,
                    "max_budget": max_budget,
                    "down_payment": down_payment,
                    "interest_rate": interest_rate / 100.0, # converti in decimale
                    "loan_term_years": loan_term_years
                }
                final_state = initial_state.copy()
                
                # Iteriamo sugli step in tempo reale
                for output in graph_app.stream(initial_state):
                    for node_name, node_state in output.items():
                        # Aggiorniamo la UI in base al nodo appena completato
                        if node_name == "scraper_node":
                            st.write("✅ **Scraper:** Pagina web scaricata ed estratta.")
                        elif node_name == "data_extractor_node":
                            st.write("🧠 **AI Extractor:** Parametri e indirizzo strutturati.")
                        elif node_name == "commuter_node":
                            st.write("🚗 **Google Maps:** Calcolo del pendolarismo completato.")
                        elif node_name == "osint_node":
                            st.write("🌍 **Tavily OSINT:** Analisi del quartiere completata.")
                        elif node_name == "financial_node" or node_name == "financial":
                            st.write("💰 **Financial:** Simulazione mutuo e target di prezzo calcolati.")
                        elif node_name == "evaluator_node":
                            st.write("✍️ **AI Evaluator:** Stesura del report finale in corso...")
                        elif node_name == "negotiator_node" or node_name == "negotiator":
                            st.write("🤝 **Negotiator:** Stesura email di negoziazione in corso...")
                        
                        # Aggiorniamo lo stato globale ad ogni step
                        final_state.update(node_state)
                
                status.update(label="Analisi completata con successo!", state="complete", expanded=False)
                
                # Riassegnamo a final_output per non rompere il resto della UI
                final_output = final_state

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

                # Analisi Finanziaria & Mutuo
                st.subheader("📊 Analisi Finanziaria & Mutuo")
                fin_data = final_output.get("financial_data", {})
                if fin_data:
                    col_fin1, col_fin2 = st.columns(2)
                    
                    orig_price = fin_data.get("original_price", 0)
                    orig_inst = fin_data.get("original_installment", 0)
                    disc_price = fin_data.get("discounted_price", 0)
                    disc_inst = fin_data.get("discounted_installment", 0)
                    disc_perc = fin_data.get("discount_percentage", 12)
                    
                    with col_fin1:
                        st.metric(
                            label=f"Prezzo Originale",
                            value=f"€ {orig_price:,.2f}".replace(",", "."),
                            delta=f"Rata mensile: € {orig_inst:,.2f}".replace(",", "."),
                            delta_color="off"
                        )
                    with col_fin2:
                        st.metric(
                            label=f"Prezzo Target (-{disc_perc}%)",
                            value=f"€ {disc_price:,.2f}".replace(",", "."),
                            delta=f"Rata mensile: € {disc_inst:,.2f}".replace(",", "."),
                            delta_color="normal"
                        )
                else:
                    st.info("Dati finanziari non disponibili.")

                # Email di Negoziazione Generata
                st.subheader("✉️ Email di Negoziazione Generata")
                email_content = final_output.get("negotiation_email", "")
                if email_content:
                    st.info(email_content)
                else:
                    st.warning("Email di negoziazione non generata.")
