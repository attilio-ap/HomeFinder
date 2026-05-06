import streamlit as st
import os
from src.core.graph import app as graph_app
from src.agents.nodes import data_extractor_node, commuter_node, osint_node, financial_node, evaluator_node, negotiator_node

# Page configuration
st.set_page_config(page_title="Home Finder", page_icon="🏢", layout="wide")

# Custom CSS
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
/* Adaptation for Streamlit dark theme */
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
st.title("🏢 Home Finder")
st.markdown("#### The Artificial Intelligence that evaluates your next real estate investment")
st.write("") # Spacing

# User Input in Main Layout (No Sidebar)
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    target_url = st.text_input("Real Estate Listing URL", placeholder="E.g. https://www.immobiliare.it/annunci/...")
with col2:
    user_office_address = st.text_input("Office Address", value="Piazza del Duomo, Milano")
with col3:
    max_budget = st.number_input("Maximum Budget (€)", min_value=50000.0, value=350000.0, step=10000.0)

# Financial Inputs
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    down_payment = st.number_input("Available Down Payment (€)", value=50000.0, step=5000.0)
with f_col2:
    interest_rate = st.number_input("Estimated Interest Rate (%)", value=3.5, step=0.1)
with f_col3:
    loan_term_years = st.number_input("Mortgage Term (Years)", value=30, step=1)


with st.expander("🛠️ Can't paste the URL? Use manual text"):
    manual_text = st.text_area("Paste the listing text here if the site blocks the scraper")

st.write("") # Spacing

# Execution Button in the center
start_analysis = st.button("Start Full Analysis", type="primary", use_container_width=True)

if start_analysis:
    if not target_url and not manual_text:
        st.warning("Please enter a URL or paste the listing text to proceed.")
    else:
        with st.status("Starting LangGraph agent...", expanded=True) as status:
            if manual_text:
                # Skips the scraper node and executes subsequent nodes manually using the provided text
                state = {
                    "raw_listing_text": manual_text,
                    "user_office_address": user_office_address,
                    "max_budget": max_budget,
                    "down_payment": down_payment,
                    "interest_rate": interest_rate / 100.0, # convert to decimal
                    "loan_term_years": loan_term_years
                }
                
                st.write("✅ **Scraper:** Bypassed (manual text provided).")
                
                # Node 1: Extraction
                state.update(data_extractor_node(state))
                st.write("🧠 **AI Extractor:** Parameters and address structured.")
                
                # If hard constraints are met, proceeds in parallel (simulated) and then evaluates
                if state.get("hard_constraints_met"):
                    state.update(commuter_node(state))
                    st.write("🚗 **Google Maps:** Commuting calculation completed.")
                    
                    state.update(osint_node(state))
                    st.write("🌍 **Tavily OSINT:** Neighborhood analysis completed.")
                    
                    state.update(financial_node(state))
                    st.write("💰 **Financial:** Mortgage simulation and target price calculated.")

                    st.write("✍️ **AI Evaluator:** Drafting final report...")
                    state.update(evaluator_node(state))

                    st.write("🤝 **Negotiator:** Drafting negotiation email...")
                    state.update(negotiator_node(state))
                    
                final_output = state
                status.update(label="Analysis successfully completed!", state="complete", expanded=False)
            else:
                # Starts the graph normally
                initial_state = {
                    "target_url": target_url,
                    "user_office_address": user_office_address,
                    "max_budget": max_budget,
                    "down_payment": down_payment,
                    "interest_rate": interest_rate / 100.0, # convert to decimal
                    "loan_term_years": loan_term_years
                }
                final_state = initial_state.copy()
                
                # Iterate over steps in real-time
                for output in graph_app.stream(initial_state):
                    for node_name, node_state in output.items():
                        # Update UI based on the newly completed node
                        if node_name == "scraper_node":
                            st.write("✅ **Scraper:** Web page downloaded and extracted.")
                        elif node_name == "data_extractor_node":
                            st.write("🧠 **AI Extractor:** Parameters and address structured.")
                        elif node_name == "commuter_node":
                            st.write("🚗 **Google Maps:** Commuting calculation completed.")
                        elif node_name == "osint_node":
                            st.write("🌍 **Tavily OSINT:** Neighborhood analysis completed.")
                        elif node_name == "financial_node" or node_name == "financial":
                            st.write("💰 **Financial:** Mortgage simulation and target price calculated.")
                        elif node_name == "evaluator_node":
                            st.write("✍️ **AI Evaluator:** Drafting final report...")
                        elif node_name == "negotiator_node" or node_name == "negotiator":
                            st.write("🤝 **Negotiator:** Drafting negotiation email...")
                        
                        # Update global state at each step
                        final_state.update(node_state)
                
                status.update(label="Analysis successfully completed!", state="complete", expanded=False)
                
                # Reassign to final_output to not break the rest of the UI
                final_output = final_state

            # Results Display Section
            st.divider()
            st.header("📊 Analysis Results")
            
            params = final_output.get("extracted_parameters")
            
            if params is None:
                err_msg = final_output.get("evaluation_report", "❌ Unable to analyze the property. The text might be blocked (403) or invalid.")
                st.error(err_msg)
            elif not final_output.get("hard_constraints_met"):
                st.warning("⚠️ The property does not meet the hard constraints (e.g., exceeds maximum budget). Analysis aborted.")
                if params:
                    price = getattr(params, "price", "N/A")
                    price_str = f"€ {price:,.2f}".replace(",", ".") if price != "N/A" and price is not None else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Extracted Price</h3>
                        <h2>{price_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ Analysis successfully completed!")
                
                # Create a row of 4 columns for key metrics
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                # 1. Score
                score = final_output.get("final_score", 0)
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Score</h3>
                        <h2>{score} / 100</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # 2. Price
                price = getattr(params, "price", "N/A")
                price_str = f"€ {price:,.2f}".replace(",", ".") if price != "N/A" and price is not None else "N/A"
                with m_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Price</h3>
                        <h2>{price_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # 3. Distance (Commute)
                commute = final_output.get("commute_data")
                dist_str = f"{commute.transit_time_mins} min" if commute and getattr(commute, 'transit_time_mins', None) is not None else "N/A"
                with m_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Office Distance</h3>
                        <h2>{dist_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # 4. Security/Fiber (OSINT)
                osint = final_output.get("osint_data")
                # Extract a summary datum for security/fiber
                osint_str = osint.broadband_type if osint and getattr(osint, 'broadband_type', None) else "N/A"
                with m_col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Connectivity</h3>
                        <h2>{osint_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Executive Summary
                st.subheader("📝 Executive Summary")
                report = final_output.get("evaluation_report", "")
                
                # Use report-box
                st.markdown(f"""
                <div class="report-box">
                    {report}
                </div>
                """, unsafe_allow_html=True)

                # Financial & Mortgage Analysis
                st.subheader("📊 Financial & Mortgage Analysis")
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
                            label=f"Original Price",
                            value=f"€ {orig_price:,.2f}".replace(",", "."),
                            delta=f"Monthly installment: € {orig_inst:,.2f}".replace(",", "."),
                            delta_color="off"
                        )
                    with col_fin2:
                        st.metric(
                            label=f"Target Price (-{disc_perc}%)",
                            value=f"€ {disc_price:,.2f}".replace(",", "."),
                            delta=f"Monthly installment: € {disc_inst:,.2f}".replace(",", "."),
                            delta_color="normal"
                        )
                else:
                    st.info("Financial data not available.")

                # Generated Negotiation Email
                st.subheader("✉️ Generated Negotiation Email")
                email_content = final_output.get("negotiation_email", "")
                if email_content:
                    st.info(email_content)
                else:
                    st.warning("Negotiation email not generated.")
