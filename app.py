import asyncio
from typing import Any, Dict, cast

import streamlit as st

from src.agents.nodes import (
    commuter_node,
    data_extractor_node,
    evaluator_node,
    financial_node,
    negotiator_node,
    osint_node,
    scraper_node,
)
from src.core.graph import app as graph_app
from src.core.state import PropertyState

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Home Finder", page_icon="🏢", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM CSS - PREMIUM PROPTECH THEME
# ─────────────────────────────────────────────────────────────────────────────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. UI - HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="premium-header">
    <h1>Home Finder<span style="color: #0F52BA;">.</span></h1>
    <p>Advanced AI-driven real estate investment evaluation</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. UI - INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📍 Property Details</div>', unsafe_allow_html=True)

col_url, col_addr = st.columns([2, 1], gap="medium")
with col_url:
    target_url = st.text_input(
        "Real Estate Listing URL", placeholder="https://www.immobiliare.it/annunci/..."
    )
with col_addr:
    user_office_address = st.text_input("Office Address", value="Piazza del Duomo, Milan")

with st.expander("🛠️ Manual Text Entry (Use if scraper is blocked)"):
    manual_text = st.text_area("Paste the listing text here", height=150)

st.markdown('<div class="section-title">🏛️ Financial Parameters</div>', unsafe_allow_html=True)

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    max_budget = st.number_input("Max Budget (€)", min_value=10000, value=350000, step=10000)
with f_col2:
    down_payment = st.number_input("Down Payment (€)", value=50000, step=5000)
with f_col3:
    interest_rate = st.number_input("Interest Rate (%)", value=3.5, step=0.1, format="%.1f")
with f_col4:
    loan_term_years = st.number_input("Term (Years)", value=30, min_value=1, max_value=40)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ACTION BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.write("")
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    start_analysis = st.button("🚀 Analyze Property", type="primary", use_container_width=True)
st.write("")


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXECUTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
async def run_langgraph_stream(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    final_state = initial_state.copy()
    async for output in graph_app.astream(cast(Any, initial_state)):
        for node_name, node_state in output.items():
            if node_name == "scraper_node":
                st.write("✅ **Scraper:** Content extracted.")
            elif node_name == "data_extractor_node":
                st.write("🧠 **AI Extractor:** Data structured.")
            elif node_name == "commuter_node":
                st.write("🚗 **Maps:** Commute calculated.")
            elif node_name == "osint_node":
                st.write("🌍 **OSINT:** Neighborhood analyzed.")
            elif node_name == "financial_node":
                st.write("💰 **Financial:** Mortgage simulated.")
            elif node_name == "evaluator_node":
                st.write("✍️ **Evaluator:** Report drafted.")
            elif node_name == "negotiator_node":
                st.write("🤝 **Negotiator:** Email generated.")
            final_state.update(node_state)
    return final_state


if start_analysis:
    if not target_url and not manual_text:
        st.warning("Please provide a URL or manual text.")
    else:
        with st.status("Initializing Analysis Engine...", expanded=True) as status:
            initial_state: Dict[str, Any] = {
                "user_office_address": user_office_address,
                "max_budget": float(max_budget),
                "down_payment": float(down_payment),
                "interest_rate": interest_rate / 100.0,
                "loan_term_years": int(loan_term_years),
            }

            if manual_text:
                st.write("📝 Using manual text input...")
                state = initial_state.copy()
                state["raw_listing_text"] = manual_text

                # Manual sequential execution for manual text
                state.update(asyncio.run(data_extractor_node(cast(PropertyState, state))))
                st.write("🧠 **AI Extractor:** Data structured.")

                if state.get("hard_constraints_met"):
                    # Run subsequent nodes
                    state.update(asyncio.run(commuter_node(cast(PropertyState, state))))
                    st.write("🚗 **Maps:** Commute calculated.")

                    state.update(asyncio.run(osint_node(cast(PropertyState, state))))
                    st.write("🌍 **OSINT:** Neighborhood analyzed.")

                    state.update(asyncio.run(financial_node(cast(PropertyState, state))))
                    st.write("💰 **Financial:** Mortgage simulated.")

                    st.write("⚙️ **Engine:** Parallel calculations completed.")

                    state.update(asyncio.run(evaluator_node(cast(PropertyState, state))))
                    st.write("✍️ **Evaluator:** Report drafted.")

                    state.update(asyncio.run(negotiator_node(cast(PropertyState, state))))
                    st.write("🤝 **Negotiator:** Email generated.")

                final_output = state
            else:
                initial_state["target_url"] = target_url
                final_output = asyncio.run(run_langgraph_stream(initial_state))

            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # ─────────────────────────────────────────────────────────────────────────────
        # 7. RESULTS DISPLAY
        # ─────────────────────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)

        params = final_output.get("extracted_parameters")

        if params is None:
            st.error(
                final_output.get(
                    "evaluation_report", "❌ Analysis failed. The listing might be inaccessible."
                )
            )
        elif not final_output.get("hard_constraints_met"):
            price = getattr(params, "price", 0)
            st.warning(f"⚠️ Property exceeds budget. Price: € {price:,.0f}".replace(",", "."))
        else:
            # Metrics Row
            score = final_output.get("final_score", 0)
            price = getattr(params, "price", 0)
            commute = cast(Any, final_output.get("commute_data"))
            time = f"{getattr(commute, 'transit_time_mins', 'N/A')} min" if commute else "N/A"
            osint = cast(Any, final_output.get("osint_data"))

            # Neighborhood Score calculation for UI (0-100)
            if osint:
                neigh_score = int(
                    ((getattr(osint, 'safety_score', 0) + 1) * 50 * 0.5) + (getattr(osint, 'amenities_score', 0) * 100 * 0.5)
                )
                neigh_label = f"{neigh_score}/100"
            else:
                neigh_label = "N/A"

            st.markdown(
                f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-label">Investment Score</div>
                    <div class="metric-value" style="color: #059669;">{score}/100</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Asking Price</div>
                    <div class="metric-value">€ {price:,.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Commute Time</div>
                    <div class="metric-value">{time}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Neighborhood</div>
                    <div class="metric-value">{neigh_label}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Report
            st.markdown(
                '<div class="section-title">📝 Executive Summary</div>', unsafe_allow_html=True
            )
            report_md = final_output.get("evaluation_report", "")
            st.markdown(f'<div class="report-box">{report_md}</div>', unsafe_allow_html=True)

            # Financials
            st.markdown(
                '<div class="section-title">💰 Financial Projection</div>', unsafe_allow_html=True
            )
            fin = cast(Any, final_output.get("financial_data", {}))
            if fin:
                c1, c2 = st.columns(2)
                with c1:
                    st.info(
                        f"**Original Monthly Installment**\n\n€ {fin.get('original_installment', 0):,.2f}"
                    )
                with c2:
                    st.success(
                        f"**Target Price Monthly Installment (-{fin.get('discount_percentage', 0)}%)**\n\n€ {fin.get('discounted_installment', 0):,.2f}"
                    )

            # Negotiation Email
            st.markdown(
                '<div class="section-title">✉️ Negotiation Strategy</div>', unsafe_allow_html=True
            )
            email = final_output.get("negotiation_email", "")
            with st.container():
                st.code(email, language="markdown")