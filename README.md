# 🏡 HomeFinder: Automated Real Estate Due Diligence

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange)
![Gemini](https://img.shields.io/badge/LLM-Gemini_3.1_Flash-green)
![Status](https://img.shields.io/badge/Status-Functional_Prototype-brightgreen)

HomeFinder is an autonomous Multi-Agent System (MAS) engineered to eliminate information asymmetry in the real estate market. By leveraging a **LangGraph-based Map-Reduce architecture**, the system autonomously cross-references property listings with unstructured, real-world data (commuting times, neighborhood safety, internet coverage) to generate objective, quantifiable evaluations.

---

## 🚀 Key Features

The evaluation pipeline is divided into three distinct phases:

### 1. Dynamic Scraping & Extraction
*   **Jina Reader Integration:** Bypasses anti-bot 403 blocks to extract clean text from any property listing URL.
*   **AI Data Extraction:** Uses **Gemini 1.5 Flash** to extract price, sqm, floor, features, and the exact property address.
*   **Hard Constraints Filtering:** Automatically rejects properties exceeding the user's maximum budget to save on processing costs.

### 2. Deep-Dive Parallel Agents
Parallelized agents gather localized data for properties that pass the initial filter:
*   ⏱️ **Commuter Agent:** Calculates exact transit times to the user's office via **Google Maps API**.
*   🌍 **OSINT Agent:** Analyzes neighborhood safety and broadband (FTTH) availability via **Tavily Search API**.

### 3. Weighted Sum Model (WSM) & Executive Summary
*   **Mathematical Scoring:** Calculates a final score (0-100) based on price (40%), size (20%), commute (25%), and OSINT data (15%).
*   **Executive Summary:** Generates a "ruthless and pragmatic" Markdown report using **Gemini 1.5 Flash**, highlighting pros and cons with bullet points for maximum readability.

---

## 🧠 System Architecture

HomeFinder utilizes a sophisticated workflow within the LangGraph framework:
1.  **Scraper Node:** Entry point using Jina Reader.
2.  **Data Extractor Node:** Structured extraction and budget validation.
3.  **Router:** Decides whether to proceed (Go) or terminate (No-Go) based on hard constraints.
4.  **Parallel Execution:** Concurrent calls to Google Maps and Tavily.
5.  **Evaluator Node:** Final mathematical scoring and LLM report generation.

---

## 🖥️ User Interface (Streamlit Dashboard)

The project includes a professional **Streamlit dashboard** featuring:
*   **Real-time Tracking:** Uses `st.status` and LangGraph's `.stream()` to show the agent's progress step-by-step.
*   **Custom UI/UX:** Styled metric cards and report boxes for a modern real estate dashboard feel.
*   **Manual Fallback:** An expander to manually paste listing text if scraping is blocked, ensuring 100% usability.

---

## 🛠️ Technology Stack

*   **Core:** Python 3.10+
*   **Orchestration:** LangGraph, LangChain
*   **LLMs:** Google Gemini API (Flash & Pro models)
*   **APIs:** Google Maps Distance Matrix, Tavily Search, Jina Reader
*   **UI:** Streamlit with Custom CSS
*   **Data Validation:** Pydantic

---

## ⚙️ Getting Started (Local Development)

### Prerequisites
* Python 3.10 or higher
* Valid API keys for Google Gemini, Google Maps, and Tavily.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HomeFinder.git
   cd HomeFinder
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GOOGLE_API_KEY=your_gemini_key_here
   GOOGLE_MAPS_API_KEY=your_maps_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

### 🏃‍♂️ Running the Dashboard
Launch the Streamlit interface:
```bash
streamlit run app.py
```

---

## 🗺️ Roadmap & Future Integrations

- [x] LangGraph Core Workflow
- [x] Jina Reader Scraping Integration
- [x] Streamlit Dashboard with Real-time Tracking
- [ ] **Future:** Computer Vision integration for floor plan analysis.
- [ ] **Future:** Cadastral API integration for legal verification.

---

