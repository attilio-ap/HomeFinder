# 🏡 HomeFinder: Automated Real Estate Due Diligence

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange)
![Gemini](https://img.shields.io/badge/LLM-Gemini_3.1_Pro-green)
![Status](https://img.shields.io/badge/Status-Work_In_Progress-yellow)

HomeFinder is an autonomous Multi-Agent System (MAS) engineered to eliminate information asymmetry in the real estate market[cite: 1]. By leveraging a **LangGraph-based Map-Reduce architecture**[cite: 1], the system autonomously cross-references property listings with unstructured, real-world data (commuting times, neighborhood safety, internet coverage) to generate objective, quantifiable evaluations[cite: 1].

📖 **[Read the Full Technical Specification (Whitepaper) here](./docs/main.pdf)**

---

## 🚀 Key Features

The evaluation pipeline is divided into two phases[cite: 1]:

### 1. Hard Constraints Filtering (Boolean)
If a property fails any of these user-defined thresholds, the system triggers a "No-Go" state to save API costs[cite: 1]:
*   **Budget Cap & Value:** Max absolute price and price/m²[cite: 1].
*   **Spatial Dimensions:** Minimum square footage, bedrooms, bathrooms[cite: 1].
*   **Accessibility:** Floor level, elevator presence (extracted via NLP)[cite: 1].

### 2. Soft Constraints (Deep-Dive Agents)
Parallelized agents gather localized data for properties that pass the initial filter[cite: 1]:
*   ⏱️ **Commuter Agent:** Calculates exact transit times for rush hours via Google Maps API[cite: 1].
*   🌐 **Infrastructure Agent:** Verifies FTTH/FTTC broadband or 5G coverage[cite: 1].
*   📰 **OSINT Agent:** Analyzes 24-month local news sentiment to generate a neighborhood safety score[cite: 1].
*   🛒 **POI Radius:** Calculates walking distance to essential services[cite: 1].

---

## 🧠 System Architecture

HomeFinder utilizes a dual-model approach within the LangGraph framework[cite: 1]:
*   **Gemini 3.1 Pro:** Handles complex reasoning, state orchestration, and the final Weighted Sum Model (WSM) scoring[cite: 1].
*   **Gemini 3.1-Lite-Preview:** Deployed for high-throughput, low-latency data extraction tasks[cite: 1].

*(Insert LangGraph flow diagram here - Coming soon)*

---

## 🛠️ Technology Stack

*   **Core:** Python 3.10+
*   **Orchestration:** LangGraph, LangChain[cite: 1]
*   **LLMs:** Google Gemini API[cite: 1]
*   **Data Validation:** Pydantic[cite: 1]
*   **Resilience:** Tenacity (Exponential Backoff for API rate limits)[cite: 1]

---

## ⚙️ Getting Started (Local Development)

### Prerequisites
* Python 3.10 or higher
* Valid API keys for Google Gemini, Google Maps, and a Search API (e.g., Tavily/Serper).

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/HomeFinder.git](https://github.com/YOUR_USERNAME/HomeFinder.git)
   cd HomeFinder
Set up the virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Environment Variables:
Create a .env file in the root directory and add your API keys:

Snippet di codice
GEMINI_API_KEY=your_gemini_key_here
GOOGLE_MAPS_API_KEY=your_maps_key_here
TAVILY_API_KEY=your_tavily_key_here


### 🏃‍♂️ Running the Pipeline (TBD)
*Instructions on how to execute the main graph will be added as the core nodes are finalized.*

---

## 🗺️ Roadmap & Future Integrations

- [ ] Core LangGraph `PropertyState` definition[cite: 1].
- [ ] Implement `Data Extractor Agent`[cite: 1].
- [ ] Implement parallel Map-Reduce routing[cite: 1].
- [ ] **Future:** Computer Vision integration for floor plan analysis[cite: 1].
- [ ] **Future:** Cadastral API integration for legal verification[cite: 1].

---
*Built as a portfolio project demonstrating Senior-level AI engineering and Multi-Agent orchestration.*
