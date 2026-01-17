# 🕵️‍♂️ InsightEngine: Local Deep Research Agent

**InsightEngine** is an autonomous AI agent that performs deep research on any topic. Unlike standard chatbots, it plans a research strategy, browses multiple websites to gather real-time data, and synthesizes a professional, cited report—all running locally on your machine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)
![Gemini](https://img.shields.io/badge/AI-Gemini_Flash-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## 🚀 Features

-   **🧠 Autonomous Planning:** Breaks down vague topics into specific, strategic search queries.
-   **🔍 Deep Web Scraping:** Visits real URLs, strips ads/trackers, and reads raw content (10k+ words).
-   **⚡ Real-Time Streaming:** Hybrid UI that shows the "thinking" process and streams the final report token-by-token.
-   **💸 100% Free to Run:** Built on Google's Gemini Flash (Free Tier) and open-source tools.

## 🛠️ Tech Stack

-   **Brain:** Google Gemini 2.5 Flash (via `langchain-google-genai`)
-   **Orchestration:** LangGraph (Stateful agentic workflows)
-   **Search:** DuckDuckGo (via `duckduckgo-search`)
-   **Scraping:** Trafilatura (Clean text extraction)
-   **UI:** Streamlit

---

## ⚙️ Installation

### Prerequisites
-   Python 3.10 or higher
-   A Google Gemini API Key (Get it free at [aistudio.google.com](https://aistudio.google.com/))

### Steps

```bash
# 1. Clone the repository
git clone git@github.com:SushilDluffy/Agentic-AI.git
cd Agentic-AI\deep-research-ai\deep-research-ai\agent

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

```
## 🏃‍♂️ Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2.  **Configure the Agent:**
    -   The app will open in your browser (usually `http://localhost:8501`).
    -   (Optional) Toggle "Fast Streaming" for the writer mode.

3.  **Start Researching:**
    -   Enter a topic like: *"Technical comparison of Rust vs Go for backend services in 2026"*
    -   Watch the agent plan, search, scraping, and write!
