import streamlit as st
from typing import List, TypedDict
from model import get_model
from langgraph.graph import StateGraph, START, END
from ddgs import DDGS
import trafilatura

# --- PAGE CONFIG (Force Light Mode via Code as fallback) ---
st.set_page_config(
    page_title="InsightEngine Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- CSS FOR "HACKER" VIBE ---
st.markdown("""
<style>
    /* Hide the default header */
    .stApp > header {visibility: hidden;}

    /* Style the status box */
    .stStatusWidget {
        background-color: #F0F2F6;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
    }

    /* Professional Font for Report */
    .report-text {
        font-family: 'Georgia', serif;
        line-height: 1.6;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)
# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.info("Using Gemini Flash")
    st.divider()
    enable_streaming = st.toggle("⚡ Enable Fast Streaming", value=True)
    st.write("This agent performs:")
    st.caption("1. 🧠 Deep Planning")
    st.caption("2. 🔍 Multi-Step Search")
    st.caption("3. 🕸️ Full Site Scraping")
    st.caption("4. ✍️ Technical Synthesis")


@st.cache_resource
def get_llm():
    model = get_model("gemma-3-12b-it", temperature=0)
    return model


class AgentState(TypedDict):
    query: str  # User query
    search_list: List[str]  # List of possible topics
    urls: List[str]  # URLS picked
    raw_content: List[str]  # This is gonna be your content from these URLs
    answer: str  # The LLM generated answer
    status_updates: List[str]  # For the UI


def planner_node(state: AgentState):
    query = state['query']
    system_prompt = """You are a research lead
    This is the topic: {query}
    
    Based on the topic, generate 3 distinct, specific search queries to gather comprehensive data.
    Return ONLY the queries, one per line.
    """
    search_queries = []
    try:
        model = get_llm()
        result = model.invoke(system_prompt.format(query=query))
        search_queries = [res.strip() for res in result.content.split('\n')]
        print("These are the search queries: ", search_queries)
        return {'search_list': search_queries}
    except Exception:
        print("Error while generating the search queries. Updating state with default query")
        return {'search_list': [query], "status_updates": [f"Planned {len(search_queries)} searches"]}


def search_node(state: AgentState):
    unique_urls = []
    with DDGS() as ddgs:
        for query in state['search_list']:
            try:
                # We fetch 2 results per query to respect time/limits
                results = list(ddgs.text(query, max_results=2))
                for r in results:
                    unique_urls.append(r['href'])
            except Exception as e:
                print("Unable to get the URLs")

    # Deduplicate
    final_urls = list(set(unique_urls))
    print("Final URLS fetched. These are the URLs: ", final_urls)
    return {"urls": final_urls, "status_updates": [f"Found {len(unique_urls)} URLs"]}


def scraper_node(state: AgentState):
    search_list = state['urls']
    scraped_content = []
    for url in search_list:
        print(f"Executing search for : {url}")
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text) > 500:
                scraped_content.append(f"SOURCE: {url}\nCONTENT:\n{text}\n{'=' * 50}")
            else:
                print("Skipped it due to low content")
        else:
            print(f"Unable to scrape the data for url {url}")
    return {'raw_content': scraped_content, "status_updates": [f"Scraped {len(scraped_content)} pages"]}


#
# def analyse_node(state: AgentState):
#     if not state['raw_content']:
#         return {"answer": "Failed to gather any info."}
#     full_context = "\n".join(state['raw_content'])
#
#     prompt = f"""
#         You are a Senior Technical Writer.
#         Topic: {state['query']}
#
#         Research Material:
#         {full_context}
#
#         Task:
#         Write a comprehensive "Deep Research Report" in Markdown.
#         1. Executive Summary
#         2. Detailed Technical Findings (Use bullet points)
#         3. Code Examples (if relevant found in text else skip this point)
#         4. Citations (Explicitly mention the Source URL for facts)
#         """
#     try:
#         print("thinking.......")
#         response = model.invoke(prompt)
#         print("Response generated successfully")
#         return {"answer": response.content, "status_updates": [f"Report generated successfully"]}
#     except Exception as e:
#         return {"answer": f"Analysis failed: {e}", "status_updates": ["There was an error generating the response"]}


graph = StateGraph(AgentState)
graph.add_node("planner_node", planner_node)
graph.add_node("search_node", search_node)
graph.add_node("scraper_node", scraper_node)
graph.add_edge(START, "planner_node")
graph.add_edge("planner_node", "search_node")
graph.add_edge("search_node", "scraper_node")
graph.add_edge("scraper_node", END)
app = graph.compile()

# --- MAIN UI ---
st.header("⚡ What do you want to research today?")
topic = st.chat_input("E.g., 'Compare React vs Vue performance in 2026'")

if topic:

    # Clear previous chat if you want a fresh report feeling
    # st.session_state.messages = []

    with st.chat_message("user"):
        st.write(topic)

    with st.chat_message("assistant"):
        status_container = st.status("🚀 Launching Agent...", expanded=True)

        # Run the Research Phase (Non-streaming part)
        inputs = {"query": topic, "status_updates": []}
        final_state = None

        for output in app.stream(inputs):
            for key, value in output.items():
                if "status_updates" in value:
                    for msg in value["status_updates"]:
                        status_container.write(msg)
                if "status_updates" in value:
                    final_state = value

        status_container.update(label="✅ Research Complete! Writing Report...", state="complete", expanded=False)

        # The "Writer" Phase (Streaming Part)
        if final_state and final_state['raw_content']:
            context = "\n\n".join(final_state['raw_content'])

            prompt = f"""
                    You are a Senior Technical Writer. 
                    Topic: {topic}

                    Research Material:
                    {context}

                    Task:
                    Write a comprehensive "Deep Research Report" in Markdown.
                    1. Executive Summary
                    2. Detailed Technical Findings (Use bullet points)
                    3. Citations (Explicitly mention the Source URL for facts)
                    """

            llm = get_llm()

            if enable_streaming:
                st.write("### 📝 Final Report")
                # THE MAGIC LINE: Streams the response chunk by chunk
                response = st.write_stream(llm.stream(prompt))
            else:
                # Traditional static load
                with st.spinner("Writing..."):
                    response = llm.invoke(prompt).content
                    st.markdown(response)

            # Save button for the result
            st.download_button("📥 Download Report", response, file_name="report.md")

        else:
            st.error("Could not find enough data to write a report.")
