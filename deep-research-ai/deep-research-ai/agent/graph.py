import streamlit as st
from typing import List, TypedDict

from model import get_model
from langgraph.graph import StateGraph, START, END
from ddgs import DDGS
import trafilatura

# --- PAGE CONFIG ---
st.set_page_config(page_title="Deep Research Agent", page_icon="🕵️‍♂️", layout="wide")

# --- CSS FOR "HACKER" VIBE ---
st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #1E1E1E;
        color: #00FF00;
        border: 1px solid #333;
    }
    .report-font {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.info("Using Gemini Flash")
    st.divider()
    st.write("This agent performs:")
    st.caption("1. 🧠 Deep Planning")
    st.caption("2. 🔍 Multi-Step Search")
    st.caption("3. 🕸️ Full Site Scraping")
    st.caption("4. ✍️ Technical Synthesis")


@st.cache_resource
def get_agent_graph():
    model = get_model("gemini-2.5-flash", temperature=0)

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
            result = model.invoke(system_prompt.format(query=query))
            search_queries = [res.strip() for res in result.content.split('\n')]
            print("These are the search queries: ", search_queries)
            return {'search_list': search_queries}
        except Exception:
            print("Error while generating the search queries. Updating state with default query")
            return {'search_list': query, "status_updates": [f"Planned {len(search_queries)} searches"]}

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
        return {'raw_content': scraped_content, "status_updates":[f"Scraped {len(scraped_content)} pages"]}

    def analyse_node(state: AgentState):
        if not state['raw_content']:
            return {"answer": "Failed to gather any info."}
        full_context = "\n".join(state['raw_content'])

        prompt = f"""
            You are a Senior Technical Writer. 
            Topic: {state['query']}
    
            Research Material:
            {full_context}
    
            Task:
            Write a comprehensive "Deep Research Report" in Markdown.
            1. Executive Summary
            2. Detailed Technical Findings (Use bullet points)
            3. Code Examples (if relevant found in text else skip this point)
            4. Citations (Explicitly mention the Source URL for facts)
            """
        try:
            print("thinking.......")
            response = model.invoke(prompt)
            print("Response generated successfully")
            return {"answer": response.content, "status_updates":[f"Report generated successfully"]}
        except Exception as e:
            return {"answer": f"Analysis failed: {e}","status_updates":["There was an error generating the response"]}

    graph = StateGraph(AgentState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("search_node", search_node)
    graph.add_node("scraper_node", scraper_node)
    graph.add_node("analyse_node", analyse_node)
    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", "search_node")
    graph.add_edge("search_node", "scraper_node")
    graph.add_edge("scraper_node", "analyse_node")
    graph.add_edge("analyse_node", END)
    app = graph.compile()
    return app


# --- MAIN UI ---
st.title("🕵️‍♂️ InsightEngine: Deep Research Agent")
st.markdown("Build your own Perplexity-style researcher locally.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
topic = st.chat_input("What do you want to research?")

if topic:
    st.session_state.messages.append({"role": "user", "content": topic})
    with st.chat_message("user"):
        st.write(topic)

    with st.chat_message("assistant"):
        app = get_agent_graph()

        # THE COOL PART: Live Status Container
        status_container = st.status("🚀 Agent starting...", expanded=True)

        try:
            inputs = {"query": topic, "status_updates": []}
            final_output = None

            # Streaming the graph steps
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Update status based on which node finished
                    if "status_updates" in value:
                        msg = value["status_updates"][0]
                        status_container.write(msg)

                    if "answer" in value:
                        final_output = value["answer"]

            status_container.update(label="✅ Research Complete!", state="complete", expanded=False)

            if final_output:
                st.markdown(final_output)
                st.session_state.messages.append({"role": "assistant", "content": final_output})

                # Download Button
                st.download_button(
                    label="📥 Download Report",
                    data=final_output,
                    file_name="research_report.md",
                    mime="text/markdown"
                )
        except Exception as e:
            st.error(f"Agent Error: {e}")
