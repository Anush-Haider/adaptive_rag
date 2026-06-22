from typing import Dict, Any
from tools.retriever import vector_manager
from tools.search_engine import web_search_fallback
from state.graph_state import AgentState


def retrieve_documents(state: AgentState) -> Dict[str, Any]:
    """
    Queries the local Chroma vector database using the user's initial question
    and saves the retrieved chunks to the state.
    """
    print("--- NODE: RETRIEVING DOCUMENTS FROM LOCAL CHROMA VDB ---")
    question = state["question"]

    # Fetch our singleton retriever instance
    retriever = vector_manager.get_retriever()
    retrieved_docs = retriever.invoke(question)

    print(f"--- RETRIEVED {len(retrieved_docs)} DOCUMENT CHUNKS ---")

    return {
        "documents": retrieved_docs,
        "steps": state["steps"] + ["retrieve_documents"]
    }


def execute_web_search(state: AgentState) -> Dict[str, Any]:
    """
    Executes a web fallback search using DuckDuckGo.
    Appends the scraped online web snippets to any existing valid documents.
    """
    print("--- NODE: EXECUTE WEB SEARCH FALLBACK ---")

    # Prioritize using the optimized keyword query if it was transformed,
    # otherwise fall back to the raw user question
    search_query = state.get("transformed_query") or state["question"]
    current_docs = state.get("documents", [])

    # Run our free search engine tool
    web_docs = web_search_fallback(search_query)

    # Merge the new web results with the existing filtered documents
    updated_docs = current_docs + web_docs
    print(f"--- WEB SEARCH COMPLETE. TOTAL RUNNING DOCS: {len(updated_docs)} ---")

    return {
        "documents": updated_docs,
        # Reset search needed flag since we just satisfied the fallback step
        "search_needed": False,
        "steps": state["steps"] + ["execute_web_search"]
    }