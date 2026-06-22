from typing import List
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from config.settings import settings


def web_search_fallback(query: str) -> List[Document]:
    """
    Executes an open-source web search query using DuckDuckGo
    and transforms the raw text outputs into standardized LangChain Document structures.
    """
    print(f"--- EXECUTING FREE WEB SEARCH FALLBACK FOR: '{query}' ---")

    # Initialize the zero-auth scraper wrapper
    search_tool = DuckDuckGoSearchRun()

    try:
        # Fetch search snippets
        search_result = search_tool.run(query)

        # Turn the raw payload into a standardized Document object matching the state schema
        document = Document(
            page_content=search_result,
            metadata={"source": "duckduckgo_search_fallback", "query": query}
        )

        return [document]

    except Exception as e:
        print(f"--- ERROR DURING WEB SEARCH EXECUTION: {str(e)} ---")
        # Return an empty list gracefully so the agent loops back cleanly or safely fails
        return []