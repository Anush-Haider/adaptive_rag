from langchain_core.documents import Document
from tools.retriever import vector_manager
from graph import app


def seed_local_vector_db():
    """
    Seeds the local vector database with highly specific mock data.
    This helps us test if the agent accurately falls back to web search
    when asked about a topic it doesn't know.
    """
    print("\n--- SEEDING LOCAL ARCHIVE CONTEXTS ---")

    # Let's seed context about a specific mock project
    mock_documents = [
        Document(
            page_content="Project Titan Alpha is an internal next-generation orchestration engine built by Team X in 2026. It utilizes a custom Python runtime optimized for sub-millisecond async task delivery.",
            metadata={"source": "internal_docs_project_titan"}
        ),
        Document(
            page_content="The deployment blueprint of Project Titan Alpha requires a minimum configuration of 3 Redis replicas and uses a modified Raft consensus protocol for handling state replication across clusters.",
            metadata={"source": "deployment_manual_titan"}
        )
    ]

    # Initialize and persist the database using our manager
    vector_manager.get_vectorstore(documents=mock_documents)
    print("--- CHROMA DB SEEDED AND READY ---")


def run_agent(query: str):
    """
    Invokes the compiled self-correcting LangGraph state machine.
    """
    print(f"\n========================================================")
    print(f"RUNNING AGENT FOR QUERY: '{query}'")
    print(f"========================================================\n")

    # Initialize default state parameters matching our AgentState TypedDict contract
    initial_state = {
        "question": query,
        "transformed_query": "",
        "generation": "",
        "documents": [],
        "search_needed": False,
        "hallucination_retry": 0,
        "steps": []
    }

    # Execute the graph
    final_output = app.invoke(initial_state)

    print(f"\n========================================================")
    print(f"FINAL AGENT RESPONSE")
    print(f"========================================================")
    print(final_output["generation"])
    print(f"\nExecution Path Audit Trail: {final_output['steps']}")
    print(f"========================================================\n")


if __name__ == "__main__":
    # 1. Seed our local vector database with specific context
    seed_local_vector_db()

    # Test Scenario A: A query that can be answered 100% from our local database.
    # Expected behavior: Retrieve -> Grade (Pass) -> Generate -> End. (No web search)
    run_agent("What is Project Titan Alpha and what protocol does it use for state replication?")

    # Test Scenario B: A query that is completely missing from our database.
    # Expected behavior