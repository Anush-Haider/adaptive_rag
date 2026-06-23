# Enterprise Adaptive RAG (Retrieval-Augmented Generation) Pipeline 🧠

An advanced, production-ready document orchestration and retrieval engine that implements an **Adaptive RAG architecture**. 

Unlike static RAG systems, this pipeline uses an intelligent LLM classifier router to dynamically analyze user queries, determine the optimal retrieval strategy (Vector Store Search, Live Web Scraping, or Direct LLM Generation), and self-correct retrieval gaps using automated query-rewriting loops.

## 🏗️ Systems Architecture & Decision Flow

```mermaid
graph TD
    A[User Query] --> B[FastAPI Endpoint / Entrypoint]
    B --> C[Adaptive LLM Router / LiteLLM]
    
    C -->|Complex / Real-Time Query| D[scrapper.py / Async Search Ingestion]
    C -->|Domain-Specific Knowledge| E[pipeline.py / Semantic Retriever]
    C -->|Conversational / General Context| F[Direct Response Generation]
    
    E -->|Context Ingestion| G[(ChromaDB / Vector Store)]
    D -->|Web Scraping Data| G
    
    G --> H[Evaluator Loop / Agentic Validation]
    H -->|Is Context Sufficient? Yes| I[Final Multi-Model Generation]
    H -->|Is Context Insufficient? No| J[Query-Rewriting Engine]
    J -->|Refetched Query| C
