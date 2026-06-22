# Stateful Self-Correcting Graph RAG Agent

An enterprise-grade, stateful Retrieval-Augmented Generation (RAG) agent engineered with **LangGraph** and **FastAPI**. The system implements an asynchronous, self-correcting state machine workflow utilizing local open-source Large Language Models (LLMs) via Hugging Face Transformers, supported by an optimized vector database index and automated web-search fallbacks.

---

## 🛠️ Architectural Blueprint

The core architecture operates as an explicit, directed acyclic pipeline controlled by a LangGraph state machine runtime. The application decouples operational nodes from conditional evaluation gates to ensure complete code traceability.

### Advanced System Safeguards
* **Log History Desynchronization Fix**: Unlike naive routing implementations that evaluate hallucinations *before* the evaluation node logs updates, this graph introduces linear synchronization: `generate_answer` $\rightarrow$ `check_hallucination` $\rightarrow$ `Conditional Router`. 
* **State-Based Loop Breaker**: A deterministic guardrail inside `route_after_grading` monitors the `state["steps"]` execution trace. If the workflow attempts more than two web-search iterations due to local model parsing formatting drops, it forces a routing fallback to `generate_answer`, guaranteeing strict compute ceilings.

**Core Dependencies**
         Orchestration: LangGraph, LangChain Core
         Vector Database: ChromaDB (Embedded)
         LLM Engine: Ollama / Local Runtime (Qwen/Qwen2.5-1.5B-Instruct or Llama-3.2-1B-Instruct)
         Web Scraper: duckduckgo-search
         Configuration & Execution: Pydantic v2, Pydantic Settings

🐳 **Containerized Infrastructure & Deployment**
The deployment pipeline is fully containerized, utilizing multi-stage volume caching strategies to isolate application operations, handle local                   databases, and preserve large AI model weights across container restarts.

**Prerequisites**
         Docker Desktop or Docker Engine installed on the host machine.
