from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from config.settings import settings
from state.graph_state import AgentState, GradeDocuments, GradeHallucination, GradeAnswer


def _get_llm_engine():
    """Helper factory to isolate model selection based on environment configuration."""
    if settings.LLM_PROVIDER == "groq":
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0
        )
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=0
    )


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates retrieved documents for relevance to the original user question.
    If any documents are deemed irrelevant, flags the state for a web-search fallback.
    """
    print("--- NODE: EVALUATING DOCUMENT RELEVANCE ---")
    question = state["question"]
    documents = state["documents"]

    # Instantiate LLM and bind it strictly to the validation schema
    base_llm = _get_llm_engine()
    structured_grader = base_llm.with_structured_output(GradeDocuments, method="json_schema")

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI gatekeeper scoring document relevance against a user question.\n"
                   "Analyze the context provided. If it contains matching keywords, topics, or "
                   "semantic meaning related to the question, respond with binary_score='yes'.\n"
                   "If it is completely unrelated, respond with binary_score='no'."),
        ("human", "User Question: {question}\n\nRetrieved Context:\n{context}")
    ])

    grader_chain = grade_prompt | structured_grader

    filtered_documents = []
    search_needed = False

    for doc in documents:
        # Check relevance per document chunk
        res: GradeDocuments = grader_chain.invoke({
            "question": question,
            "context": doc.page_content
        })

        if res.binary_score.lower() == "yes":
            print("--- EVALUATION: DOCUMENT RELEVANT ---")
            filtered_documents.append(doc)
        else:
            print("--- EVALUATION: DOCUMENT IRRELEVANT (Filtered Out) ---")
            search_needed = True

    # If all docs were filtered out, ensure search_needed triggers
    if not filtered_documents:
        search_needed = True

    return {
        "documents": filtered_documents,
        "search_needed": search_needed,
        "steps": state["steps"] + ["grade_documents"]
    }


def check_hallucination(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether the generated answer is strictly grounded in and
    supported by the context documents to block hallucinations.
    """
    print("--- NODE: RUNNING HALLUCINATION EVALUATION ---")
    generation = state["generation"]
    documents = state["documents"]

    # Combine active docs into a single body of facts
    context_str = "\n\n".join([doc.page_content for doc in documents])

    base_llm = _get_llm_engine()
    structured_grader = base_llm.with_structured_output(GradeHallucination, method="json_schema")

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an audit assistant checking for hallucinations.\n"
                   "Compare the generated response against the verified reference documents.\n"
                   "If the response introduces facts, claims, or assumptions NOT directly explicitly stated "
                   "in the reference documents, respond with binary_score='no'.\n"
                   "If the response is 100% grounded and supported by the references, respond with binary_score='yes'."),
        ("human", "Reference Documents:\n{context}\n\nGenerated Response:\n{generation}")
    ])

    grader_chain = hallucination_prompt | structured_grader
    res: GradeHallucination = grader_chain.invoke({
        "context": context_str,
        "generation": generation
    })

    # We pass the score forward so the LangGraph workflow route edge can capture it
    is_grounded = res.binary_score.lower() == "yes"
    print(f"--- EVALUATION: RESPONSE GROUNDED = {is_grounded} ---")

    return {
        "steps": state["steps"] + [f"check_hallucination_grounded_{is_grounded}"]
    }


def evaluate_answer_utility(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates if the generated answer actually resolves the user's intent.
    Even if it isn't hallucinated, does it actually answer the prompt?
    """
    print("--- NODE: EVALUATING ANSWER UTILITY ---")
    question = state["question"]
    generation = state["generation"]

    base_llm = _get_llm_engine()
    structured_grader = base_llm.with_structured_output(GradeAnswer, method="json_schema")

    utility_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert grading if an answer actually resolves a user query.\n"
                   "If the answer answers the core question completely and accurately, respond with binary_score='yes'.\n"
                   "If the answer skips the core requirement or says it doesn't know, respond with binary_score='no'."),
        ("human", "User Question: {question}\n\nGenerated Answer:\n{generation}")
    ])

    grader_chain = utility_prompt | structured_grader
    res: GradeAnswer = grader_chain.invoke({
        "question": question,
        "generation": generation
    })

    is_useful = res.binary_score.lower() == "yes"
    print(f"--- EVALUATION: ANSWER USEFUL = {is_useful} ---")

    return {
        "steps": state["steps"] + [f"evaluate_utility_{is_useful}"]
    }