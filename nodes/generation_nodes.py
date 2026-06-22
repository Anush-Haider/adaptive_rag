from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nodes.grading_nodes import _get_llm_engine
from state.graph_state import AgentState


def generate_answer(state: AgentState) -> Dict[str, Any]:
    """
    Generates a response based on the compiled documents in the state.
    """
    print("--- NODE: GENERATING RESPONSE CHUNKS ---")
    question = state["question"]
    documents = state["documents"]

    # Concatenate all verified background knowledge
    context_str = "\n\n".join([doc.page_content for doc in documents])

    llm = _get_llm_engine()

    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical assistant responding strictly using the provided context.\n"
                   "Answer the question clearly, concisely, and ensure every statement is fully "
                   "grounded in the provided references. If you do not know the answer, state that you do not know."),
        ("human", "Context Reference:\n{context}\n\nUser Question: {question}")
    ])

    generation_chain = generation_prompt | llm | StrOutputParser()
    response = generation_chain.invoke({"context": context_str, "question": question})

    return {
        "generation": response,
        "steps": state["steps"] + ["generate_answer"]
    }


def transform_query(state: AgentState) -> Dict[str, Any]:
    """
    Optimizes the query by rewriting it into a highly descriptive
    keyword search string optimized for search engine retrieval.
    """
    print("--- NODE: TRANSFORMING/OPTIMIZING QUERY FOR WEB ---")
    question = state["question"]

    llm = _get_llm_engine()

    transform_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a search query optimizer engineering explicit web-search strings.\n"
                   "Analyze the input question and rewrite it to extract core keywords, technical entities, "
                   "and semantic concepts optimized for a public search engine. Output ONLY the optimized string."),
        ("human", "Initial Query: {question}")
    ])

    transform_chain = transform_prompt | llm | StrOutputParser()
    optimized_query = transform_chain.invoke({"question": question})

    print(f"--- OPTIMIZED QUERY FORWARDED: '{optimized_query.strip()}' ---")

    return {
        "transformed_query": optimized_query.strip(),
        "steps": state["steps"] + ["transform_query"]
    }