from langgraph.graph import StateGraph, END

from state.graph_state import AgentState
from nodes.retrieval_nodes import retrieve_documents, execute_web_search
from nodes.grading_nodes import grade_documents, check_hallucination, evaluate_answer_utility
from nodes.generation_nodes import generate_answer, transform_query

# ------------------------------------------------------------------------------
# CONDITIONAL ROUTING EDGE FUNCTIONS
# ------------------------------------------------------------------------------


def route_after_grading(state: AgentState) -> str:
    """
    Evaluates document relevance. Includes a loop-breaking guardrail to prevent
    infinite web search cycles if local HF model parsers fail consistently.
    """
    # Look back at execution logs to count web search frequencies
    web_search_count = state["steps"].count("execute_web_search")
    if web_search_count >= 2:
        print("--- GUARDRAIL: MAX WEB SEARCH TRIPS REACHED. FORCING GENERATION ---")
        return "generate_answer"

    if state["search_needed"]:
        print("--- ROUTING PROXY: IRRELEVANT DOCS FOUND -> ROUTING TO WEB SEARCH ---")
        return "transform_query"
    else:
        print("--- ROUTING PROXY: ALL DOCS CLEAN -> ROUTING TO GENERATION ---")
        return "generate_answer"


def route_after_hallucination_check(state: AgentState) -> str:
    """
    Evaluates the state *after* check_hallucination executes to determine
    if the answer is grounded.
    """
    last_step = state["steps"][-1]

    if "check_hallucination_grounded_True" in last_step:
        print("--- ROUTING PROXY: NO HALLUCINATIONS DETECTED -> CHECKING UTILITY ---")
        return "useful"
    else:
        print("--- ROUTING PROXY: HALLUCINATION DETECTED -> REWRITING QUERY LOOP ---")
        return "hallucinated"


def route_after_utility_check(state: AgentState) -> str:
    """
    Final gatekeeper verifying if the response actually satisfies user intent.
    """
    last_step = state["steps"][-1]

    if "evaluate_utility_True" in last_step:
        print("--- ROUTING PROXY: ANSWER IS USEFUL -> TERMINATING WORKFLOW ---")
        return "complete"
    else:
        print("--- ROUTING PROXY: ANSWER NOT USEFUL -> FALLING BACK TO WEB ---")
        return "retry_search"


# ------------------------------------------------------------------------------
# GRAPH COMPILATION
# ------------------------------------------------------------------------------

workflow = StateGraph(AgentState)

# 1. Register Nodes
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("transform_query", transform_query)
workflow.add_node("execute_web_search", execute_web_search)

# Standardize grading operations as explicit nodes
def run_hallucination_node(state: AgentState):
    return check_hallucination(state)

def run_utility_node(state: AgentState):
    return evaluate_answer_utility(state)

workflow.add_node("check_hallucination", run_hallucination_node)
workflow.add_node("evaluate_answer_utility", run_utility_node)

# 2. Build Linear Progressions
workflow.set_entry_point("retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_documents")
workflow.add_edge("transform_query", "execute_web_search")
workflow.add_edge("execute_web_search", "generate_answer")

# Once an answer is generated, pass it directly into our auditing node matrix
workflow.add_edge("generate_answer", "check_hallucination")

# 3. Dynamic Conditional Routing Edges

# Route after initial document grading
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "transform_query": "transform_query",
        "generate_answer": "generate_answer"
    }
)

# Route after checking for hallucinations
workflow.add_conditional_edges(
    "check_hallucination",
    route_after_hallucination_check,
    {
        "hallucinated": "transform_query",
        "useful": "evaluate_answer_utility"
    }
)

# Route after evaluating final answer utility
workflow.add_conditional_edges(
    "evaluate_answer_utility",
    route_after_utility_check,
    {
        "complete": END,
        "retry_search": "transform_query"
    }
)

# Compile into execution instance
app = workflow.compile()