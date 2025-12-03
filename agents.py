from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from typing import TypedDict, Annotated
import operator
from langchain.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

from web_search_2 import web_search_hybrid

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    temperature = 0.1
)


class ResearchState(TypedDict):
    query: str
    plan: str
    raw_evidence: list[str]
    facts: list[str]
    citations: list[str]
    draft: str
    evaluator_feedback: str
    final_summary: str
    messages: Annotated[list, operator.add]
    next_agent: str


class PlannerOutput(BaseModel):
    """BaseModel for structured response for the planner agent"""
    plan: str = Field(description="Plan for the research assistant based on the user query")


# Planner agent
def planner_agent(state: ResearchState) -> ResearchState:
    structured_llm = llm.with_structured_output(PlannerOutput)

    prompt = f"""
        You are the Planner Agent. Your job is to read the user query and break it into clear research steps.

        Instructions:
        1. Do not answer the query.
        2. Create an ordered plan of specific steps.
        3. Steps must focus on gathering facts, data, citations, comparisons, and context needed for a final research brief.
        4. Output only the plan field required by the schema.

        Goal of the system:
        Produce a polished research summary with key insights, data points, citations, and recommended next steps.

        Generate a concise, actionable plan for the Search Agent.

        User Query: {state["query"]}
    """

    result: PlannerOutput = structured_llm.invoke(prompt)
    
    return {
        **state,
        "plan": result.plan,
        "messages": [AIMessage(content="Planner: Steps created")],
        "next_agent": "search_agent"
    }


class SearchOutput(BaseModel):
    """Response structure for search agent"""
    questions: list[str] = Field(description="List of questions based on the research plan")


def search_agent(state: ResearchState) -> ResearchState:
    structured_llm = llm.with_structured_output(SearchOutput)

    prompt = f"""
        You are the Search Agent. Your job is to read the plan provided by the Planner Agent and convert it into ten atomic search questions.

        Instructions:
        1. Do not answer the plan.
        2. Produce ten questions that fully cover every part of the plan.
        3. Each question must be atomic. Each should target one fact, one process step, one policy detail, or one specific concept.
        4. Questions must be crafted so the web search API returns concrete evidence such as facts, statistics, expert statements, definitions, timelines, technology details, or policy information.
        5. Avoid broad or multi part questions. Avoid interpretation, analysis, or recommendations.
        6. Questions must directly support the Extraction Agent, which will pull facts, claims, and citations from the search results.
        7. Output only the questions field required by the schema.

        Goal:
        Generate ten precise, evidence focused search questions that will be sent to the web search API. These questions must allow the Extraction Agent to collect reliable factual evidence for the research brief.

        User Query: {state["query"]}
        Plan: {state["plan"]}

        Generate ten atomic questions.

    """

    result: SearchOutput = structured_llm.invoke(prompt)

    raw_evidence = web_search_hybrid(result.questions, serper_api_key="c1c2522ba6d8aea94360cd66986bd6695b356918")

    return {
        **state,
        "raw_evidence": raw_evidence,
        "messages": [AIMessage(content="Search Agent: Evidence collected in raw_evidence")],
        "next_agent": "extraction_agent"
    }


class ExtractionOutput(BaseModel):
    """Structure for extraction agent"""
    facts: list[str] = Field(description="List of facts based on the evidence")
    citations: list[str] = Field(description="Citations of the facts based on the evidence")


def extraction_agent(state: ResearchState) -> ResearchState:
    structured_llm = llm.with_structured_output(ExtractionOutput)

    prompt = f"""
        You are the Extraction Agent. Your job is to read the raw evidence and extract all factual information that will support the final research brief.

        Instructions:
        1. Extract factual statements, statistics, definitions, process details, policy information, expert viewpoints, and any other verifiable claims.
        2. Each fact can be a full statement. It does not need to be atomic, but it must reflect what is explicitly present in the evidence.
        3. Do not summarise or rewrite information. Capture facts as they appear or as close to the original wording as possible.
        4. Extract citations. Include any URLs, publication names, or source identifiers present in the evidence.
        5. Ignore irrelevant content, repetition, images, and navigation text.
        6. Output only the fields required by the schema: facts and citations.

        Goal:
        Provide clean, accurate factual claims and citation references that the Writer Agent will use to build the research summary.

        Evidence:
        {state["raw_evidence"]}

        Extract factual claims and citations.
    """

    response: ExtractionOutput = structured_llm.invoke(prompt)

    return {
        **state,
        "facts": response.facts,
        "citations": response.citations,
        "messages": [AIMessage(content="Extractor: facts extracted")],
        "next_agent": "writer_agent"
    }


class WriterOutput(BaseModel):
    """Structure for the writer agent"""
    draft: str


def writer_agent(state: ResearchState) -> ResearchState:
    structured_llm = llm.with_structured_output(WriterOutput)

    feedback_section = ""
    if state.get("evaluator_feedback"):
        feedback_section = f"""
        Previous Draft Issues (CRITICAL - Address these):
        {state["evaluator_feedback"]}

        You MUST revise the draft to address all the feedback points above.
        """

    prompt = f"""
        You are the Writer Agent. Your job is to take extracted facts and citations and turn them into a polished research brief.

        {feedback_section}

        Instructions:
        1. Use clear markdown formatting.
        2. Begin with a strong, informative title.
        3. Structure the brief using multiple bold section headers. Use headers appropriate to the topic. Examples include: **Overview**, **Background**, **Key Insights**, **Technical Details**, **Policy Context**, **Findings**, **Challenges**, **Opportunities**, **Future Outlook**, **Citations**, or any similar suitable headers.
        4. Produce a long research brief. Aim for a length equivalent to two to three pages of text when rendered. This means several paragraphs per section, deep explanations, detailed synthesis of the facts, and thorough coverage of the topic.
        5. Expand each section using the provided facts. Combine evidence from different parts of the fact set to create cohesive explanations while remaining grounded in the extracted information.
        6. Use bullet points, short paragraphs, and occasional tables where appropriate.
        7. Maintain a professional, neutral, research oriented tone.
        8. Include a final section for citations using the extracted citation list.
        9. Ensure the draft directly answers the user's original query.

        Goal:
        Produce a comprehensive, multi section research brief based entirely on the provided evidence.

        User Query:
        {state["query"]}

        Facts:
        {state["facts"]}

        Citations:
        {state["citations"]}

        Write the complete research brief.
    """

    response: WriterOutput = structured_llm.invoke(prompt)

    return {
        **state,
        "draft": response.draft,
        "messages": [AIMessage(content="Writer Agent: Draft created")],
        "next_agent": "evaluator_agent"
    }


class EvalOutput(BaseModel):
    """Structure for the evaluation agent"""
    needs_fix: bool = Field(description="Whether the draft needs fixes")
    feedback: str = Field(description="Detailed feedback on what needs to be fixed (empty if draft is approved)")


def evaluator_agent(state: ResearchState) -> ResearchState:
    structured_llm = llm.with_structured_output(EvalOutput)

    prompt = f"""
You are the Evaluator Agent. Your task is to evaluate the Writer Agent's draft against the user's query.

Instructions:
1. Read the draft and the user's query carefully.
2. Check if the draft:
   - Directly addresses ALL parts of the user's query
   - Stays within the scope of the query
   - Does not contain obvious hallucinations or unrelated claims
   - Is complete and well-structured
   - Uses the extracted facts appropriately

3. Set needs_fix to true if there are issues, false if the draft is good.

4. If needs_fix is true, provide specific, actionable feedback:
   - Point out which parts of the query are not addressed
   - Identify any hallucinations or unsupported claims
   - Suggest what needs to be added or removed
   - Be specific and constructive

5. If needs_fix is false, set feedback to an empty string.

User Query:
{state["query"]}

Draft:
{state["draft"]}

Available Facts (for reference):
{state["facts"][:3]}...  

Evaluate the draft and provide your assessment.
    """

    response: EvalOutput = structured_llm.invoke(prompt)

    if response.needs_fix:
        return {
            **state,
            "evaluator_feedback": response.feedback,
            "messages": [AIMessage(content="Evaluator: issues found, sending back")],
            "next_agent": "writer_agent"
        }
    else:
        return {
            **state,
            "final_summary": state["draft"],
            "messages": [AIMessage(content="Evaluator: summary finalized")],
            "next_agent": "end"
        }
    

def super_agent(state: ResearchState) -> ResearchState:
    return {
        **state,
        "messages": [AIMessage(content=f"Router: sending to {state['next_agent']}")]
    }


def route_agent(state: ResearchState):
    return state.get("next_agent", "planner_agent")


workflow = StateGraph(ResearchState)

workflow.add_node("planner_agent", planner_agent)
workflow.add_node("search_agent", search_agent)
workflow.add_node("extraction_agent", extraction_agent)
workflow.add_node("writer_agent", writer_agent)
workflow.add_node("evaluator_agent", evaluator_agent)
workflow.add_node("super_agent", super_agent)

workflow.set_entry_point("planner_agent")

workflow.add_conditional_edges(
    "super_agent",
    route_agent,
    {
        "search_agent": "search_agent",
        "extraction_agent": "extraction_agent",
        "writer_agent": "writer_agent",
        "evaluator_agent": "evaluator_agent",
        "end": END
    }
)

workflow.add_edge("planner_agent", "super_agent")
workflow.add_edge("search_agent", "super_agent")
workflow.add_edge("extraction_agent", "super_agent")
workflow.add_edge("writer_agent", "super_agent")
workflow.add_edge("evaluator_agent", "super_agent")

app = workflow.compile()


def run_research(query: str):
    initial_state = {
        "query": query,
        "plan": "",
        "raw_evidence": [],
        "facts": [],
        "citations": [],
        "draft": "",
        "evaluator_feedback": "",
        "final_summary": "",
        "messages": [],
        "next_agent": ""
    }

    print("\n" + "=" * 60)
    print(f"Starting research workflow for query: {query}")
    print("=" * 60 + "\n")

    final_state = None  # Initialize to capture the last state
    
    # Stream through each event from the graph
    for event in app.stream(initial_state):
        # Each event is {node_name: node_state}
        for node_name, node_state in event.items():
            print(f"Node: {node_name}")
            
            # Capture the state (it gets updated with each node)
            final_state = node_state
            
            # Print the latest internal message if any
            if "messages" in node_state and node_state["messages"]:
                latest = node_state["messages"][-1]
                print(f"  Message: {latest.content}")

            # Planner output preview
            if node_name == "planner_agent" and node_state.get("plan"):
                preview = node_state["plan"][:120].replace("\n", " ")
                print(f"  Plan preview: {preview}...")

            # Search evidence preview
            if node_name == "search_agent" and node_state.get("raw_evidence"):
                ev_count = len(node_state["raw_evidence"])
                print(f"  Evidence items collected: {ev_count}")

            # Extraction preview
            if node_name == "extraction_agent" and node_state.get("facts"):
                print(f"  Extracted facts: {len(node_state['facts'])}")
                print(f"  Extracted citations: {len(node_state['citations'])}")

            # Writer draft preview
            if node_name == "writer_agent" and node_state.get("draft"):
                preview = node_state["draft"][:120].replace("\n", " ")
                print(f"  Draft preview: {preview}...")

            # Evaluator decision
            if node_name == "evaluator_agent":
                if node_state.get("next_agent") == "writer_agent":
                    feedback_preview = node_state.get("evaluator_feedback", "")[:150]
                    print(f"  Evaluator found issues: {feedback_preview}...")
                elif node_state.get("next_agent") == "end":
                    print("  Evaluator approved the summary.")

            print()

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)

    # Use the final_state from streaming, don't invoke again!
    print("\nFinal Research Summary:\n")
    print(final_state["final_summary"])

    return final_state

run_research("What are the most effective ways to get a job in India in 2025 for experienced professionals with 2-3 years experience, considering hiring trends, required skills, demand sectors, and changes in recruitment practices?")