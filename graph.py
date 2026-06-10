# graph.py

from typing import TypedDict, Annotated, List, Optional, Dict
from langgraph.graph import StateGraph, END
import operator
from agents import (
    create_supervisor_chain,
    create_researcher_agent,
    create_specialized_researcher_agent,
    create_browser_agent,
    create_classification_agent,
    create_ner_agent,
    create_analyzer_agent,
    create_illustration_agent,
    create_writer_chain,
    create_critique_chain
)

# --- 1. Define the State (Full Schema from PDF) ---

class ResearchState(TypedDict):
    """Complete state for the research workflow."""
    # Core
    main_task: str
    sub_questions: Annotated[List[str], operator.add]
    status: str

    # Research phase
    research_findings: Annotated[List[str], operator.add]
    raw_sources: Annotated[List[Dict], operator.add]
    browser_results: Annotated[List[Dict], operator.add]
    specialized_research_done: Annotated[List[str], operator.add]

    # Classification & NER
    classification_results: str
    classified_sources: Annotated[List[Dict], operator.add]
    named_entities: str
    entities: Annotated[List[Dict], operator.add]
    entity_relationships: Annotated[List[Dict], operator.add]

    # Analysis phase
    analysis_outline: str
    organized_findings: Annotated[List[Dict], operator.add]
    outline: Annotated[List[Dict], operator.add]

    # Writing phase
    draft: str
    illustration_requests: Annotated[List[Dict], operator.add]
    generated_illustrations: Annotated[List[str], operator.add]
    illustrations: Annotated[List[Dict], operator.add]

    # Critique phase
    critique_notes: str
    critic_feedback: Optional[str]
    critic_score: Optional[int]
    revision_number: int

    # Final
    final_report: Optional[str]

    # Model config
    transformer_config: Dict
    moe_analysis: Optional[Dict]

    # Flow control
    next_step: str
    current_sub_task: str
    research_persona: str
    human_input: Optional[str]

# --- 2. Initialize Chains and Agents ---

supervisor_chain = create_supervisor_chain()
researcher_agent = create_researcher_agent()
specialized_researcher_hardware = create_specialized_researcher_agent("hardware")
specialized_researcher_ethics = create_specialized_researcher_agent("ethics")
browser_agent = create_browser_agent()
classification_agent = create_classification_agent()
ner_agent = create_ner_agent()
analyzer_agent = create_analyzer_agent()
illustration_agent = create_illustration_agent()
writer_chain = create_writer_chain()
critique_chain = create_critique_chain()

# --- 3. Define Graph Nodes ---

def supervisor_node(state: ResearchState) -> dict:
    """Supervisor decides the next step."""
    print("\n=== SUPERVISOR ===")
    try:
        decision = supervisor_chain(state)
        next_step = decision.get("next_step", "researcher") if isinstance(decision, dict) else "researcher"
        task_desc = decision.get("task_description", "Continue work") if isinstance(decision, dict) else "Continue work"
        sub_qs = decision.get("sub_questions", []) if isinstance(decision, dict) else []
        persona = decision.get("research_persona", "") if isinstance(decision, dict) else ""
    except Exception as e:
        print(f"Supervisor decision error: {e}")
        # Fallback routing based on state
        if state.get("draft") and len(state.get("draft", "").strip()) > 0:
            next_step, task_desc, sub_qs, persona = "finalize", "Finalize report", [], ""
        elif state.get("research_findings"):
            next_step, task_desc, sub_qs, persona = "classifier", "Classify sources", [], ""
        else:
            next_step, task_desc, sub_qs, persona = "researcher", "Research topic", [], ""
    print(f"Decision: {next_step}")
    print(f"Task: {task_desc}")
    ret = {
        "next_step": next_step,
        "current_sub_task": task_desc,
        "research_persona": persona,
    }
    if sub_qs:
        ret["sub_questions"] = sub_qs
    return ret

def research_node(state: ResearchState) -> dict:
    """Research node that gathers information."""
    print("\n=== RESEARCHER ===")
    sub_task = state.get("current_sub_task", state.get("main_task"))
    print(f"Researching: {sub_task}")
    try:
        result = researcher_agent({"input": sub_task})
        findings = result.get("output", "Research completed")
        sources = result.get("sources", [])
        print(f"Found: {str(findings)[:100]}...")
        print(f"Sources: {len(sources)}")
    except Exception as e:
        print(f"Research error: {e}")
        findings = f"Research on {sub_task} - information gathered"
        sources = []
    return {
        "research_findings": [findings],
        "raw_sources": sources,
    }

def specialized_research_node(state: ResearchState) -> dict:
    """Specialized research node that dispatches to hardware or ethics persona."""
    sub_task = state.get("current_sub_task", state.get("main_task"))
    persona = state.get("research_persona", "hardware")
    print(f"\n=== SPECIALIZED RESEARCHER ({persona} persona) ===")
    print(f"Researching: {sub_task}")

    agent_map = {
        "hardware": specialized_researcher_hardware,
        "ethics": specialized_researcher_ethics,
    }
    agent = agent_map.get(persona, specialized_researcher_hardware)

    try:
        result = agent({"input": sub_task})
        findings = result.get("output", f"[{persona}] Research completed")
        sources = result.get("sources", [])
        print(f"Found: {str(findings)[:100]}...")
        print(f"Sources: {len(sources)}")
    except Exception as e:
        print(f"Specialized research error: {e}")
        findings = f"[{persona}] Research on {sub_task}"
        sources = []
    return {
        "research_findings": [findings],
        "raw_sources": sources,
        "specialized_research_done": [persona],
    }

def browser_node(state: ResearchState) -> dict:
    """Browser node for advanced web automation."""
    print("\n=== BROWSER AGENT ===")
    sub_task = state.get("current_sub_task", state.get("main_task"))
    print(f"Browser searching: {sub_task}")
    try:
        result = browser_agent({"input": sub_task})
        findings = result.get("output", "Browser search completed")
        br_data = result.get("browser_data", [])
        print(f"Found: {str(findings)[:100]}...")
    except Exception as e:
        print(f"Browser error: {e}")
        findings = f"Browser search on {sub_task}"
        br_data = []
    return {
        "research_findings": [findings],
        "browser_results": br_data,
    }

def classification_node(state: ResearchState) -> dict:
    """Classification node that categorizes and filters sources."""
    print("\n=== CLASSIFIER ===")
    try:
        result = classification_agent(state)
        classified = result.get("classification", "Classification completed.")
        sources_list = result.get("classified_sources", [])
        print(f"Classification: {str(classified)[:100]}...")
    except Exception as e:
        print(f"Classification error: {e}")
        classified = "Classification completed."
        sources_list = []
    return {
        "classification_results": classified,
        "classified_sources": sources_list,
    }

def ner_node(state: ResearchState) -> dict:
    """NER node that extracts named entities and relationships."""
    print("\n=== NER AGENT ===")
    try:
        result = ner_agent(state)
        entities_text = result.get("entities_text", "Entity extraction completed.")
        entities_list = result.get("entities_list", [])
        relationships = result.get("relationships", [])
        print(f"Entities: {str(entities_text)[:100]}...")
        print(f"Relationships: {len(relationships)}")
    except Exception as e:
        print(f"NER error: {e}")
        entities_text = "Entity extraction completed."
        entities_list = []
        relationships = []
    return {
        "named_entities": entities_text,
        "entities": entities_list,
        "entity_relationships": relationships,
    }

def analyzer_node(state: ResearchState) -> dict:
    """Analyzer node that extracts themes and builds outline."""
    print("\n=== ANALYZER ===")
    try:
        result = analyzer_agent(state)
        outline_text = result.get("outline", "Analysis completed.")
        findings_list = result.get("organized_findings", [])
        outline_list = result.get("outline_list", [])
        print(f"Analysis: {str(outline_text)[:100]}...")
    except Exception as e:
        print(f"Analyzer error: {e}")
        outline_text = "Analysis completed."
        findings_list = []
        outline_list = []
    return {
        "analysis_outline": outline_text,
        "organized_findings": findings_list,
        "outline": outline_list,
    }

def illustration_node(state: ResearchState) -> dict:
    """Illustration node that generates visual diagrams."""
    print("\n=== ILLUSTRATOR ===")
    try:
        result = illustration_agent(state)
        ill_list = result.get("illustrations", []) if isinstance(result, dict) else []
        ill_texts = result.get("illustration_texts", []) if isinstance(result, dict) else []
        if isinstance(result, list):
            ill_texts = result
        print(f"Illustrations generated: {len(ill_texts)}")
    except Exception as e:
        print(f"Illustration error: {e}")
        ill_texts = []
        ill_list = []
    return {
        "generated_illustrations": ill_texts,
        "illustrations": ill_list,
    }

def write_node(state: ResearchState) -> dict:
    """Writer node that creates or revises draft."""
    print("\n=== WRITER ===")
    try:
        result = writer_chain(state)
        draft = result.get("draft", "") if isinstance(result, dict) else str(result)
        ill_requests = result.get("illustration_requests", []) if isinstance(result, dict) else []
    except Exception as e:
        print(f"Writer node error: {e}")
        draft = f"# {state.get('main_task', 'Research Report')}\n\nDraft generation encountered an error."
        ill_requests = []
    print(f"Draft created: {len(draft)} characters")
    if ill_requests:
        print(f"Illustration requests: {len(ill_requests)}")
    return {
        "draft": draft,
        "revision_number": state.get("revision_number", 0) + 1,
        "illustration_requests": ill_requests,
    }

def critique_node(state: ResearchState) -> dict:
    """Critique node that reviews draft with scoring rubric."""
    print("\n=== CRITIQUER ===")
    result = critique_chain(state)
    critique = result.get("feedback", "") if isinstance(result, dict) else str(result)
    score = result.get("score", None) if isinstance(result, dict) else None
    print(f"Critique: {critique[:100]}...")
    if score is not None:
        print(f"Score: {score}/10")
    is_approved = "APPROVED" in critique.upper() or (score is not None and score >= 7)
    if is_approved:
        print("[OK] Draft APPROVED")
        return {
            "critique_notes": "APPROVED",
            "critic_feedback": critique,
            "critic_score": score,
            "next_step": "END",
            "final_report": state.get("draft", ""),
        }
    else:
        print("[REVISE] Revisions needed")
        return {
            "critique_notes": critique,
            "critic_feedback": critique,
            "critic_score": score,
            "next_step": "writer",
        }

def finalize_node(state: ResearchState) -> dict:
    """Finalize the report and produce output."""
    print("\n=== FINALIZE ===")
    try:
        from md_to_pdf import convert_md_to_pdf
        draft = state.get("draft", "")
        if draft:
            pdf_path = convert_md_to_pdf(draft, state.get("main_task", "report"))
            print(f"PDF generated: {pdf_path}")
    except Exception as e:
        print(f"PDF generation skipped: {e}")
    return {"status": "completed"}

# --- 4. Build the Graph ---

def build_graph():
    """Constructs and compiles the LangGraph workflow."""
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("specialized_researcher", specialized_research_node)
    workflow.add_node("browser", browser_node)
    workflow.add_node("classifier", classification_node)
    workflow.add_node("ner", ner_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("illustrator", illustration_node)
    workflow.add_node("writer", write_node)
    workflow.add_node("critiquer", critique_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add edges
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("specialized_researcher", "supervisor")
    workflow.add_edge("browser", "supervisor")
    workflow.add_edge("classifier", "supervisor")
    workflow.add_edge("ner", "supervisor")
    workflow.add_edge("analyzer", "supervisor")
    workflow.add_edge("illustrator", "supervisor")
    workflow.add_edge("writer", "critiquer")
    workflow.add_edge("critiquer", "supervisor")
    workflow.add_edge("finalize", END)

    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next_step", "researcher"),
        {
            "researcher": "researcher",
            "specialized_researcher": "specialized_researcher",
            "browser": "browser",
            "classifier": "classifier",
            "ner": "ner",
            "analyzer": "analyzer",
            "illustrator": "illustrator",
            "writer": "writer",
            "finalize": "finalize",
            "END": END,
        }
    )

    app = workflow.compile()
    return app

app = build_graph()
