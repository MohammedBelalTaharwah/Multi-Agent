# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from graph import app
import time
import json

load_dotenv()

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="brain", layout="wide")


def check_api_keys():
    llm_provider = os.environ.get("LLM_PROVIDER", "groq").lower().strip()
    search_provider = os.environ.get("SEARCH_PROVIDER", "duckduckgo").lower().strip()
    missing = []
    if llm_provider == "groq" and not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY (get free at console.groq.com)")
    elif llm_provider == "together" and not os.environ.get("TOGETHER_API_KEY"):
        missing.append("TOGETHER_API_KEY")
    if search_provider == "tavily" and not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        st.error(f"Missing API keys: {', '.join(missing)}. Please set them in your .env file.")
        return False
    st.success("Configuration loaded successfully.")
    return True


st.title("Multi-Agent Research Assistant")
st.markdown("""
Enter a research topic, and a team of 9 AI agents will collaborate to produce a comprehensive report.

**Agent Team:**
- **Supervisor** - Manages workflow & generates sub-questions
- **Researcher** - Searches web, scrapes pages, stores in vector DB
- **Browser** - Playwright automation for academic databases
- **Classifier** - Classifies sources by category & relevance (score 1-10)
- **NER Agent** - Extracts entities (people, orgs, tech) & relationships
- **Analyzer** - Identifies themes & builds structured outline
- **Writer** - Creates academic report with citations & image requests
- **Illustrator** - Generates illustrations via Stable Diffusion API
- **Critiquer** - Reviews with scoring rubric (1-10), approves or revises
""")

st.divider()

if not check_api_keys():
    st.stop()

st.header("Start Your Research")

topic = st.text_input("Enter your research topic:", placeholder="e.g., Impact of quantum computing on cybersecurity", key="topic_input")

with st.sidebar:
    st.header("Configuration")
    max_iterations = st.slider("Max Workflow Iterations", min_value=5, max_value=30, value=20)
    enable_browser = st.checkbox("Enable Browser Agent (Playwright)", value=False)
    enable_illustrations = st.checkbox("Enable Image Generation (Stability AI)", value=False)
    st.divider()
    st.subheader("How it works")
    st.markdown("""
    1. **Supervisor** plans research & generates sub-questions
    2. **Researcher** gathers info (web search + scraping)
    3. **Browser** searches academic databases (optional)
    4. **Classifier** categorizes & scores sources
    5. **NER Agent** extracts entities & relationships
    6. **Analyzer** finds themes & builds outline
    7. **Writer** creates the report
    8. **Illustrator** generates diagrams (optional)
    9. **Critiquer** scores & approves/revises
    10. **Finalize** exports PDF
    """)

if st.button("Start Research", type="primary", use_container_width=True):
    if not topic:
        st.error("Please enter a research topic.")
    else:
        initial_state = {
            "main_task": topic,
            "sub_questions": [],
            "status": "started",
            "research_findings": [],
            "raw_sources": [],
            "browser_results": [],
            "specialized_research_done": [],
            "classification_results": "",
            "classified_sources": [],
            "named_entities": "",
            "entities": [],
            "entity_relationships": [],
            "analysis_outline": "",
            "organized_findings": [],
            "outline": [],
            "draft": "",
            "illustration_requests": [],
            "generated_illustrations": [],
            "illustrations": [],
            "critique_notes": "",
            "critic_feedback": None,
            "critic_score": None,
            "revision_number": 0,
            "final_report": None,
            "transformer_config": {"provider": os.environ.get("LLM_PROVIDER", "groq"), "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")},
            "moe_analysis": None,
            "next_step": "",
            "current_sub_task": "",
            "research_persona": "",
            "human_input": None,
        }
        config = {"recursion_limit": max_iterations}

        st.info("Agents are starting their work...")
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        progress_container = st.container()

        final_state = None
        step_count = 0
        all_states = []

        try:
            with progress_container:
                st.subheader("Agent Activity Log")

                for step in app.stream(initial_state, config=config):
                    step_count += 1
                    progress_bar.progress(min(step_count / max_iterations, 1.0))

                    node_name = list(step.keys())[0]
                    node_output = step[node_name]
                    all_states.append((node_name, node_output))
                    final_state = node_output

                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### Agent: `{node_name.upper()}`")
                        with col2:
                            st.caption(f"Step {step_count}")

                        if node_name == "supervisor":
                            ns = node_output.get('next_step', 'N/A')
                            task = node_output.get('current_sub_task', 'N/A')
                            st.markdown(f"**Decision:** {ns}")
                            st.markdown(f"**Task:** {task}")

                        elif node_name == "researcher":
                            findings = node_output.get('research_findings', [])
                            sources = node_output.get('raw_sources', [])
                            if findings:
                                st.success(f"Research completed ({len(sources)} sources)")
                                with st.expander(f"View Research (Step {step_count})"):
                                    st.markdown(findings[-1])
                            if sources:
                                with st.expander(f"Sources ({len(sources)})"):
                                    for s in sources:
                                        st.markdown(f"- [{s.get('title','')}]({s.get('url','')})")

                        elif node_name == "browser":
                            findings = node_output.get('research_findings', [])
                            if findings:
                                st.success("Browser search completed")
                                with st.expander(f"View Browser Results (Step {step_count})"):
                                    st.markdown(findings[-1])

                        elif node_name == "classifier":
                            result = node_output.get('classification_results', '')
                            st.success("Sources classified and scored")
                            with st.expander(f"View Classification (Step {step_count})"):
                                st.markdown(result)
                            cls_srcs = node_output.get('classified_sources', [])
                            if cls_srcs:
                                st.caption(f"{len(cls_srcs)} classified sources")

                        elif node_name == "ner":
                            result = node_output.get('named_entities', '')
                            rels = node_output.get('entity_relationships', [])
                            st.success(f"Entities extracted ({len(rels)} relationships)")
                            with st.expander(f"View NER Results (Step {step_count})"):
                                st.markdown(result)
                            if rels:
                                st.caption(f"Relationships: {len(rels)}")

                        elif node_name == "analyzer":
                            result = node_output.get('analysis_outline', '')
                            st.success("Themes extracted & outline built")
                            with st.expander(f"View Analysis & Outline (Step {step_count})"):
                                st.markdown(result)

                        elif node_name == "illustrator":
                            ill_texts = node_output.get('generated_illustrations', [])
                            ill_data = node_output.get('illustrations', [])
                            st.success(f"{len(ill_texts)} illustrations processed")
                            with st.expander(f"View Illustrations (Step {step_count})"):
                                for idx, ill in enumerate(ill_texts, 1):
                                    st.markdown(f"**Diagram {idx}:**")
                                    st.info(ill)
                                    if idx < len(ill_texts):
                                        st.divider()
                                for ill in ill_data:
                                    if ill.get("image_path"):
                                        st.image(ill["image_path"], caption=ill.get("section", ""))

                        elif node_name == "writer":
                            draft = node_output.get('draft', '')
                            revision = node_output.get('revision_number', 0)
                            st.success(f"Draft {revision} generated ({len(draft)} chars)")
                            preview = 400
                            if len(draft) > preview:
                                st.markdown("**Preview:**")
                                st.info(draft[:preview] + "...")
                                with st.expander(f"View Full Draft (Step {step_count})"):
                                    st.markdown(draft)
                            else:
                                st.markdown("**Draft:**")
                                st.info(draft)

                        elif node_name == "critiquer":
                            feedback = node_output.get('critic_feedback', '')
                            score = node_output.get('critic_score', None)
                            if "APPROVED" in feedback.upper() or (score and score >= 7):
                                st.success(f"Draft APPROVED (score: {score}/10)" if score else "Draft APPROVED!")
                            else:
                                st.warning(f"Revisions requested (score: {score}/10)" if score else "Revisions requested")
                            with st.expander(f"View Critique (Step {step_count})"):
                                st.markdown(feedback)
                                if score:
                                    st.metric("Score", f"{score}/10")

                        elif node_name == "finalize":
                            st.success("Report finalized!")

                        st.divider()
                    time.sleep(0.2)

            status_placeholder.success("Research Complete!")
            progress_bar.progress(1.0)

        except Exception as e:
            status_placeholder.error("Error occurred")
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            final_state = None

        # Display final report
        st.divider()
        final_draft = None
        if final_state and isinstance(final_state, dict):
            final_draft = final_state.get("draft", "")
        if not final_draft or len(final_draft.strip()) < 50:
            for node_name, state in reversed(all_states):
                if isinstance(state, dict) and state.get("draft"):
                    dc = state.get("draft", "")
                    if len(dc.strip()) > 50:
                        final_draft = dc
                        final_state = state
                        break

        if final_draft and len(final_draft.strip()) > 50:
            st.header("Final Research Report")

            # Check for PDF
            import glob as gb
            pdfs = gb.glob("reports/*.pdf")
            if pdfs:
                latest_pdf = max(pdfs, key=os.path.getctime)
                with open(latest_pdf, "rb") as f:
                    st.download_button("Download PDF", data=f, file_name=os.path.basename(latest_pdf), mime="application/pdf")

            # Show report tabs
            tab1, tab2, tab3 = st.tabs(["Report", "Statistics", "Entities & Sources"])
            with tab1:
                st.markdown(final_draft)

            with tab2:
                rev_count = final_state.get("revision_number", 0) if isinstance(final_state, dict) else 0
                src_count = len(final_state.get("raw_sources", [])) if isinstance(final_state, dict) else 0
                ent_count = len(final_state.get("entities", [])) if isinstance(final_state, dict) else 0
                score = final_state.get("critic_score", None) if isinstance(final_state, dict) else None
                word_count = len(final_draft.split())
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Revisions", rev_count)
                    st.metric("Sources", src_count)
                with col2:
                    st.metric("Entities", ent_count)
                    st.metric("Word Count", word_count)
                with col3:
                    if score:
                        st.metric("Critic Score", f"{score}/10")

            with tab3:
                entities_list = final_state.get("entities", []) if isinstance(final_state, dict) else []
                relationships = final_state.get("entity_relationships", []) if isinstance(final_state, dict) else []
                if entities_list:
                    st.subheader("Named Entities")
                    st.json(entities_list)
                if relationships:
                    st.subheader("Entity Relationships")
                    st.json(relationships)

            st.download_button(
                label="Download Report (TXT)",
                data=final_draft,
                file_name=f"research_report_{topic.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.error("No report was generated. Please try again.")
            if final_state:
                with st.expander("Debug: View Final State"):
                    st.json(final_state if isinstance(final_state, dict) else {"error": "Not a dictionary"})

st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
<p>Powered by LangChain, LangGraph, Groq, ChromaDB & Streamlit</p>
</div>
""", unsafe_allow_html=True)
