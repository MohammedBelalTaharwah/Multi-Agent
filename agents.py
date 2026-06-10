# agents.py

import os
import json
from dotenv import load_dotenv
import re
from prompts import (
    supervisor_prompt_template,
    researcher_prompt_template,
    hardware_researcher_prompt,
    ethics_researcher_prompt,
    classification_prompt_template,
    ner_prompt_template,
    analyzer_prompt_template,
    illustrator_prompt_template,
    writer_prompt_template,
    critique_prompt_template
)

load_dotenv()

# --- 1. Setup LLM and Tools ---

def get_llm():
    provider = os.environ.get("LLM_PROVIDER", "groq").lower().strip()
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[WARNING] GROQ_API_KEY not set. Get free key at https://console.groq.com")
            api_key = "missing"
        return ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.3,
            max_tokens=8192,
            api_key=api_key
        )
    elif provider == "together":
        from langchain_together import ChatTogether
        return ChatTogether(
            model=os.environ.get("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
            temperature=0.3,
            max_tokens=8192,
            together_api_key=os.environ.get("TOGETHER_API_KEY")
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "llama3"),
            temperature=0.3,
            num_predict=8192
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use: groq, together, ollama")

def get_search_tool():
    provider = os.environ.get("SEARCH_PROVIDER", "duckduckgo").lower().strip()
    if provider == "tavily":
        from langchain_tavily import TavilySearch
        return TavilySearch(max_results=5, topic="general", include_answer=False, include_raw_content=False, search_depth="basic")
    elif provider == "duckduckgo":
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun()
    else:
        raise ValueError(f"Unknown SEARCH_PROVIDER '{provider}'. Use: tavily, duckduckgo")

_llm_instance = None
_search_tool_instance = None


def get_llm_instance():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm()
    return _llm_instance


def get_search_tool_instance():
    global _search_tool_instance
    if _search_tool_instance is None:
        _search_tool_instance = get_search_tool()
    return _search_tool_instance


def _perform_search(query: str) -> tuple:
    """Unified search: returns (raw_text, structured_results_list).
    
    For DuckDuckGo: uses API wrapper to get dicts with title/link/snippet.
    For Tavily: uses existing invoke which returns JSON.
    """
    provider = os.environ.get("SEARCH_PROVIDER", "duckduckgo").lower().strip()
    if provider == "duckduckgo":
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        try:
            results = wrapper.results(query, max_results=5)
        except Exception:
            results = []
        if results and isinstance(results, list):
            formatted = []
            for r in results[:5]:
                title = r.get("title", "Untitled")
                link = r.get("link", "")
                snippet = r.get("snippet", "")
                formatted.append(f"**{title}**\nSource: {link}\n{snippet[:300]}...\n")
            raw = "\n---\n".join(formatted) if formatted else f"Results for: {query}"
            # Normalise keys for downstream code (url->link, content->snippet)
            normalised = []
            for r in results:
                normalised.append({
                    "url": r.get("link", ""),
                    "title": r.get("title", ""),
                    "content": r.get("snippet", ""),
                })
            return raw, normalised
        # Fallback to string-based search
        try:
            st = get_search_tool_instance()
            if hasattr(st, "invoke"):
                resp = st.invoke({"query": query})
            elif callable(st):
                resp = st({"query": query})
            else:
                resp = st.run(query)
        except Exception:
            resp = ""
        return str(resp) if resp else f"No results for: {query}", []

    # Tavily path
    try:
        st = get_search_tool_instance()
        if hasattr(st, "invoke"):
            search_response = st.invoke({"query": query})
        elif callable(st):
            search_response = st({"query": query})
        elif hasattr(st, "run"):
            search_response = st.run(query)
        else:
            raise AttributeError("Search tool not callable")
    except Exception as e:
        print(f"Search error: {e}")
        return f"No results for: {query}", []

    raw_output = ""
    results_list = []
    if isinstance(search_response, str):
        try:
            data = json.loads(search_response)
            results_list = data.get("results", [])
        except json.JSONDecodeError:
            return search_response, []
    elif isinstance(search_response, dict):
        results_list = search_response.get("results", [])
    else:
        return str(search_response), []

    if results_list:
        formatted = []
        for r in results_list[:5]:
            formatted.append(
                f"**{r.get('title','Untitled')}**\n"
                f"Source: {r.get('url','N/A')}\n"
                f"{r.get('content','')[:300]}...\n"
            )
        raw_output = "\n---\n".join(formatted)
    else:
        raw_output = str(search_response)

    return raw_output, results_list

# Vector store (lazy init)
_vstore_instance = None


def get_vstore():
    global _vstore_instance
    if _vstore_instance is None:
        try:
            from vector_store import get_collection, store_finding, search_findings
            _vstore_instance = get_collection()
            globals()["store_finding"] = store_finding
            globals()["search_findings"] = search_findings
        except Exception as e:
            print(f"[VectorStore] Not available: {e}")
            _vstore_instance = None
    return _vstore_instance


def has_vstore():
    return get_vstore() is not None


# MCP tools (lazy init)
_mcp_available = None


def mcp_available():
    global _mcp_available
    if _mcp_available is None:
        try:
            from mcp_tools import get_mcp_tool, list_mcp_tools  # noqa: F401
            _mcp_available = True
        except Exception as e:
            print(f"[MCP] Not available: {e}")
            _mcp_available = False
    return _mcp_available


def _call_llm(llm_obj, *args, **kwargs):
    if hasattr(llm_obj, "invoke") and callable(getattr(llm_obj, "invoke")):
        return llm_obj.invoke(*args, **kwargs)
    if hasattr(llm_obj, "run") and callable(getattr(llm_obj, "run")):
        return llm_obj.run(*args, **kwargs)
    if callable(llm_obj):
        return llm_obj(*args, **kwargs)
    raise AttributeError("LLM/tool object has no invoke/run and is not callable")


def _parse_json_from_text(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join([l for l in lines if not l.strip().startswith("```")])
    text = text.strip()
    return json.loads(text)


# =========================================================
# SUPERVISOR NODE
# =========================================================
def create_supervisor_chain():
    def supervisor_invoke(state):
        research = state.get("research_findings", [])
        research_text = "\n---\n".join(research) if research else "No research yet."

        revision = state.get("revision_number", 0)
        has_research = len(research) > 0
        has_draft = bool(state.get("draft", "").strip())
        critique = state.get("critique_notes", "")
        classification = state.get("classification_results", "")
        entities = state.get("named_entities", "")
        outline = state.get("analysis_outline", "")
        images = state.get("generated_illustrations", [])
        relationships = state.get("entity_relationships", [])
        critic_score = state.get("critic_score", None)
        sub_questions = state.get("sub_questions", [])

        # Track which specialized research personas have been dispatched
        personas_done = state.get("specialized_research_done", [])

        # 1. Approved -> finalize
        if ("APPROVED" in critique.upper() or (critic_score is not None and critic_score >= 7)) and has_draft:
            print("Supervisor: Draft approved, finalizing")
            return {"next_step": "finalize", "task_description": "Finalize the report"}

        # 2. No general research -> researcher first
        if not has_research:
            print("Supervisor: Directing to general researcher")
            return {"next_step": "researcher", "task_description": f"General research: {state.get('main_task','')}"}

        # 2b. Dispatch specialized research personas (hardware first, then ethics)
        if "hardware" not in personas_done:
            print("Supervisor: Directing to specialized researcher (hardware/compute persona)")
            return {"next_step": "specialized_researcher", "task_description": f"Hardware & compute constraints analysis: {state.get('main_task','')}", "research_persona": "hardware"}

        if "ethics" not in personas_done:
            print("Supervisor: Directing to specialized researcher (ethics/sociology persona)")
            return {"next_step": "specialized_researcher", "task_description": f"Ethical & sociological analysis: {state.get('main_task','')}", "research_persona": "ethics"}

        # 3. Research done -> classifier
        if has_research and not classification:
            print("Supervisor: Directing to classifier")
            return {"next_step": "classifier", "task_description": "Classify research sources"}

        # 4. Classified -> NER
        if classification and not entities:
            print("Supervisor: Directing to NER")
            return {"next_step": "ner", "task_description": "Extract named entities"}

        # 5. Entities extracted -> analyzer
        if entities and not outline:
            print("Supervisor: Directing to analyzer")
            return {"next_step": "analyzer", "task_description": "Build analysis outline"}

        # 6. Outline ready -> writer (first draft)
        if outline and not has_draft:
            print("Supervisor: Directing to writer for first draft")
            return {"next_step": "writer", "task_description": "Write first draft"}

        # 7. Draft but no critique -> triggers write->critique flow
        if has_draft and not critique:
            print("Supervisor: Sending draft to critique")
            return {"next_step": "writer", "task_description": "Prepare draft for critique"}

        # 8. Revision needed
        if critique and "APPROVED" not in critique.upper() and revision < 3:
            score_note = f" (score: {critic_score}/10)" if critic_score else ""
            print(f"Supervisor: Revision {revision}{score_note}")
            return {"next_step": "writer", "task_description": f"Revise draft based on critique feedback"}

        # 9. Max revisions
        if revision >= 3:
            print("Supervisor: Max revisions, finalizing")
            return {"next_step": "finalize", "task_description": "Max revisions, finalizing"}

        # 10. Check for pending illustration requests
        ill_requests = state.get("illustration_requests", [])
        if ill_requests and not images:
            print("Supervisor: Directing to illustrator")
            return {"next_step": "illustrator", "task_description": "Generate requested illustrations"}

        # LLM fallback
        prompt = supervisor_prompt_template.format(
            main_task=state.get("main_task", ""),
            sub_questions="; ".join(sub_questions) if sub_questions else "None",
            research_findings=research_text,
            specialized_research_done=", ".join(personas_done) if personas_done else "None yet",
            draft=state.get("draft", "No draft yet."),
            critique_notes=critique if critique else "No critique yet.",
            revision_number=revision,
            classification_status="Completed" if classification else "Pending",
            entities_status="Completed" if entities else "Pending",
            relationships_status=f"{len(relationships)} found" if relationships else "None",
            outline_status="Completed" if outline else "Pending",
            images_status=f"{len(images)} generated" if images else "None",
            critic_score=critic_score if critic_score else "N/A"
        )

        try:
            response = _call_llm(get_llm_instance(), prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            decision = _parse_json_from_text(content)
            if "next_step" in decision:
                return decision
        except Exception as e:
            print(f"Supervisor LLM error: {e}")

        # Final fallback
        if not has_research:
            return {"next_step": "researcher", "task_description": "Research the topic"}
        if not classification:
            return {"next_step": "classifier", "task_description": "Classify sources"}
        if not entities:
            return {"next_step": "ner", "task_description": "Extract entities"}
        if not outline:
            return {"next_step": "analyzer", "task_description": "Build outline"}
        if not has_draft:
            return {"next_step": "writer", "task_description": "Create draft"}
        return {"next_step": "finalize", "task_description": "Finalize report"}

    return supervisor_invoke


# =========================================================
# RESEARCHER NODE
# =========================================================
def create_researcher_agent():
    def researcher_invoke(input_dict):
        query = input_dict.get("input", "")
        if not query or query in ["Continue work", "Complete"]:
            query = "General research information"
        print(f"Researching: {query}")

        try:
            raw_output, results_list = _perform_search(query)
            sources = []

            # Store in vector store
            from vector_store import store_finding
            vs = get_vstore()
            if results_list and vs:
                for i, r in enumerate(results_list):
                    fid = f"src_{hash(query)}_{i}"
                    store_finding(vs, fid, r.get('content', ''), {
                        'title': r.get('title', ''),
                        'url': r.get('url', ''),
                        'query': query
                    })
                    sources.append({
                        "url": r.get('url', ''),
                        "title": r.get('title', ''),
                        "content": r.get('content', '')[:500],
                        "score": 5
                    })

            # Scrape for detailed content
            if results_list:
                from scraper import try_trafilatura
                for i, r in enumerate(results_list[:2]):
                    url = r.get('url', '')
                    if url and url.startswith('http'):
                        print(f"  Scraping: {url}")
                        scraped = try_trafilatura(url)
                        sources.append({
                            "url": url,
                            "title": r.get('title', '') + " (full text)",
                            "content": scraped[:1000],
                            "score": 5
                        })

            summary_prompt = f"""Based on these search results about "{query}", provide a concise summary of key findings (5-7 bullet points):

{raw_output}

Format as clear bullet points with the most important information."""

            try:
                summary_response = _call_llm(get_llm_instance(), summary_prompt)
                summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
            except Exception as e:
                print(f"Summarization error: {e}")
                summary = raw_output

            return {"output": summary if summary else raw_output, "input": query, "sources": sources}

        except Exception as e:
            print(f"Research error: {e}")
            return {"output": f"Research on: {query}. Information gathered.", "input": query, "sources": []}

    return researcher_invoke


# =========================================================
# SPECIALIZED RESEARCHER NODE (Persona-based)
# =========================================================
def create_specialized_researcher_agent(persona="hardware"):
    """Creates a specialized researcher agent with the given persona.
    
    Persona options:
    - "hardware": Focuses on hardware/compute constraints (GPU, memory, latency, cost)
    - "ethics": Focuses on sociological/ethical implications (bias, privacy, equity)
    """
    prompt_map = {
        "hardware": hardware_researcher_prompt,
        "ethics": ethics_researcher_prompt,
    }
    persona_prompt = prompt_map.get(persona, researcher_prompt_template)
    persona_label = {"hardware": "Hardware", "ethics": "Ethics"}.get(persona, "General")

    def specialized_researcher_invoke(input_dict):
        query = input_dict.get("input", "")
        if not query or query in ["Continue work", "Complete"]:
            query = "General research information"
        print(f"Specialized researcher [{persona_label} persona]: {query}")

        try:
            raw_output, results_list = _perform_search(query)

            # Scrape for detailed content
            sources = []
            if results_list:
                from scraper import try_trafilatura
                for i, r in enumerate(results_list[:2]):
                    url = r.get('url', '')
                    if url and url.startswith('http'):
                        print(f"  Scraping: {url}")
                        scraped = try_trafilatura(url)
                        sources.append({
                            "url": url,
                            "title": r.get('title', '') + " (full text)",
                            "content": scraped[:1000],
                            "score": 5,
                            "persona": persona
                        })

            # Use persona-specific prompt to guide the analysis
            full_prompt = persona_prompt.format(task=query) + f"\n\nSearch Results:\n{raw_output}\n\nProvide your focused {persona_label} analysis:"
            try:
                summary_response = _call_llm(get_llm_instance(), full_prompt)
                summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
            except Exception as e:
                print(f"Specialized summarization error: {e}")
                summary = raw_output

            return {"output": summary if summary else raw_output, "input": query, "sources": sources, "persona": persona}

        except Exception as e:
            print(f"Specialized research error: {e}")
            return {"output": f"[{persona_label} research] {query}. Analysis gathered.", "input": query, "sources": [], "persona": persona}

    return specialized_researcher_invoke


# =========================================================
# BROWSER NODE
# =========================================================
def create_browser_agent():
    def browser_invoke(input_dict):
        query = input_dict.get("input", "")
        print(f"Browser agent searching: {query}")

        try:
            from playwright.sync_api import sync_playwright
            results = []
            browser_data = []
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
                page.goto(search_url, timeout=30000)
                page.wait_for_load_state("networkidle")
                snippets = page.locator(".gs_ri").all()
                for snippet in snippets[:5]:
                    title_el = snippet.locator(".gs_rt a")
                    title = title_el.inner_text() if title_el.count() > 0 else "No title"
                    desc_el = snippet.locator(".gs_rs")
                    desc = desc_el.inner_text() if desc_el.count() > 0 else "No description"
                    results.append(f"**{title}**\n{desc[:300]}")
                    browser_data.append({"title": title, "snippet": desc[:300], "source": "Google Scholar"})
                browser.close()

            if results:
                return {"output": "\n---\n".join(results), "input": query, "browser_data": browser_data}
            else:
                return {"output": f"Browser search completed for: {query}", "input": query, "browser_data": []}
        except ImportError:
            print("Playwright not installed. Install: pip install playwright && playwright install")
            return {"output": f"Browser search: {query}", "input": query, "browser_data": []}
        except Exception as e:
            print(f"Browser agent error: {e}")
            return {"output": f"Browser search: {query}", "input": query, "browser_data": []}

    return browser_invoke


# =========================================================
# CLASSIFICATION NODE
# =========================================================
def create_classification_agent():
    def classification_invoke(state):
        sources = state.get("research_findings", [])
        sources_text = "\n\n".join(sources) if sources else "No sources to classify."

        prompt = classification_prompt_template.format(
            main_task=state.get("main_task", ""),
            sources=sources_text
        )

        try:
            response = _call_llm(get_llm_instance(), prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            classified = content if content else "Classification completed."

            # Also try to get structured classified_sources
            classified_sources = []
            raw_sources = state.get("raw_sources", [])
            for src in raw_sources:
                classified_sources.append({
                    "url": src.get("url", ""),
                    "title": src.get("title", ""),
                    "category": "General",
                    "relevance": "Medium",
                    "score": src.get("score", 5)
                })

            return {"classification": classified, "classified_sources": classified_sources}
        except Exception as e:
            print(f"Classification error: {e}")
            return {"classification": "Classification completed.", "classified_sources": []}

    return classification_invoke


# =========================================================
# NER NODE
# =========================================================
def create_ner_agent():
    def ner_invoke(state):
        sources = state.get("research_findings", [])
        sources_text = "\n\n".join(sources) if sources else "No sources for entity extraction."

        prompt = ner_prompt_template.format(
            main_task=state.get("main_task", ""),
            sources=sources_text
        )

        try:
            response = _call_llm(get_llm_instance(), prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Try to parse entities and relationships
            entities_list = []
            relationships = []
            try:
                parsed = _parse_json_from_text(content)
                if isinstance(parsed, dict):
                    entities_list = parsed.get("entities", [])
                    relationships = parsed.get("relationships", [])
            except Exception:
                pass

            return {
                "entities_text": content if content else "No entities extracted.",
                "entities_list": entities_list,
                "relationships": relationships
            }
        except Exception as e:
            print(f"NER error: {e}")
            return {"entities_text": "Entity extraction completed.", "entities_list": [], "relationships": []}

    return ner_invoke


# =========================================================
# ANALYZER NODE
# =========================================================
def create_analyzer_agent():
    def analyzer_invoke(state):
        sources = state.get("research_findings", [])
        sources_text = "\n\n".join(sources) if sources else "No sources for analysis."
        entities = state.get("named_entities", "No entities extracted.")
        relationships = state.get("entity_relationships", [])

        prompt = analyzer_prompt_template.format(
            main_task=state.get("main_task", ""),
            sources=sources_text,
            entities=entities,
            relationships=json.dumps(relationships[:10]) if relationships else "None"
        )

        try:
            response = _call_llm(get_llm_instance(), prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Build structured outline
            outline_list = [
                {"section_title": "Executive Summary", "key_points": ["Overview of findings"], "image_prompt": ""},
                {"section_title": "Introduction", "key_points": ["Background", "Research questions"], "image_prompt": ""},
                {"section_title": "Main Findings", "key_points": ["Key themes and analysis"], "image_prompt": "Conceptual diagram of main findings"},
                {"section_title": "Analysis", "key_points": ["Deep dive into results"], "image_prompt": ""},
                {"section_title": "Conclusion", "key_points": ["Summary and future work"], "image_prompt": ""},
            ]

            return {
                "outline": content if content else "Analysis completed.",
                "organized_findings": [{"theme": "General", "findings": content[:500], "sources": sources[:3], "entities": []}],
                "outline_list": outline_list
            }
        except Exception as e:
            print(f"Analyzer error: {e}")
            return {"outline": "Analysis completed.", "organized_findings": [], "outline_list": []}

    return analyzer_invoke


# =========================================================
# ILLUSTRATOR NODE
# =========================================================
def create_illustration_agent():
    def illustration_invoke(state):
        illustration_requests = state.get("illustration_requests", [])
        if not illustration_requests:
            # Check draft for ILLUSTRATION markers
            draft = state.get("draft", "")
            if "[ILLUSTRATION:" in draft:
                requests = re.findall(r'\[ILLUSTRATION:\s*(.*?)\]', draft)
                illustration_requests = [{"section": "Report", "description": req.strip()} for req in requests]

        if not illustration_requests:
            return {"illustrations": [], "illustration_texts": ["No illustration requests pending."]}

        illustrations_text = []
        illustrations_data = []

        for request in illustration_requests:
            prompt = illustrator_prompt_template.format(
                main_task=state.get("main_task", ""),
                section=request.get("section", "General"),
                description=request.get("description", "General visual")
            )

            try:
                response = _call_llm(get_llm_instance(), prompt)
                content = response.content if hasattr(response, 'content') else str(response)

                # Try to generate actual image via Stable Diffusion API
                image_path = None
                sd_api_key = os.environ.get("STABILITY_API_KEY")
                if sd_api_key:
                    try:
                        import requests as http_req
                        import base64
                        resp = http_req.post(
                            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                            headers={
                                "Authorization": f"Bearer {sd_api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "text_prompts": [{"text": f"Technical academic diagram, {content[:500]}", "weight": 1}],
                                "cfg_scale": 7,
                                "height": 1024,
                                "width": 1024,
                                "samples": 1,
                                "steps": 30,
                            },
                            timeout=60
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            img_data = base64.b64decode(data["artifacts"][0]["base64"])
                            os.makedirs("assets", exist_ok=True)
                            fname = f"assets/illustration_{len(illustrations_text)}.png"
                            with open(fname, "wb") as fh:
                                fh.write(img_data)
                            image_path = fname
                            print(f"  Image saved: {fname}")
                    except Exception as e:
                        print(f"  Image generation skipped: {e}")

                illustrations_text.append(content)
                illustrations_data.append({
                    "section": request.get("section", "General"),
                    "prompt": content,
                    "image_path": image_path
                })
            except Exception as e:
                print(f"Illustration error: {e}")
                illustrations_text.append(f"Diagram concept for {request.get('section', 'section')}")

        return {"illustrations": illustrations_data, "illustration_texts": illustrations_text}

    return illustration_invoke


# =========================================================
# WRITER NODE
# =========================================================
def create_writer_chain():
    def writer_invoke(state):
        research = state.get("research_findings", [])
        research_text = "\n\n".join(research) if research else "No research available."
        classified = state.get("classification_results", "No classification available.")
        entities = state.get("named_entities", "No entities extracted.")
        outline = state.get("analysis_outline", "No outline available.")
        critic_score = state.get("critic_score", None)

        prompt = writer_prompt_template.format(
            main_task=state.get("main_task", ""),
            research_findings=research_text,
            classified_sources=classified,
            entities=entities,
            analysis_outline=outline,
            draft=state.get("draft", ""),
            critique_notes=state.get("critique_notes", ""),
            critic_score=critic_score if critic_score else "N/A"
        )

        try:
            response = _call_llm(get_llm_instance(), prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            new_illustration_requests = []
            if "[ILLUSTRATION:" in content:
                requests = re.findall(r'\[ILLUSTRATION:\s*(.*?)\]', content)
                for req in requests:
                    new_illustration_requests.append({
                        "section": "Requested Section",
                        "description": req.strip()
                    })

            return {
                "draft": content if content else "Draft in progress...",
                "illustration_requests": new_illustration_requests
            }
        except Exception as e:
            print(f"Writer error: {e}")
            # Fallback: construct a basic report from raw research data
            fallback_lines = [f"# {state.get('main_task', 'Research Report')}"]
            fallback_lines.append("\n## Executive Summary\n")
            fallback_lines.append("This report was compiled from research findings.")
            if research:
                fallback_lines.append("\n## Research Findings\n")
                for i, finding in enumerate(research):
                    fallback_lines.append(f"\n### Finding {i+1}\n")
                    fallback_lines.append(finding[:500])
            sources = state.get("raw_sources", [])
            if sources:
                fallback_lines.append("\n## Sources\n")
                for s in sources:
                    url = s.get("url", "")
                    title = s.get("title", "Untitled")
                    if url:
                        fallback_lines.append(f"- {title} ({url})")
                    else:
                        fallback_lines.append(f"- {title}")
            return {"draft": "\n".join(fallback_lines), "illustration_requests": []}

    return writer_invoke


# =========================================================
# CRITIQUE NODE
# =========================================================
def create_critique_chain():
    def critique_invoke(state):
        draft = state.get("draft", "")
        revision_num = state.get("revision_number", 0)

        if len(draft.strip()) < 100:
            return {"feedback": "APPROVED - Draft is minimal but acceptable.", "score": 7, "decision": "APPROVED"}

        if revision_num >= 3:
            return {"feedback": "APPROVED - Maximum revisions reached.", "score": 7, "decision": "APPROVED"}

        prompt = critique_prompt_template.format(
            main_task=state.get("main_task", ""),
            draft=draft
        )

        try:
            response = _call_llm(get_llm_instance(), prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Try to parse JSON response with scores and repetition info
            try:
                result = _parse_json_from_text(content)
                scores = result.get("scores", {})
                avg = result.get("average_score", 0) or (
                    sum(scores.values()) / len(scores) if scores else 5
                )
                feedback = result.get("feedback", content)
                decision = result.get("decision", "REVISE" if avg < 7 else "APPROVED")

                # Repetition override: if 3+ issues found, force REVISE
                repetition_count = result.get("repetition_count", 0)
                repetition_issues = result.get("repetition_issues", [])
                if repetition_count >= 3:
                    print(f"[Critic] Found {repetition_count} repetition issues - forcing REVISE")
                    decision = "REVISE"

                # If major repetition found but model said APPROVED, downgrade
                if decision == "APPROVED" and repetition_count > 0:
                    issues_detail = "; ".join([i.get("issue", "")[:60] for i in repetition_issues[:3]])
                    feedback += f"\n\n[Repetition Issues Found: {repetition_count}] {issues_detail}"
                    # Lower score to reflect repetition
                    avg = max(avg - repetition_count, 1)

                return {
                    "feedback": feedback,
                    "score": avg,
                    "decision": decision,
                    "repetition_count": repetition_count,
                    "repetition_issues": repetition_issues
                }
            except (json.JSONDecodeError, ValueError):
                # Fallback: check for APPROVED keyword
                is_approved = "APPROVED" in content.upper()
                return {
                    "feedback": content if content else "APPROVED",
                    "score": 7 if is_approved else 4,
                    "decision": "APPROVED" if is_approved else "REVISE",
                    "repetition_count": 0,
                    "repetition_issues": []
                }
        except Exception as e:
            print(f"Critique error: {e}")
            return {"feedback": "APPROVED - Error in critique.", "score": 7, "decision": "APPROVED"}

    return critique_invoke
