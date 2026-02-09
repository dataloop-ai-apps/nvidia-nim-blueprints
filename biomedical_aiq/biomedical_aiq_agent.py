"""
NVIDIA Biomedical AI-Q Research Agent - Dataloop Service Runner

Pipeline flow:
  Input → [NVIDIA AI-Q Research] → [NIM Nemotron 49B] → [NVIDIA AI-Q Reflect]
                                                             ├─ "research_more" → back to Research
                                                             └─ "finalize" →
          [NVIDIA AI-Q Report Prep] → [NIM Llama 70B] → [NVIDIA AI-Q Output]

Internal LLM calls (query generation, summarization, relevancy) use langchain_nvidia_ai_endpoints directly.
Pipeline NIM model nodes handle reflection (Nemotron 49B) and report generation (Llama 70B).
"""

import dtlpy as dl
import os
import json
import logging
import re
from typing import List, Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import parse_json_markdown
from tavily import TavilyClient

from biomedical_aiq.prompts import (
    QUERY_WRITER_INSTRUCTIONS,
    SUMMARIZER_INSTRUCTIONS,
    REPORT_EXTENDER,
    REFLECTION_INSTRUCTIONS,
    RELEVANCY_CHECKER,
    FINALIZE_REPORT,
    CHECK_VIRTUAL_SCREENING,
    CHECK_PROTEIN_MOLECULE_FOUND,
    COMBINE_VS_INTO_REPORT,
)
from biomedical_aiq.virtual_screening import run_virtual_screening

logger = logging.getLogger('[BiomedicalAIQ]')

# Default configuration
DEFAULT_NUM_REFLECTIONS = 2
DEFAULT_NUM_QUERIES = 3
DEFAULT_REASONING_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
DEFAULT_GENERATION_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class BiomedicalAIQAgent(dl.BaseServiceRunner):
    """Service runner for the NVIDIA Biomedical AI-Q Research Agent pipeline."""

    def __init__(self):
        # API keys from Dataloop integrations (set as env vars)
        nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
        if nvidia_api_key is None:
            raise ValueError("Missing NVIDIA_API_KEY environment variable.")

        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if tavily_api_key is None:
            raise ValueError("Missing TAVILY_API_KEY environment variable.")

        self.tavily_client = TavilyClient(api_key=tavily_api_key)

        # Internal LLM clients for direct API calls (query gen, summarization, relevancy)
        base_url = os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)

        self.reasoning_llm = ChatNVIDIA(
            model=os.environ.get("REASONING_MODEL", DEFAULT_REASONING_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.2,
            max_tokens=4096,
        )
        self.generation_llm = ChatNVIDIA(
            model=os.environ.get("GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.2,
            max_tokens=4096,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_state(self, item: dl.Item) -> dict:
        """Get the aiq_state from item metadata, or initialize it."""
        item_meta = item.metadata.get('user', {})
        return item_meta.get('aiq_state', {})

    def _set_state(self, item: dl.Item, state: dict):
        """Save aiq_state to item metadata."""
        item.metadata.setdefault('user', {})
        item.metadata['user']['aiq_state'] = state
        item = item.update(system_metadata=True)
        return item

    def _get_main_item(self, item: dl.Item) -> dl.Item:
        """Get the main item from a temp prompt item's metadata."""
        main_item_id = item.metadata.get('user', {}).get('main_item')
        if main_item_id:
            return dl.items.get(item_id=main_item_id)
        return item

    def _create_prompt_item(self, item: dl.Item, prompt_text: str, prompt_name: str, main_item: dl.Item = None):
        """Create a temporary PromptItem for NIM model nodes (same pattern as report_generation)."""
        if main_item is None:
            main_item = item

        prompt_item = dl.PromptItem(name=prompt_name)
        prompt = dl.Prompt(key='1')
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=prompt_text)
        prompt_item.prompts.append(prompt)

        item_prompt = item.dataset.items.upload(
            prompt_item,
            overwrite=True,
            remote_path=f"/.dataloop/aiq_temp_{main_item.name.replace('.json', '')}/",
            item_metadata={
                "user": {
                    "main_item": main_item.id
                }
            }
        )
        return item_prompt

    def _get_annotation_response(self, item: dl.Item) -> str:
        """Read the LLM response from the item's annotations (set by NIM model predict node)."""
        annotations = item.annotations.list()
        if annotations.items_count > 0:
            # Get the most recent annotation
            latest = annotations.items[-1]
            return latest.coordinates if hasattr(latest, 'coordinates') else str(latest)
        return ""

    def _invoke_reasoning_llm(self, prompt_text: str) -> str:
        """Call the reasoning LLM directly via langchain_nvidia_ai_endpoints."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert biomedical research assistant. Think step by step."),
            ("human", "{input}"),
        ])
        chain = prompt | self.reasoning_llm
        result = chain.invoke({"input": prompt_text})
        return result.content

    def _invoke_generation_llm(self, prompt_text: str) -> str:
        """Call the generation LLM directly via langchain_nvidia_ai_endpoints."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert biomedical report writer."),
            ("human", "{input}"),
        ])
        chain = prompt | self.generation_llm
        result = chain.invoke({"input": prompt_text})
        return result.content

    def _parse_json_response(self, text: str) -> dict | list | None:
        """Parse JSON from LLM response, handling <think> tags and markdown blocks."""
        # Remove <think>...</think> tags
        cleaned = text
        while "<think>" in cleaned and "</think>" in cleaned:
            start = cleaned.find("<think>")
            end = cleaned.find("</think>") + len("</think>")
            cleaned = cleaned[:start] + cleaned[end:]
        # Also handle unclosed <think> tags
        if "<think>" in cleaned:
            start = cleaned.find("<think>")
            cleaned = cleaned[:start]

        try:
            return parse_json_markdown(cleaned)
        except Exception:
            # Try to find JSON in the text
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            # Try direct JSON parse
            try:
                return json.loads(cleaned.strip())
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response: {cleaned[:200]}")
                return None

    def _extract_params_from_prompt(self, prompt_text: str) -> dict:
        """Extract research parameters from the user's input prompt.

        Expected format:
        Topic: <topic>
        Report Organization: <organization>
        Number of queries: <int> (optional, default 3)
        Number of reflections: <int> (optional, default 2)
        """
        params = {
            'topic': '',
            'report_organization': '',
            'num_queries': DEFAULT_NUM_QUERIES,
            'num_reflections': DEFAULT_NUM_REFLECTIONS,
        }
        lines = prompt_text.strip().split('\n')

        for i, line in enumerate(lines):
            if line.strip().lower().startswith('topic:'):
                params['topic'] = line.split(':', 1)[1].strip()
            elif line.strip().lower().startswith('number of queries:'):
                try:
                    params['num_queries'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.strip().lower().startswith('number of reflections:'):
                try:
                    params['num_reflections'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

        # Extract report organization (everything between "Report Organization:" and next known field or end)
        org_lines = []
        capture = False
        for line in lines:
            if line.strip().lower().startswith('report organization:'):
                capture = True
                rest = line.split(':', 1)[1].strip()
                if rest:
                    org_lines.append(rest)
                continue
            elif capture and any(line.strip().lower().startswith(k) for k in
                                ['number of queries:', 'number of reflections:', 'topic:']):
                capture = False
                continue
            if capture:
                org_lines.append(line)

        params['report_organization'] = '\n'.join(org_lines).strip()

        # If no structured format found, use the whole text as topic
        if not params['topic'] and not params['report_organization']:
            params['topic'] = prompt_text.strip()
            params['report_organization'] = "Introduction, Key Findings, Analysis, Conclusion"

        return params

    # ─── Tavily Search ────────────────────────────────────────────────────────

    def _tavily_search(self, query: str) -> dict:
        """Perform a web search using Tavily API."""
        try:
            result = self.tavily_client.search(
                query,
                max_results=5,
                include_raw_content=True,
                topic="general"
            )
            return result
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {"results": []}

    def _format_search_results(self, search_results: list) -> str:
        """Format search results into a source string for LLM consumption."""
        formatted = "Sources:\n\n"
        for i, result in enumerate(search_results, 1):
            formatted += f"Source {i}: {result.get('title', 'N/A')}\n"
            formatted += f"URL: {result.get('url', 'N/A')}\n"
            formatted += f"Content: {result.get('content', 'N/A')}\n"
            raw = result.get('raw_content', '') or ''
            if len(raw) > 4000:
                raw = raw[:4000] + "... [truncated]"
            if raw:
                formatted += f"Full content: {raw}\n"
            formatted += "\n"
        return formatted.strip()

    def _collect_citations(self, search_results: list) -> list:
        """Extract citation strings from search results."""
        citations = []
        for result in search_results:
            title = result.get('title', 'N/A')
            url = result.get('url', 'N/A')
            citations.append(f"- {title}: {url}")
        return citations

    # ─── Pipeline Node Functions ──────────────────────────────────────────────

    def init_research(self, item: dl.Item):
        """
        Pipeline node: NVIDIA AI-Q Init (ROOT node - no incoming connections)

        Simple pass-through that serves as the pipeline entry point.
        Needed because the Research node receives a loop-back connection
        and Dataloop root nodes cannot have incoming connections.

        Returns: the item unchanged → goes to execute_research
        """
        logger.info("=== NVIDIA AI-Q Init node ===")
        return item

    def execute_research(self, item: dl.Item):
        """
        Pipeline node: NVIDIA AI-Q Research

        First call: Extracts user query, generates search queries, executes searches,
                    summarizes sources, prepares reflection prompt.
        Subsequent calls (loop): Reads reflection feedback, does targeted research,
                                 extends the report, prepares new reflection prompt.

        Returns: temp PromptItem with reflection prompt → goes to NIM Nemotron 49B
        """
        logger.info("=== NVIDIA AI-Q Research node ===")

        # Determine if this is the main item or we're in a loop
        state = self._get_state(item)

        if not state:
            # ── FIRST CALL: Initialize from user's query ──
            logger.info("First research iteration - initializing")
            prompt_item = dl.PromptItem.from_item(item)
            prompts_json = prompt_item.to_json()['prompts']
            first_key = list(prompts_json.keys())[0]
            prompt_text = prompts_json[first_key][0]['value']

            params = self._extract_params_from_prompt(prompt_text)
            logger.info(f"Research topic: {params['topic']}")

            # Step 1: Generate search queries using reasoning LLM (direct API)
            query_prompt = QUERY_WRITER_INSTRUCTIONS.format(
                number_of_queries=params['num_queries'],
                topic=params['topic'],
                report_organization=params['report_organization']
            )
            query_response = self._invoke_reasoning_llm(query_prompt)
            queries = self._parse_json_response(query_response)
            if not queries or not isinstance(queries, list):
                queries = [{"query": params['topic'], "report_section": "All", "rationale": "Main topic"}]
            logger.info(f"Generated {len(queries)} search queries")

            # Step 2: Execute searches for each query
            all_results = []
            all_citations = []
            for q in queries:
                query_text = q.get('query', q) if isinstance(q, dict) else str(q)
                search_result = self._tavily_search(query_text)
                results = search_result.get('results', [])
                all_results.extend(results)
                all_citations.extend(self._collect_citations(results))

            source_str = self._format_search_results(all_results)
            logger.info(f"Collected {len(all_results)} search results")

            # Step 3: Summarize sources using generation LLM (direct API)
            summary_prompt = SUMMARIZER_INSTRUCTIONS.format(
                report_organization=params['report_organization'],
                source=source_str
            )
            running_summary = self._invoke_generation_llm(summary_prompt)
            logger.info("Initial summary generated")

            # Save state
            state = {
                'topic': params['topic'],
                'report_organization': params['report_organization'],
                'num_reflections': params['num_reflections'],
                'num_queries': params['num_queries'],
                'iteration': 0,
                'running_summary': running_summary,
                'citations': all_citations,
                'queries': queries,
            }
            item = self._set_state(item, state)

        else:
            # ── SUBSEQUENT CALL: Loop iteration - do targeted research based on reflection ──
            iteration = state.get('iteration', 0)
            logger.info(f"Research iteration {iteration + 1} - using reflection feedback")

            reflection_query = state.get('last_reflection_query', state.get('topic', ''))

            # Search for the reflection query
            search_result = self._tavily_search(reflection_query)
            results = search_result.get('results', [])
            new_citations = self._collect_citations(results)
            source_str = self._format_search_results(results)

            # Extend the report with new sources
            extend_prompt = REPORT_EXTENDER.format(
                report=state['running_summary'],
                source=source_str
            )
            updated_summary = self._invoke_generation_llm(extend_prompt)

            # Update state
            state['running_summary'] = updated_summary
            state['citations'] = state.get('citations', []) + new_citations
            item = self._set_state(item, state)

        # ── Prepare reflection prompt for NIM Nemotron 49B ──
        reflection_prompt = REFLECTION_INSTRUCTIONS.format(
            topic=state['topic'],
            report_organization=state['report_organization'],
            report=state['running_summary']
        )

        # Create temp PromptItem with the reflection prompt
        temp_item = self._create_prompt_item(
            item=item,
            prompt_text=reflection_prompt,
            prompt_name=f"reflection_{state.get('iteration', 0)}",
            main_item=item
        )

        logger.info("Returning reflection prompt to NIM Nemotron 49B")
        return temp_item

    def evaluate_reflection(self, item: dl.Item, context: dl.Context, progress: dl.Progress):
        """
        Pipeline node: NVIDIA AI-Q Reflect (with action output)

        Reads the NIM Nemotron 49B reflection response.
        Decides whether to loop back for more research or finalize.

        Actions:
          - "research_more": loops back to execute_research
          - "finalize": continues to prepare_report
        """
        logger.info("=== NVIDIA AI-Q Reflect node ===")

        # Read reflection response from the NIM model's annotation
        reflection_response = self._get_annotation_response(item)
        logger.info(f"Reflection response: {reflection_response[:200]}...")

        # Get the main item
        main_item = self._get_main_item(item)
        state = self._get_state(main_item)

        # Parse the reflection to extract the follow-up query
        reflection_obj = self._parse_json_response(reflection_response)
        if reflection_obj and isinstance(reflection_obj, dict) and 'query' in reflection_obj:
            state['last_reflection_query'] = reflection_obj['query']
            logger.info(f"Reflection query: {reflection_obj['query']}")
        else:
            state['last_reflection_query'] = state.get('topic', '')
            logger.warning("Could not parse reflection, using topic as fallback query")

        # Increment iteration
        state['iteration'] = state.get('iteration', 0) + 1
        max_reflections = state.get('num_reflections', DEFAULT_NUM_REFLECTIONS)

        main_item = self._set_state(main_item, state)

        if state['iteration'] < max_reflections:
            logger.info(f"Iteration {state['iteration']}/{max_reflections} - routing to research_more")
            progress.update(action="research_more")
        else:
            logger.info(f"Completed {state['iteration']}/{max_reflections} reflections - routing to finalize")
            progress.update(action="finalize")

        return main_item

    def prepare_report(self, item: dl.Item):
        """
        Pipeline node: NVIDIA AI-Q Report Prep

        Checks for virtual screening intent, runs VS if needed,
        prepares the final report prompt for NIM Llama 70B.

        Returns: temp PromptItem with finalize prompt → goes to NIM Llama 70B
        """
        logger.info("=== NVIDIA AI-Q Report Prep node ===")

        state = self._get_state(item)
        report = state.get('running_summary', '')
        topic = state.get('topic', '')
        report_org = state.get('report_organization', '')

        # ── Check if virtual screening is needed ──
        vs_check_prompt = CHECK_VIRTUAL_SCREENING.format(
            topic=topic,
            report_organization=report_org
        )
        vs_response = self._invoke_reasoning_llm(vs_check_prompt)
        vs_obj = self._parse_json_response(vs_response)

        do_vs = False
        if vs_obj and isinstance(vs_obj, dict):
            do_vs = vs_obj.get('intention', 'no').lower() == 'yes'

        if do_vs:
            logger.info("Virtual screening is intended - running VS pipeline")

            # Find target protein and molecule from research
            pm_prompt = CHECK_PROTEIN_MOLECULE_FOUND.format(
                topic=topic,
                knowledge_sources=report[:3000]  # Limit context size
            )
            pm_response = self._invoke_reasoning_llm(pm_prompt)
            pm_obj = self._parse_json_response(pm_response)

            target_protein = ""
            recent_molecule = ""
            vs_queries_info = ""

            if pm_obj and isinstance(pm_obj, dict):
                if 'target_protein' in pm_obj and 'recent_small_molecule_therapy' in pm_obj:
                    target_protein = pm_obj['target_protein']
                    recent_molecule = pm_obj['recent_small_molecule_therapy']
                elif 'query' in pm_obj:
                    # Need to search for missing info
                    vs_query = pm_obj['query']
                    vs_search = self._tavily_search(vs_query)
                    vs_results = vs_search.get('results', [])
                    vs_source = self._format_search_results(vs_results)
                    vs_queries_info = f"Query: {vs_query}\nResults: {vs_source[:2000]}"

                    # Try again with new info
                    pm_prompt2 = CHECK_PROTEIN_MOLECULE_FOUND.format(
                        topic=topic,
                        knowledge_sources=report[:2000] + "\n" + vs_source[:2000]
                    )
                    pm_response2 = self._invoke_reasoning_llm(pm_prompt2)
                    pm_obj2 = self._parse_json_response(pm_response2)
                    if pm_obj2 and isinstance(pm_obj2, dict):
                        target_protein = pm_obj2.get('target_protein', '')
                        recent_molecule = pm_obj2.get('recent_small_molecule_therapy', '')

            # Run virtual screening
            vs_info = run_virtual_screening(target_protein, recent_molecule)
            logger.info("Virtual screening completed")

            # Combine VS results into the report
            combine_prompt = COMBINE_VS_INTO_REPORT.format(
                report_organization=report_org,
                report=report,
                vs_queries=vs_queries_info,
                vs_queries_results="",
                vs_info=vs_info
            )
            updated_report = self._invoke_generation_llm(combine_prompt)
            state['running_summary'] = updated_report
            state['vs_info'] = vs_info
        else:
            logger.info("Virtual screening not needed")

        # ── Prepare the finalization prompt ──
        citations = state.get('citations', [])
        citations_str = "\n".join(citations) if citations else ""

        finalize_prompt = FINALIZE_REPORT.format(
            report=state['running_summary'],
            report_organization=report_org
        )

        # Store citations for later
        state['final_citations'] = citations_str
        item = self._set_state(item, state)

        # Create temp PromptItem with the finalize prompt
        temp_item = self._create_prompt_item(
            item=item,
            prompt_text=finalize_prompt,
            prompt_name="finalize_report",
            main_item=item
        )

        logger.info("Returning finalize prompt to NIM Llama 70B")
        return temp_item

    def write_output(self, item: dl.Item):
        """
        Pipeline node: NVIDIA AI-Q Output

        Reads the final report from NIM Llama 70B's annotation.
        Adds the report as an assistant response to the original PromptItem.
        Also uploads a markdown file of the report.

        Returns: the original (main) PromptItem with the final report
        """
        logger.info("=== NVIDIA AI-Q Output node ===")

        # Read the final report from NIM Llama's annotation
        final_report = self._get_annotation_response(item)

        # Get the main item
        main_item = self._get_main_item(item)
        state = self._get_state(main_item)

        # Append sources
        citations_str = state.get('final_citations', '')
        if citations_str:
            final_report = f"{final_report}\n\n## Sources\n\n{citations_str}"

        # Clean up any remaining <think> tags
        while "<think>" in final_report and "</think>" in final_report:
            start = final_report.find("<think>")
            end = final_report.find("</think>") + len("</think>")
            final_report = final_report[:start] + final_report[end:]

        # Add the report as assistant response to the original PromptItem
        prompt_item = dl.PromptItem.from_item(main_item)
        prompt_item.add(
            prompt_key='1',
            message={
                "role": "assistant",
                "content": [{
                    "mimetype": dl.PromptType.TEXT,
                    "value": final_report
                }]
            },
            model_info={
                'name': 'nvidia-biomedical-aiq',
                'confidence': 1.0,
                'model_id': 'nvidia-biomedical-aiq-1'
            }
        )

        # Upload a markdown file of the report
        try:
            topic = state.get('topic', 'report')
            safe_name = re.sub(r'[^\w\s-]', '', topic)[:50].strip().replace(' ', '_')
            md_filename = f"biomedical_report_{safe_name}.md"

            # Write markdown file locally and upload
            local_path = os.path.join("/tmp", md_filename)
            with open(local_path, 'w') as f:
                f.write(final_report)

            main_item.dataset.items.upload(
                local_path=local_path,
                remote_path=f"/.dataloop/aiq_reports/",
                overwrite=True
            )
            logger.info(f"Uploaded report as {md_filename}")
        except Exception as e:
            logger.warning(f"Could not upload markdown report: {e}")

        logger.info("=== Pipeline complete ===")
        return main_item
