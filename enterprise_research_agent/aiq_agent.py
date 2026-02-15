"""
NVIDIA AIQ Enterprise Research Agent - Dataloop Service Runner

Replicated from NVIDIA AIQ Research Assistant Blueprint:
https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant

Pipeline flow:
  Input -> [Init] -> [AIQ Agent]
                       |-- "research" -> [Research Node] -> [AIQ Agent] (cycle)
                       '-- "generate_report" -> [NIM Llama 3.3 70B Instruct] (end)

The Agent node is the brain: it plans, summarizes, reflects, and decides next action.
The Research node handles combined RAG-first + web search fallback with LLM-as-judge.
The final report is formatted by the NIM Llama predict node using nearestItems context.
"""

import dtlpy as dl
import os
import json
import logging
import re
import time
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.json import parse_json_markdown
from tavily import TavilyClient

from enterprise_research_agent.prompts import (
    QUERY_WRITER_INSTRUCTIONS,
    SUMMARIZER_INSTRUCTIONS,
    REPORT_EXTENDER,
    REFLECTION_INSTRUCTIONS,
    RELEVANCY_CHECKER,
)

logger = logging.getLogger('[AIQ-Enterprise-Research]')

# Default configuration (matching NVIDIA blueprint)
DEFAULT_NUM_REFLECTIONS = 2
DEFAULT_NUM_QUERIES = 3
DEFAULT_REASONING_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class AIQEnterpriseAgent(dl.BaseServiceRunner):
    """Service runner for the NVIDIA AIQ Enterprise Research Agent pipeline."""

    def __init__(self):
        nvidia_api_key = os.environ.get("NGC_API_KEY")
        if nvidia_api_key is None:
            raise ValueError("Missing NGC_API_KEY environment variable.")

        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if tavily_api_key is None:
            raise ValueError("Missing TAVILY_API_KEY environment variable.")

        self.tavily_client = TavilyClient(api_key=tavily_api_key)

        base_url = os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)

        # Reasoning LLM for planning, summarization, reflection, relevancy checking
        self.reasoning_llm = ChatNVIDIA(
            model=os.environ.get("REASONING_MODEL", DEFAULT_REASONING_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.5,
            max_tokens=5000,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _upload_data_file(self, dataset, data: str, remote_path: str, filename: str) -> dl.Item:
        """Upload a text/JSON string as a hidden file. Returns the uploaded Item."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write(data)
            local_path = f.name
        try:
            return dataset.items.upload(
                local_path=local_path,
                remote_path=remote_path,
                remote_name=filename,
                overwrite=True,
            )
        finally:
            os.remove(local_path)

    def _download_data_file(self, item_id: str) -> str:
        """Download a data file and return its text content."""
        data_item = dl.items.get(item_id=item_id)
        buf = data_item.download(save_locally=False)
        if hasattr(buf, 'read'):
            return buf.read().decode('utf-8', errors='replace')
        return str(buf)

    def _get_state(self, item: dl.Item) -> dict:
        """Load agent state from a JSON file referenced in item metadata."""
        state_file_id = item.metadata.get('user', {}).get('aiq_state_file_id')
        if not state_file_id:
            return {}
        try:
            content = self._download_data_file(state_file_id)
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Could not load state file {state_file_id}: {e}")
            return {}

    def _set_state(self, item: dl.Item, state: dict) -> dl.Item:
        """Save agent state as a JSON file and store the file ID in item metadata."""
        state_json = json.dumps(state, ensure_ascii=False)
        state_item = self._upload_data_file(
            dataset=item.dataset,
            data=state_json,
            remote_path="/.dataloop/aiq_state/",
            filename=f"state_{item.id[:12]}.json",
        )
        item.metadata.setdefault('user', {})
        item.metadata['user']['aiq_state_file_id'] = state_item.id
        item = item.update(system_metadata=True)
        return item

    def _get_main_item(self, item: dl.Item) -> dl.Item:
        """Get the original PromptItem from a temp item's metadata."""
        main_item_id = item.metadata.get('user', {}).get('main_item')
        if main_item_id:
            return dl.items.get(item_id=main_item_id)
        return item

    def _is_temp_item(self, item: dl.Item) -> bool:
        """Check if this is a temp item created by the agent."""
        return 'main_item' in item.metadata.get('user', {})

    def _create_temp_item(self, main_item: dl.Item, content: str, name: str) -> dl.Item:
        """Create a temp item in a hidden folder for passing data between nodes."""
        safe_name = re.sub(r'[^\w\s-]', '', name)[:50].strip().replace(' ', '_')
        filename = f"{safe_name}.txt"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            local_path = f.name

        try:
            temp_item = main_item.dataset.items.upload(
                local_path=local_path,
                remote_path=f"/.dataloop/aiq_temp_{main_item.id[:8]}/",
                remote_name=filename,
                overwrite=True,
                item_metadata={
                    "user": {
                        "main_item": main_item.id
                    }
                }
            )
            return temp_item
        finally:
            os.remove(local_path)

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> reasoning blocks from LLM output."""
        cleaned = text
        while "<think>" in cleaned and "</think>" in cleaned:
            start = cleaned.find("<think>")
            end = cleaned.find("</think>") + len("</think>")
            cleaned = cleaned[:start] + cleaned[end:]
        # If there's an unclosed <think> tag, remove everything from it onward
        if "<think>" in cleaned:
            cleaned = cleaned[:cleaned.find("<think>")]
        return cleaned.strip()

    def _invoke_llm(self, prompt_text: str, system_prompt: str = "You are an expert research assistant.") -> str:
        """Call the reasoning LLM and strip any <think> reasoning tags."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        chain = prompt | self.reasoning_llm
        result = chain.invoke({"input": prompt_text})
        return self._strip_think_tags(result.content)

    def _parse_json_response(self, text: str) -> dict | list | None:
        """Parse JSON from LLM response, handling <think> tags and markdown blocks."""
        cleaned = text
        while "<think>" in cleaned and "</think>" in cleaned:
            start = cleaned.find("<think>")
            end = cleaned.find("</think>") + len("</think>")
            cleaned = cleaned[:start] + cleaned[end:]
        if "<think>" in cleaned:
            cleaned = cleaned[:cleaned.find("<think>")]

        try:
            return parse_json_markdown(cleaned)
        except Exception:
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            try:
                return json.loads(cleaned.strip())
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response: {cleaned[:200]}")
                return None

    def _extract_params_from_prompt(self, prompt_text: str) -> dict:
        """Extract research parameters from the user's input prompt."""
        params = {
            'topic': '',
            'report_organization': '',
            'num_queries': DEFAULT_NUM_QUERIES,
            'num_reflections': DEFAULT_NUM_REFLECTIONS,
        }
        lines = prompt_text.strip().split('\n')

        for line in lines:
            lower = line.strip().lower()
            if lower.startswith('topic:'):
                params['topic'] = line.split(':', 1)[1].strip()
            elif lower.startswith('number of queries:'):
                try:
                    params['num_queries'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif lower.startswith('number of reflections:'):
                try:
                    params['num_reflections'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

        # Extract report organization
        org_lines = []
        capture = False
        for line in lines:
            lower = line.strip().lower()
            if lower.startswith('report organization:'):
                capture = True
                rest = line.split(':', 1)[1].strip()
                if rest:
                    org_lines.append(rest)
                continue
            elif capture and any(lower.startswith(k) for k in
                                ['number of queries:', 'number of reflections:', 'topic:']):
                capture = False
                continue
            if capture:
                org_lines.append(line)

        params['report_organization'] = '\n'.join(org_lines).strip()

        # If no structured format, use whole text as topic
        if not params['topic'] and not params['report_organization']:
            params['topic'] = prompt_text.strip()
            params['report_organization'] = "Introduction, Key Findings, Analysis, Conclusion"

        return params

    # ─── RAG Search Helper ────────────────────────────────────────────────────

    def _execute_rag_query(self, query: str, rag_pipeline_id: str, dataset) -> str:
        """Execute a RAG pipeline query and return the response text."""
        try:
            rag_pipeline = dl.pipelines.get(pipeline_id=rag_pipeline_id)
            if not rag_pipeline.installed:
                logger.warning(f"RAG pipeline not active - skipping")
                return ""

            # Create a PromptItem with the query
            safe_name = re.sub(r'[^\w\s-]', '', query)[:30].strip().replace(' ', '_')
            prompt_item = dl.PromptItem(name=f"rag_{safe_name}_{int(time.time())}")
            prompt_item.add(
                message={
                    "role": "user",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": query}]
                }
            )

            rag_prompt_item = dataset.items.upload(
                prompt_item,
                remote_path="/.dataloop/aiq_rag_queries/",
                overwrite=True,
            )
            logger.info(f"Created RAG PromptItem for query: '{query[:60]}...'")

            # Execute the RAG pipeline
            execution = rag_pipeline.execute(
                execution_input={"item": [rag_prompt_item.id]}
            )

            # Wait for completion
            max_wait_seconds = 300
            poll_interval = 5
            elapsed = 0
            while elapsed < max_wait_seconds:
                time.sleep(poll_interval)
                elapsed += poll_interval
                success, response = dl.client_api.gen_request(
                    req_type="get",
                    path=f"/pipelines/{rag_pipeline_id}/executions/{execution.id}"
                )
                if success:
                    status = response.json().get('status', '')
                    if status in ['success', 'completed']:
                        break
                    elif status in ['failed', 'error']:
                        logger.error(f"RAG execution failed")
                        return ""

            # Read the response
            rag_prompt_item = dl.items.get(item_id=rag_prompt_item.id)
            updated_prompt = dl.PromptItem.from_item(rag_prompt_item)
            prompts_json = updated_prompt.to_json().get('prompts', {})
            rag_answer = ''
            for key in prompts_json:
                for msg in prompts_json[key]:
                    if msg.get('mimetype') == 'application/text' and msg.get('role', '') != 'user':
                        rag_answer = msg.get('value', '')

            if not rag_answer:
                annotations = rag_prompt_item.annotations.list()
                for annotation in annotations:
                    ann_data = annotation.to_json()
                    coordinates = ann_data.get('coordinates', {})
                    if isinstance(coordinates, dict):
                        for key in ['response', 'text', 'value']:
                            if key in coordinates:
                                rag_answer = str(coordinates[key])
                                break
                    if rag_answer:
                        break

            return rag_answer

        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return ""

    # ─── Relevancy Check (LLM-as-Judge) ──────────────────────────────────────

    def _check_relevancy(self, query: str, document: str) -> dict:
        """Check if a RAG response is relevant to the query using LLM-as-judge."""
        prompt = RELEVANCY_CHECKER.format(query=query, document=document)
        response = self._invoke_llm(prompt, system_prompt="You are a relevancy checker.")
        result = self._parse_json_response(response)
        if isinstance(result, dict) and 'score' in result:
            return result
        return {"score": "no"}

    # ─── Tavily Web Search ───────────────────────────────────────────────────

    def _tavily_search(self, query: str) -> list:
        """Perform a web search using Tavily API, filter by relevance > 0.6."""
        try:
            result = self.tavily_client.search(
                query, max_results=5, include_raw_content=True, topic="general"
            )
            # Filter by relevance score > 0.6 (matching NVIDIA)
            return [
                r for r in result.get('results', [])
                if float(r.get('score', 0)) > 0.6
            ]
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    # ─── Source Deduplication and XML Formatting (NVIDIA pattern) ─────────────

    def _deduplicate_and_format_sources(
        self, queries: list, rag_answers: list, relevancies: list, web_answers: list
    ) -> str:
        """Convert RAG and fallback results into XML <sources> structure.
        Matches NVIDIA's deduplicate_and_format_sources pattern.
        If RAG was relevant, use RAG answer; otherwise use web answer."""
        root = ET.Element("sources")

        for query_obj, rag_ans, relevancy, web_ans in zip(queries, rag_answers, relevancies, web_answers):
            query_text = query_obj.get('query', str(query_obj)) if isinstance(query_obj, dict) else str(query_obj)

            source_elem = ET.SubElement(root, "source")
            query_elem = ET.SubElement(source_elem, "query")
            query_elem.text = query_text
            answer_elem = ET.SubElement(source_elem, "answer")

            # If RAG was relevant, use RAG answer; else fallback to web
            if relevancy.get("score") == "yes" or not web_ans:
                answer_elem.text = rag_ans if rag_ans else (web_ans or "No relevant result found")
            else:
                answer_elem.text = web_ans

        return ET.tostring(root, encoding="unicode")

    def _format_web_citation(self, query: str, result: dict) -> str:
        """Format a web search result as a citation string (NVIDIA pattern)."""
        return f"---\nQUERY:\n{query}\n\nANSWER:\n{result.get('content', '')}\n\nCITATION:\n{result.get('url', '').strip()}\n"

    # ─── Process Single Query (NVIDIA pattern) ──────────────────────────────

    def _process_single_query(self, query_obj: dict, rag_pipeline_id: str, dataset) -> dict:
        """Process a single query: RAG-first -> relevancy check -> web fallback.
        Matches NVIDIA's process_single_query pattern."""
        query_text = query_obj.get('query', str(query_obj)) if isinstance(query_obj, dict) else str(query_obj)
        logger.info(f"Processing query: '{query_text[:80]}...'")

        rag_answer = ""
        rag_citation = ""
        relevancy = {"score": "no"}
        web_answer = ""
        web_citation = ""

        # Step 1: RAG search (if configured)
        if rag_pipeline_id:
            rag_answer = self._execute_rag_query(query_text, rag_pipeline_id, dataset)
            if rag_answer:
                rag_citation = f"---\nQUERY:\n{query_text}\n\nANSWER:\n{rag_answer}\n\nCITATION:\nRAG Pipeline\n"
                logger.info(f"RAG returned answer length: {len(rag_answer)}")

                # Step 2: LLM-as-judge relevancy check
                relevancy = self._check_relevancy(query_text, rag_answer)
                logger.info(f"RAG relevancy for '{query_text[:40]}...': {relevancy.get('score')}")

        # Step 3: Conditional web fallback
        if relevancy.get("score") == "no":
            web_results = self._tavily_search(query_text)
            if web_results:
                web_answers = [r.get('content', '') for r in web_results]
                web_citations = [self._format_web_citation(query_text, r) for r in web_results]
                web_answer = "\n".join(web_answers)
                web_citation = "\n".join(web_citations)
                logger.info(f"Web search returned {len(web_results)} relevant results")
            else:
                web_answer = "No relevant result found in web search"
                logger.info("Web search returned no relevant results")
        else:
            web_answer = "Web not searched since RAG provided relevant answer"
            logger.info("Skipping web search - RAG answer was relevant")

        return {
            "query": query_text,
            "rag_answer": rag_answer,
            "rag_citation": rag_citation,
            "relevancy": relevancy,
            "web_answer": web_answer,
            "web_citation": web_citation,
        }

    # ─── Pipeline Node Functions ──────────────────────────────────────────────

    def init_research(self, item: dl.Item, rag_pipeline_id: str = None):
        """Pipeline node: Init (ROOT)
        Validates configuration and stores in state."""
        logger.info("=== AIQ Init node ===")
        state = self._get_state(item)
        if rag_pipeline_id:
            try:
                rag_pipeline = dl.pipelines.get(pipeline_id=rag_pipeline_id)
                if rag_pipeline.status != 'Installed':
                    logger.warning(
                        f"RAG pipeline '{rag_pipeline.name}' is not installed/active. "
                        f"RAG retrieval will be skipped."
                    )
                else:
                    state['rag_pipeline_id'] = rag_pipeline_id
                    logger.info(f"RAG pipeline validated: {rag_pipeline.name} (ID: {rag_pipeline_id})")
            except Exception as e:
                logger.warning(f"Could not find RAG pipeline '{rag_pipeline_id}': {e}. Web search only.")
        else:
            logger.info("No RAG pipeline configured - web search only mode")
        self._set_state(item, state)
        return item

    def run_agent(self, item: dl.Item, context: dl.Context, progress: dl.Progress):
        """Pipeline node: NVIDIA AIQ Agent (orchestrator)

        Actions:
          - "research": send queries to Research node (RAG-first + web fallback)
          - "generate_report": send PromptItem to NIM Llama for final report
        """
        logger.info("=== AIQ Agent node ===")

        # Determine if returning from research or first call
        if self._is_temp_item(item):
            main_item = self._get_main_item(item)
            source = item.metadata.get('user', {}).get('source', 'unknown')
            logger.info(f"Received item from: {source}")
        else:
            main_item = item
            source = 'init'
            logger.info("First call - initializing research")

        state = self._get_state(main_item)

        # ── FIRST CALL: Generate queries and route to research ──
        if source == 'init' and not state.get('topic'):
            prompt_item = dl.PromptItem.from_item(main_item)
            prompts_json = prompt_item.to_json()['prompts']
            first_key = list(prompts_json.keys())[0]
            prompt_text = prompts_json[first_key][0]['value']

            params = self._extract_params_from_prompt(prompt_text)
            logger.info(f"Research topic: {params['topic']}")

            # Generate search queries using reasoning LLM
            query_prompt = QUERY_WRITER_INSTRUCTIONS.format(
                number_of_queries=params['num_queries'],
                topic=params['topic'],
                report_organization=params['report_organization']
            )
            query_response = self._invoke_llm(query_prompt)
            queries = self._parse_json_response(query_response)
            if not queries or not isinstance(queries, list):
                queries = [{"query": params['topic'], "report_section": "All", "rationale": "Main topic"}]

            logger.info(f"Generated {len(queries)} search queries")

            # Initialize state
            state = {
                'topic': params['topic'],
                'report_organization': params['report_organization'],
                'num_reflections': params['num_reflections'],
                'num_queries': params['num_queries'],
                'iteration': 0,
                'running_summary': '',
                'citations': '',
                'pending_queries': queries,
                'rag_pipeline_id': state.get('rag_pipeline_id', ''),
            }
            main_item = self._set_state(main_item, state)

            # Route to research node with queries
            temp_item = self._create_temp_item(
                main_item,
                content=json.dumps(queries),
                name="research_queries"
            )
            temp_item.metadata.setdefault('user', {})
            temp_item.metadata['user']['source'] = 'agent_to_research'
            temp_item.update(system_metadata=True)

            progress.update(action="research")
            return temp_item

        # ── RETURNING FROM RESEARCH: Summarize, reflect, decide ──
        elif source == 'research':
            # Read research results from data file
            results_file_id = item.metadata.get('user', {}).get('results_file_id', '')
            sources_xml = ''
            citations = ''
            if results_file_id:
                try:
                    results_json = json.loads(self._download_data_file(results_file_id))
                    sources_xml = results_json.get('sources_xml', '')
                    citations = results_json.get('citations', '')
                except Exception as e:
                    logger.error(f"Could not read research results file: {e}")

            logger.info(f"Research returned sources_xml length: {len(sources_xml)}")

            # Accumulate citations
            state['citations'] = (state.get('citations', '') + '\n' + citations).strip()

            # Summarize or extend report
            if not state.get('running_summary'):
                summary_prompt = SUMMARIZER_INSTRUCTIONS.format(
                    report_organization=state['report_organization'],
                    source=sources_xml
                )
                state['running_summary'] = self._invoke_llm(summary_prompt)
                logger.info("Initial summary generated")
            else:
                extend_prompt = REPORT_EXTENDER.format(
                    report_organization=state['report_organization'],
                    report=state['running_summary'],
                    source=sources_xml
                )
                state['running_summary'] = self._invoke_llm(extend_prompt)
                logger.info("Report extended with new sources")

            # Reflect
            iteration = state.get('iteration', 0)
            max_reflections = state.get('num_reflections', DEFAULT_NUM_REFLECTIONS)

            reflection_prompt = REFLECTION_INSTRUCTIONS.format(
                topic=state['topic'],
                report_organization=state['report_organization'],
                report=state['running_summary']
            )
            reflection_response = self._invoke_llm(reflection_prompt)
            reflection = self._parse_json_response(reflection_response)

            state['iteration'] = iteration + 1

            if state['iteration'] < max_reflections and reflection:
                # More research needed - generate follow-up query
                follow_up_query = reflection.get('query', state['topic']) if isinstance(reflection, dict) else state['topic']
                logger.info(f"Reflection {state['iteration']}/{max_reflections}: follow-up - {follow_up_query[:80]}")

                state['pending_queries'] = [{"query": follow_up_query}]
                main_item = self._set_state(main_item, state)

                # Route back to research
                temp_item = self._create_temp_item(main_item, content=follow_up_query, name="followup_research")
                temp_item.metadata.setdefault('user', {})
                temp_item.metadata['user']['source'] = 'agent_to_research'
                temp_item.update(system_metadata=True)

                progress.update(action="research")
                return temp_item
            else:
                # Research complete - finalize
                logger.info(f"Research complete after {state['iteration']} iterations. Preparing report.")
                return self._prepare_for_report(main_item, state, progress)

        else:
            logger.warning(f"Unknown source: {source}, routing to report")
            progress.update(action="generate_report")
            return main_item

    def _prepare_for_report(self, main_item: dl.Item, state: dict, progress: dl.Progress) -> dl.Item:
        """Internal: compile research and prepare PromptItem for NIM Llama.

        Strategy:
        - Append a brief report instruction to the original user message
          (preserving the original topic/organization text)
        - Upload the full research draft as a nearestItems document
        - The Llama node receives: user message (original + instruction) +
          context (research draft via nearestItems)
        """

        # Compile research document with everything Llama needs as context
        research_doc = f"""# Research Report Draft: {state['topic']}

## Report Organization
{state['report_organization']}

## Draft Report
{state['running_summary']}

## Sources & Citations
{state.get('citations', 'No sources collected.')}
"""

        # Upload research document
        safe_topic = re.sub(r'[^\w\s-]', '', state['topic'])[:40].strip().replace(' ', '_')
        filename = f"research_{safe_topic}.md"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(research_doc)
            local_path = f.name

        try:
            research_item = main_item.dataset.items.upload(
                local_path=local_path,
                remote_path="/.dataloop/aiq_research/",
                remote_name=filename,
                overwrite=True,
            )
            logger.info(f"Uploaded research document: {research_item.id}")
        finally:
            os.remove(local_path)


        prompt_item = dl.PromptItem.from_item(main_item)
        last_prompt = prompt_item.prompts[-1]

        if not hasattr(last_prompt, 'metadata') or last_prompt.metadata is None:
            last_prompt.metadata = {}
        last_prompt.metadata['nearestItems'] = [research_item.id]

        prompt_item.update()
        logger.info(f"Set nearestItems on PromptItem (original user message preserved)")

        main_item = self._set_state(main_item, state)

        progress.update(action="generate_report")
        return main_item

    # ─── Research Node ────────────────────────────────────────────────────────

    def research(self, item: dl.Item):
        """Pipeline node: Combined Research (RAG-first + web fallback)

        Implements NVIDIA's search strategy:
        For each query in parallel:
          1. RAG search (if configured)
          2. LLM-as-judge relevancy check
          3. Conditional web fallback (if RAG not relevant)
          4. Source deduplication and XML formatting
        """
        logger.info("=== AIQ Research node ===")

        main_item_id = item.metadata.get('user', {}).get('main_item')
        if main_item_id:
            main_item = dl.items.get(item_id=main_item_id)
        else:
            main_item = item

        state = self._get_state(main_item)
        queries = state.get('pending_queries', [])
        rag_pipeline_id = state.get('rag_pipeline_id', '')

        if not queries:
            logger.warning("No queries found in state")
            item.metadata.setdefault('user', {})
            item.metadata['user']['source'] = 'research'
            item.metadata['user']['results_file_id'] = ''
            item.update(system_metadata=True)
            return item

        logger.info(f"Processing {len(queries)} queries (RAG pipeline: {'configured' if rag_pipeline_id else 'none'})")

        # Process queries in parallel using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            futures = {
                executor.submit(
                    self._process_single_query, q, rag_pipeline_id, main_item.dataset
                ): q for q in queries
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    query = futures[future]
                    logger.error(f"Query processing failed: {e}")
                    results.append({
                        "query": str(query),
                        "rag_answer": "",
                        "rag_citation": "",
                        "relevancy": {"score": "no"},
                        "web_answer": f"Error: {e}",
                        "web_citation": "",
                    })

        # Deduplicate and format into XML (NVIDIA pattern)
        rag_answers = [r['rag_answer'] for r in results]
        relevancies = [r['relevancy'] for r in results]
        web_answers = [r['web_answer'] for r in results]
        sources_xml = self._deduplicate_and_format_sources(queries, rag_answers, relevancies, web_answers)

        # Aggregate citations
        all_citations = []
        for r in results:
            if r['rag_citation']:
                all_citations.append(r['rag_citation'])
            if r['web_citation']:
                all_citations.append(r['web_citation'])
        citations_str = "\n".join(all_citations)

        logger.info(f"Research complete: {len(results)} queries processed, XML length: {len(sources_xml)}")

        # Upload results as data file
        results_data = json.dumps({
            'sources_xml': sources_xml,
            'citations': citations_str,
        }, ensure_ascii=False)
        results_file = self._upload_data_file(
            dataset=item.dataset,
            data=results_data,
            remote_path=f"/.dataloop/aiq_temp_{main_item.id[:8]}/",
            filename=f"research_results_{item.id[:8]}.json",
        )
        logger.info(f"Uploaded research results file: {results_file.id}")

        # Store routing metadata
        item.metadata.setdefault('user', {})
        item.metadata['user']['source'] = 'research'
        item.metadata['user']['results_file_id'] = results_file.id
        item.update(system_metadata=True)

        return item
