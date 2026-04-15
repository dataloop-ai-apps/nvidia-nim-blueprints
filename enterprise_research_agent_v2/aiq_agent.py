"""
NVIDIA AIQ Enterprise Research Agent v2 - Dataloop Service Runner

Implements NVIDIA AI-Q Blueprint v2.0.0 features:
  - Two-tier research (shallow + deep)
  - Intent classification (meta / shallow / deep)
  - Human-in-the-Loop plan approval via AI Playground pipeline cycles
  - Citation verification and report sanitization
  - deepagents-based deep research with orchestrator/planner/researcher

Pipeline flow:
  Input -> [Init] -> [Intent Classifier]
                       |-- meta -> annotate response (end)
                       |-- shallow -> [Shallow Researcher]
                       |                |-- answer (end via Report Writer)
                       |                '-- escalate -> [Clarifier]
                       '-- deep -> [Clarifier]
                                     |-- plan_pending -> annotate plan for user (end, wait for approval)
                                     '-- plan_approved (via next cycle) -> [Deep Researcher] -> [Report Writer]
"""

import asyncio
import dtlpy as dl
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from enterprise_research_agent_v2.intent_classifier import IntentClassifier
from enterprise_research_agent_v2.shallow_researcher import ShallowResearcherAgent
from enterprise_research_agent_v2.clarifier import Clarifier
from enterprise_research_agent_v2.deep_researcher import DeepResearcherAgent
from enterprise_research_agent_v2.tools import build_tools

logger = logging.getLogger("[AIQ-v2-Enterprise-Research]")

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_INTENT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
DEFAULT_SHALLOW_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_CLARIFIER_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
DEFAULT_ORCHESTRATOR_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_PLANNER_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_RESEARCHER_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


class AIQEnterpriseAgentV2(dl.BaseServiceRunner):
    """Service runner for the NVIDIA AIQ Enterprise Research Agent v2 pipeline."""

    def __init__(self):
        nvidia_api_key = os.environ.get("NGC_API_KEY")
        if nvidia_api_key is None:
            raise ValueError("Missing NGC_API_KEY environment variable.")

        base_url = os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)

        # LLMs for different roles
        self.intent_llm = ChatNVIDIA(
            model=os.environ.get("INTENT_MODEL", DEFAULT_INTENT_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=2000,
        )

        self.shallow_llm = ChatNVIDIA(
            model=os.environ.get("SHALLOW_MODEL", DEFAULT_SHALLOW_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=16000,
        )

        self.clarifier_llm = ChatNVIDIA(
            model=os.environ.get("CLARIFIER_MODEL", DEFAULT_CLARIFIER_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=4000,
        )

        self.orchestrator_llm = ChatNVIDIA(
            model=os.environ.get("ORCHESTRATOR_MODEL", DEFAULT_ORCHESTRATOR_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.5,
            max_tokens=20000,
        )

        self.planner_llm = ChatNVIDIA(
            model=os.environ.get("PLANNER_MODEL", DEFAULT_PLANNER_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=8000,
        )

        self.researcher_llm = ChatNVIDIA(
            model=os.environ.get("RESEARCHER_MODEL", DEFAULT_RESEARCHER_MODEL),
            api_key=nvidia_api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=16000,
        )

        # Build tools
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        serper_api_key = os.environ.get("SERPER_API_KEY")
        self.tools = build_tools(
            tavily_api_key=tavily_api_key,
            serper_api_key=serper_api_key,
        )

        logger.info(
            "AIQ v2 initialized with %d tool(s): %s",
            len(self.tools),
            [t.name for t in self.tools],
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _item_folder(main_item_id: str) -> str:
        return f"/.dataloop/aiq_v2_{main_item_id[:8]}/"

    def _upload_data_file(self, dataset, data: str, remote_path: str, filename: str) -> dl.Item:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
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
        data_item = dl.items.get(item_id=item_id)
        buf = data_item.download(save_locally=False)
        if hasattr(buf, "read"):
            return buf.read().decode("utf-8", errors="replace")
        return str(buf)

    def _get_state(self, item: dl.Item) -> dict:
        state_file_id = item.metadata.get("user", {}).get("aiq_v2_state_file_id")
        if not state_file_id:
            return {}
        try:
            content = self._download_data_file(state_file_id)
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Could not load state file {state_file_id}: {e}")
            return {}

    def _set_state(self, item: dl.Item, state: dict) -> dl.Item:
        state_json = json.dumps(state, ensure_ascii=False)
        state_item = self._upload_data_file(
            dataset=item.dataset,
            data=state_json,
            remote_path=self._item_folder(item.id),
            filename=f"state_v2_{item.id[:12]}.json",
        )
        item.metadata.setdefault("user", {})
        item.metadata["user"]["aiq_v2_state_file_id"] = state_item.id
        item = item.update(system_metadata=True)
        return item

    def _get_main_item(self, item: dl.Item) -> dl.Item:
        main_item_id = item.metadata.get("user", {}).get("main_item")
        if main_item_id:
            return dl.items.get(item_id=main_item_id)
        return item

    def _is_temp_item(self, item: dl.Item) -> bool:
        return "main_item" in item.metadata.get("user", {})

    def _create_temp_item(self, main_item: dl.Item, content: str, name: str) -> dl.Item:
        safe_name = re.sub(r"[^\w\s-]", "", name)[:50].strip().replace(" ", "_")
        filename = f"{safe_name}.txt"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            local_path = f.name

        try:
            temp_item = main_item.dataset.items.upload(
                local_path=local_path,
                remote_path=self._item_folder(main_item.id),
                remote_name=filename,
                overwrite=True,
                item_metadata={"user": {"main_item": main_item.id}},
            )
            return temp_item
        finally:
            os.remove(local_path)

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        cleaned = text
        while "<think>" in cleaned and "</think>" in cleaned:
            start = cleaned.find("<think>")
            end = cleaned.find("</think>") + len("</think>")
            cleaned = cleaned[:start] + cleaned[end:]
        if "<think>" in cleaned:
            cleaned = cleaned[: cleaned.find("<think>")]
        return cleaned.strip()

    @staticmethod
    def _get_user_query(item: dl.Item) -> str:
        """Extract the user's text query from a PromptItem."""
        try:
            prompt_item = dl.PromptItem.from_item(item)
            prompts_json = prompt_item.to_json()["prompts"]
            first_key = list(prompts_json.keys())[0]
            return prompts_json[first_key][0]["value"]
        except Exception as e:
            logger.error(f"Could not extract user query: {e}")
            return ""

    @staticmethod
    def _annotate_response(item: dl.Item, response_text: str) -> dl.Item:
        """Write the response as an annotation on the PromptItem (visible in AI Playground)."""
        try:
            prompt_item = dl.PromptItem.from_item(item)
            prompt_item.add(
                message={
                    "role": "assistant",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": response_text}],
                }
            )
            prompt_item.update()
            logger.info("Response annotation written (%d chars)", len(response_text))
        except Exception as e:
            logger.error(f"Failed to annotate response: {e}")
        return item

    def _run_async(self, coro):
        """Run an async coroutine in a synchronous context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ─── Pipeline Node 1: Init ────────────────────────────────────────────────

    def init_research(self, item: dl.Item, rag_pipeline_id: str = None, context: dl.Context = None):
        """Pipeline node: Init (ROOT)

        Detects if this is a fresh query or a plan-approval cycle.
        Routes:
          - Fresh query -> action: classify
          - Approved plan -> action: deep_research_approved
        """
        logger.info("=== AIQ v2 Init node ===")

        state = self._get_state(item)

        # Check for plan approval: if metadata has plan_pending=True and
        # the latest user message is an approval
        user_meta = item.metadata.get("user", {})
        plan_pending = user_meta.get("plan_pending", False) or state.get("plan_pending", False)

        if plan_pending:
            user_query = self._get_user_query(item)
            lower_query = user_query.strip().lower()

            approval_signals = [
                "approved", "approve", "yes", "go ahead",
                "start", "proceed", "ok", "looks good", "lgtm",
            ]
            is_approved = any(signal in lower_query for signal in approval_signals)

            if is_approved:
                logger.info("Plan APPROVED by user. Routing to deep research.")
                state["plan_approved"] = True
                state["plan_pending"] = False
                state["user_feedback"] = user_query
                item = self._set_state(item, state)

                context.progress.update(action="deep_research_approved")
                return item
            else:
                logger.info("User provided feedback on plan. Re-routing to clarifier.")
                state["plan_feedback"] = user_query
                state["plan_pending"] = False
                feedback_history = state.get("feedback_history", [])
                feedback_history.append(user_query)
                state["feedback_history"] = feedback_history
                item = self._set_state(item, state)

                context.progress.update(action="classify")
                return item

        # Validate RAG pipeline if provided
        if rag_pipeline_id:
            try:
                rag_pipeline = dl.pipelines.get(pipeline_id=rag_pipeline_id)
                if rag_pipeline.status != "Installed":
                    logger.warning("RAG pipeline not active - skipping")
                else:
                    state["rag_pipeline_id"] = rag_pipeline_id
                    logger.info(f"RAG pipeline validated: {rag_pipeline.name}")
            except Exception as e:
                logger.warning(f"Could not find RAG pipeline: {e}")
        else:
            logger.info("No RAG pipeline configured")

        item = self._set_state(item, state)
        context.progress.update(action="classify")
        return item

    # ─── Pipeline Node 2: Intent Classifier ──────────────────────────────────

    def classify_intent(self, item: dl.Item, context: dl.Context = None, progress: dl.Progress = None):
        """Pipeline node: Intent Classification

        Routes:
          - meta -> annotate response directly (end)
          - shallow -> action: shallow
          - deep -> action: deep
        """
        logger.info("=== AIQ v2 Intent Classifier node ===")

        if self._is_temp_item(item):
            main_item = self._get_main_item(item)
        else:
            main_item = item

        state = self._get_state(main_item)
        user_query = self._get_user_query(main_item)
        logger.info("Classifying query: '%s'", user_query[:100])

        # Check if re-entering after plan feedback -> route to deep/clarifier
        if state.get("plan_feedback"):
            logger.info("Re-entering after plan feedback. Routing to clarifier (deep).")
            _progress = context.progress if context else progress
            _progress.update(action="deep")
            return main_item

        tools_info = [
            {"name": t.name, "description": t.description} for t in self.tools
        ]

        classifier = IntentClassifier(llm=self.intent_llm, tools_info=tools_info)
        result = classifier.classify(user_query)

        state["intent"] = result["intent"]
        state["research_depth"] = result.get("research_depth")
        state["query"] = user_query
        main_item = self._set_state(main_item, state)

        _progress = context.progress if context else progress

        if result["intent"] == "meta":
            meta_response = result.get("meta_response", "")
            if meta_response:
                self._annotate_response(main_item, meta_response)
            else:
                self._annotate_response(main_item, "Hello! I'm the AI Research Assistant. How can I help you today?")
            logger.info("Meta intent -> response annotated")
            _progress.update(action="meta")
            return main_item

        depth = result.get("research_depth", "shallow")
        if depth == "deep":
            logger.info("Deep research intent detected")
            _progress.update(action="deep")
        else:
            logger.info("Shallow research intent detected")
            _progress.update(action="shallow")

        return main_item

    # ─── Pipeline Node 3: Shallow Researcher ─────────────────────────────────

    def shallow_research(self, item: dl.Item, context: dl.Context = None, progress: dl.Progress = None):
        """Pipeline node: Shallow Research

        Routes:
          - answer ready -> action: generate_report
          - escalate -> action: escalate
        """
        logger.info("=== AIQ v2 Shallow Researcher node ===")

        if self._is_temp_item(item):
            main_item = self._get_main_item(item)
        else:
            main_item = item

        state = self._get_state(main_item)
        user_query = state.get("query", self._get_user_query(main_item))

        # Build tools with RAG if configured
        tools = list(self.tools)
        rag_pipeline_id = state.get("rag_pipeline_id")
        if rag_pipeline_id:
            from enterprise_research_agent_v2.tools import DataloopRAGTool
            rag_tool = DataloopRAGTool(
                rag_pipeline_id=rag_pipeline_id,
                dataset_id=main_item.dataset.id,
            )
            tools.append(rag_tool)

        agent = ShallowResearcherAgent(
            llm=self.shallow_llm,
            tools=tools,
            max_tool_iterations=5,
            max_llm_turns=10,
        )

        result = self._run_async(agent.run(user_query))

        answer = result.get("answer", "")
        should_escalate = result.get("should_escalate", False)
        sources = result.get("sources", [])

        state["shallow_answer"] = answer
        state["shallow_sources"] = sources

        _progress = context.progress if context else progress

        if should_escalate:
            logger.info("Shallow research recommends escalation to deep research")
            state["escalated"] = True
            main_item = self._set_state(main_item, state)
            _progress.update(action="escalate")
            return main_item

        # Prepare for report writer
        state["research_complete"] = True
        main_item = self._set_state(main_item, state)

        self._prepare_for_report(main_item, answer, state)
        _progress.update(action="generate_report")
        return main_item

    # ─── Pipeline Node 4: Clarifier ──────────────────────────────────────────

    def clarify_and_plan(self, item: dl.Item, context: dl.Context = None, progress: dl.Progress = None):
        """Pipeline node: Clarification + Plan Generation

        Routes:
          - plan_pending -> annotate plan in chat (end, wait for user approval in next cycle)
          - plan_approved -> action: plan_approved (proceed to deep research)
        """
        logger.info("=== AIQ v2 Clarifier node ===")

        if self._is_temp_item(item):
            main_item = self._get_main_item(item)
        else:
            main_item = item

        state = self._get_state(main_item)
        user_query = state.get("query", self._get_user_query(main_item))

        clarifier = Clarifier(llm=self.clarifier_llm)

        # Build clarifier context from any prior feedback
        clarifier_context = None
        feedback_history = state.get("feedback_history", [])
        if feedback_history:
            clarifier_context = "Previous user feedback:\n" + "\n".join(
                f"- {fb}" for fb in feedback_history
            )

        plan_feedback = state.get("plan_feedback")
        if plan_feedback:
            # User gave feedback on the plan -> regenerate plan with feedback
            logger.info("Regenerating plan with user feedback")
            plan = clarifier.generate_plan(
                query=user_query,
                clarifier_context=clarifier_context,
                feedback_history=feedback_history,
            )
            state.pop("plan_feedback", None)
        else:
            result = clarifier.run(user_query, clarifier_context=clarifier_context)

            if result["needs_clarification"] and result["clarification_question"]:
                self._annotate_response(main_item, result["clarification_question"])
                state["clarification_asked"] = True
                main_item = self._set_state(main_item, state)

                _progress = context.progress if context else progress
                _progress.update(action="plan_pending")
                return main_item

            plan = result.get("plan", {})

        # Plan generated -> present to user for approval
        plan_text = clarifier._format_plan_for_chat(plan, user_query)
        self._annotate_response(main_item, plan_text)

        state["plan"] = plan
        state["plan_pending"] = True
        state["plan_text"] = plan_text

        # Store plan_pending flag in item metadata for detection in next cycle
        main_item.metadata.setdefault("user", {})
        main_item.metadata["user"]["plan_pending"] = True
        main_item = main_item.update(system_metadata=True)
        main_item = self._set_state(main_item, state)

        logger.info("Plan presented to user, waiting for approval")

        _progress = context.progress if context else progress
        _progress.update(action="plan_pending")
        return main_item

    # ─── Pipeline Node 5: Deep Researcher ────────────────────────────────────

    def deep_research(self, item: dl.Item, context: dl.Context = None, progress: dl.Progress = None):
        """Pipeline node: Deep Research (single node using deepagents)

        Runs the full orchestrator/planner/researcher pipeline internally.
        Routes:
          - report ready -> action: generate_report
        """
        logger.info("=== AIQ v2 Deep Researcher node ===")

        if self._is_temp_item(item):
            main_item = self._get_main_item(item)
        else:
            main_item = item

        state = self._get_state(main_item)
        user_query = state.get("query", self._get_user_query(main_item))

        # Build tools with RAG if configured
        tools = list(self.tools)
        rag_pipeline_id = state.get("rag_pipeline_id")
        if rag_pipeline_id:
            from enterprise_research_agent_v2.tools import DataloopRAGTool
            rag_tool = DataloopRAGTool(
                rag_pipeline_id=rag_pipeline_id,
                dataset_id=main_item.dataset.id,
            )
            tools.append(rag_tool)

        plan = state.get("plan")
        clarifier_context = None
        if state.get("feedback_history"):
            clarifier_context = "\n".join(state["feedback_history"])

        agent = DeepResearcherAgent(
            orchestrator_llm=self.orchestrator_llm,
            planner_llm=self.planner_llm,
            researcher_llm=self.researcher_llm,
            tools=tools,
            max_loops=30,
            plan=plan,
            clarifier_context=clarifier_context,
        )

        result = self._run_async(agent.run(user_query))

        report = result.get("report", "")
        sources = result.get("sources", [])
        validation = result.get("validation", {})

        state["deep_report"] = report
        state["deep_sources"] = sources
        state["validation"] = validation
        state["research_complete"] = True
        main_item = self._set_state(main_item, state)

        # Prepare for report writer node
        self._prepare_for_report(main_item, report, state)

        _progress = context.progress if context else progress
        _progress.update(action="generate_report")

        logger.info(
            "Deep research complete. Report: %d words, complete: %s",
            len(report.split()),
            validation.get("is_complete", False),
        )
        return main_item

    # ─── Report Preparation ──────────────────────────────────────────────────

    def _prepare_for_report(self, main_item: dl.Item, report_content: str, state: dict):
        """Prepare the PromptItem for the NIM report writer node.

        Uploads the research report as a nearestItems document on the prompt,
        so the NIM predict node can use it as context for final formatting.
        """
        research_doc = f"""## Research Report

{report_content}"""

        safe_query = re.sub(r"[^\w\s-]", "", state.get("query", "research")[:40]).strip().replace(" ", "_")
        filename = f"research_v2_{safe_query}.md"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(research_doc)
            local_path = f.name

        try:
            research_item = main_item.dataset.items.upload(
                local_path=local_path,
                remote_path=self._item_folder(main_item.id),
                remote_name=filename,
                overwrite=True,
            )
            logger.info("Uploaded research document: %s", research_item.id)
        finally:
            os.remove(local_path)

        # Attach as nearestItems on the latest prompt
        prompt_item = dl.PromptItem.from_item(main_item)
        last_prompt = prompt_item.prompts[-1]

        if not hasattr(last_prompt, "metadata") or last_prompt.metadata is None:
            last_prompt.metadata = {}
        last_prompt.metadata["nearestItems"] = [research_item.id]

        prompt_item.update()
        logger.info("Set nearestItems on prompt for report writer")

        # Verify propagation
        for attempt in range(1, 6):
            time.sleep(1)
            try:
                check_pi = dl.PromptItem.from_item(dl.items.get(item_id=main_item.id))
                if check_pi.prompts[-1].metadata.get("nearestItems"):
                    logger.info("nearestItems confirmed (attempt %d)", attempt)
                    break
            except Exception:
                pass
        else:
            logger.error("nearestItems not confirmed after retries")
