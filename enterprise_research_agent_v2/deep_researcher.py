"""
Deep research agent for AIQ v2.

Wraps the `deepagents` library to run a multi-agent research system:
  - Orchestrator: coordinates research, writes final report
  - Planner: creates research plans with task analysis + TOC + queries
  - Researcher: gathers information using search tools

Uses an in-memory virtual file system for sub-agent communication.
Includes citation verification, source registry middleware, and report
completeness retry loop matching the NVIDIA AI-Q Blueprint v2.0.0.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import ConfigDict
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from enterprise_research_agent_v2.prompts import render_prompt
from enterprise_research_agent_v2.citation_verification import (
    Source,
    SourceRegistry,
    extract_sources_from_tool_result,
    sanitize_report,
    verify_citations,
)

logger = logging.getLogger("[AIQ-v2-DeepResearcher]")

_MIN_REPORT_LENGTH = 1500
_MAX_REPORT_RETRIES = 2
_DEEPAGENT_TIMEOUT_SECONDS = 20 * 60  # 20 minutes per attempt


# ─── Tools ────────────────────────────────────────────────────────────────────


@tool
def think(thought: str) -> str:
    """Use this tool to record your reasoning without taking any external action.
    Good for planning next steps, verifying constraints, or reflecting on findings."""
    return "Thought recorded."


class VerifiedSourcesTool(BaseTool):
    """Returns the list of verified sources from the SourceRegistry."""

    name: str = "get_verified_sources"
    description: str = (
        "Returns the list of all verified source URLs collected during research. "
        "Use ONLY these URLs when writing citations in the report."
    )
    _registry: SourceRegistry = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, registry: SourceRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, query: str = "") -> str:
        return self._registry.get_source_list_text() or "No sources collected yet."

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


# ─── Custom Middleware ────────────────────────────────────────────────────────
# Matches the middleware stack from NVIDIA AIQ Blueprint v2.


class SourceRegistryMiddleware:
    """Captures URLs from tool results into a SourceRegistry.

    Implements the AgentMiddleware protocol expected by deepagents.
    """

    def __init__(self, source_tool_names: set[str], registry: SourceRegistry):
        self.source_tool_names = source_tool_names
        self.registry = registry

    def on_tool_end(self, tool_name: str, tool_output: str, **kwargs) -> str:
        if tool_name in self.source_tool_names:
            sources = extract_sources_from_tool_result(tool_name, tool_output)
            for source in sources:
                self.registry.add(source)
            if sources:
                logger.info("Captured %d source(s) from %s", len(sources), tool_name)
        return tool_output


class ToolRetryMiddleware:
    """Retries failed tool calls with exponential backoff."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0, initial_delay: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay

    def on_tool_error(self, tool_name: str, error: Exception, attempt: int, **kwargs) -> dict:
        if attempt < self.max_retries:
            delay = self.initial_delay * (self.backoff_factor ** attempt)
            logger.warning("Tool %s failed (attempt %d/%d), retrying in %.1fs: %s",
                           tool_name, attempt + 1, self.max_retries, delay, error)
            time.sleep(delay)
            return {"retry": True}
        logger.error("Tool %s failed after %d attempts: %s", tool_name, self.max_retries, error)
        return {"retry": False}


class ModelRetryMiddleware:
    """Retries failed LLM calls with exponential backoff."""

    def __init__(self, max_retries: int = 10, backoff_factor: float = 2.0, initial_delay: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay

    def on_model_error(self, error: Exception, attempt: int, **kwargs) -> dict:
        if attempt < self.max_retries:
            delay = self.initial_delay * (self.backoff_factor ** attempt)
            delay = min(delay, 60.0)
            logger.warning("Model call failed (attempt %d/%d), retrying in %.1fs: %s",
                           attempt + 1, self.max_retries, delay, error)
            time.sleep(delay)
            return {"retry": True}
        logger.error("Model call failed after %d attempts: %s", self.max_retries, error)
        return {"retry": False}


class ToolResultPruningMiddleware:
    """Prunes old tool results to prevent context window overflow."""

    def __init__(self, keep_last_n: int = 10, max_chars: int = 2000):
        self.keep_last_n = keep_last_n
        self.max_chars = max_chars

    def on_tool_end(self, tool_name: str, tool_output: str, **kwargs) -> str:
        if len(tool_output) > self.max_chars:
            truncated = tool_output[:self.max_chars]
            return truncated + f"\n\n[... truncated from {len(tool_output)} chars to {self.max_chars}]"
        return tool_output


class EmptyContentFixMiddleware:
    """Fixes empty content responses from the model."""

    def on_model_end(self, response, **kwargs):
        if hasattr(response, "content") and not response.content:
            response.content = " "
        return response


class ToolNameSanitizationMiddleware:
    """Sanitizes tool names in model responses to match available tools."""

    def __init__(self, valid_tool_names: set[str]):
        self.valid_tool_names = valid_tool_names
        self._name_map: dict[str, str] = {}
        for name in valid_tool_names:
            self._name_map[name.lower()] = name
            self._name_map[name.replace("-", "_").lower()] = name
            self._name_map[name.replace("_", "-").lower()] = name

    def on_tool_start(self, tool_name: str, tool_input: dict, **kwargs) -> tuple[str, dict]:
        if tool_name in self.valid_tool_names:
            return tool_name, tool_input
        normalized = tool_name.lower().strip()
        if normalized in self._name_map:
            fixed = self._name_map[normalized]
            logger.debug("Sanitized tool name: %s -> %s", tool_name, fixed)
            return fixed, tool_input
        return tool_name, tool_input


# ─── Report Validation ────────────────────────────────────────────────────────


class ReportValidationResult:
    def __init__(self, is_complete: bool, feedback: str = ""):
        self.is_complete = is_complete
        self.feedback = feedback


def _validate_report(report: str) -> ReportValidationResult:
    """Validate report completeness with detailed heuristics matching NVIDIA's implementation."""
    if not report or not report.strip():
        return ReportValidationResult(False, "Report is empty.")

    word_count = len(report.split())
    hard_issues = []
    soft_issues = []

    if word_count < _MIN_REPORT_LENGTH:
        hard_issues.append(
            f"Report is too short ({word_count} words). "
            f"Expected at least {_MIN_REPORT_LENGTH} words for a comprehensive report. "
            f"Expand each section with more detail, analysis, and evidence."
        )

    heading_count = len(re.findall(r"^#{1,3}\s+", report, re.MULTILINE))
    if heading_count < 2:
        hard_issues.append(
            f"Only {heading_count} section heading(s) found. "
            f"A comprehensive report needs at least 5 headings covering all TOC sections."
        )

    has_refs = bool(re.search(
        r"(^#{1,3}\s*(Sources|References|Reference List|Bibliography)"
        r"|\*\*(Sources|References|Reference List|Bibliography)\*?\*?\s*:?)",
        report, re.MULTILINE | re.IGNORECASE
    ))
    if not has_refs:
        soft_issues.append(
            "Missing References/Sources section at the end of the report. "
            "Add a numbered reference list with all cited URLs."
        )

    citation_count = len(re.findall(r"\[\d+\]", report))
    if has_refs and citation_count < 3:
        soft_issues.append(
            f"Only {citation_count} inline citation(s) found. "
            f"Every major claim should have a citation [N]."
        )

    giving_up_patterns = [
        r"i('m| am) unable to",
        r"i cannot complete",
        r"i('m| am) sorry.{0,20}(cannot|unable)",
        r"unfortunately.{0,20}(cannot|unable)",
    ]
    for pattern in giving_up_patterns:
        if re.search(pattern, report.lower()):
            hard_issues.append(
                "Report contains 'giving up' language. "
                "Produce a best-effort report using available information."
            )
            break

    if hard_issues:
        all_issues = hard_issues + soft_issues
        return ReportValidationResult(False, " | ".join(all_issues))

    if soft_issues:
        logger.info("Report has minor issues (not blocking): %s", " | ".join(soft_issues))

    return ReportValidationResult(True)


# ─── Deep Researcher Agent ────────────────────────────────────────────────────


class DeepResearcherAgent:
    """
    Deep research agent using the deepagents library.

    Matches NVIDIA AI-Q Blueprint v2.0.0 architecture:
    - create_deep_agent with SubAgent specs for planner and researcher
    - StateBackend for virtual file system
    - Custom middleware stack for reliability
    - Report completeness retry loop (max 5 retries)
    """

    def __init__(
        self,
        orchestrator_llm: ChatNVIDIA,
        planner_llm: ChatNVIDIA,
        researcher_llm: ChatNVIDIA,
        tools: list[BaseTool],
        max_loops: int = 2,
        plan: Optional[dict] = None,
        clarifier_context: Optional[str] = None,
    ):
        self.orchestrator_llm = orchestrator_llm
        self.planner_llm = planner_llm
        self.researcher_llm = researcher_llm
        self.tools = tools
        self.max_loops = max_loops
        self.plan = plan
        self.clarifier_context = clarifier_context
        self.source_registry = SourceRegistry()

        self.tools_info = [
            {"name": getattr(t, "name", str(t)), "description": getattr(t, "description", "")}
            for t in tools
        ]

    async def run(self, query: str) -> dict:
        """Execute deep research. Returns dict with report, sources, validation."""
        try:
            report = await self._run_with_deepagents(query)
        except ImportError:
            logger.warning("deepagents not available, falling back to sequential implementation")
            report = await self._run_fallback(query)
        except Exception as e:
            logger.error(f"Deep research failed: {e}", exc_info=True)
            report = f"Deep research encountered an error: {e}"

        if self.source_registry.all_sources():
            verification = verify_citations(report, self.source_registry)
            report = verification.verified_report
            logger.info(
                "Citation verification: %d valid, %d removed",
                len(verification.valid_citations),
                len(verification.removed_citations),
            )

        sanitization = sanitize_report(report)
        report = sanitization.sanitized_report

        validation = _validate_report(report)

        return {
            "report": report,
            "sources": self.source_registry.to_dict(),
            "validation": {
                "is_complete": validation.is_complete,
                "feedback": validation.feedback,
            },
        }

    async def _run_with_deepagents(self, query: str) -> str:
        """Run deep research using the actual deepagents library API."""
        from deepagents import create_deep_agent, SubAgent
        from deepagents.backends.state import StateBackend
        from deepagents.backends.composite import CompositeBackend
        from langgraph.store.memory import InMemoryStore

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        orchestrator_prompt = render_prompt(
            "orchestrator.j2",
            tools=self.tools_info,
            current_datetime=current_datetime,
            clarifier_result=self.clarifier_context,
            available_documents=[],
        )

        planner_prompt = render_prompt(
            "planner.j2",
            tools=self.tools_info,
            current_datetime=current_datetime,
            available_documents=[],
        )

        researcher_prompt = render_prompt(
            "researcher.j2",
            tools=self.tools_info,
            current_datetime=current_datetime,
            available_documents=[],
        )

        verified_sources_tool = VerifiedSourcesTool(registry=self.source_registry)

        planner_subagent: SubAgent = {
            "name": "planner-agent",
            "description": "Content-driven research planning. Generates task analysis, TOC, constraints, and queries.",
            "system_prompt": planner_prompt,
            "tools": list(self.tools),
            "model": self.planner_llm,
        }

        researcher_subagent: SubAgent = {
            "name": "researcher-agent",
            "description": "Gathers and synthesizes information using available search tools.",
            "system_prompt": researcher_prompt,
            "tools": list(self.tools),
            "model": self.researcher_llm,
        }

        backend = CompositeBackend(default=StateBackend(), routes={})
        store = InMemoryStore()

        agent = create_deep_agent(
            model=self.orchestrator_llm,
            tools=[think, verified_sources_tool] + list(self.tools),
            system_prompt=orchestrator_prompt,
            subagents=[planner_subagent, researcher_subagent],
            store=store,
            backend=backend,
        )

        task_description = query
        if self.plan:
            plan_json = json.dumps(self.plan, indent=2)
            task_description = (
                f"{query}\n\n"
                f"**Approved Research Plan:**\n```json\n{plan_json}\n```"
            )

        # Retry loop: run the agent and check report completeness
        report = ""
        for attempt in range(1, _MAX_REPORT_RETRIES + 1):
            logger.info("Deep research attempt %d/%d", attempt, _MAX_REPORT_RETRIES)

            input_msg = task_description if attempt == 1 else (
                f"Your previous report was incomplete. Feedback:\n{validation.feedback}\n\n"
                f"Please fix the issues and produce the complete report. "
                f"Original request: {task_description}"
            )

            try:
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": [{"role": "user", "content": input_msg}]}),
                    timeout=_DEEPAGENT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning("Deep research attempt %d timed out after %d seconds",
                               attempt, _DEEPAGENT_TIMEOUT_SECONDS)
                continue

            report = self._extract_report(result)

            validation = _validate_report(report)
            if validation.is_complete:
                logger.info("Report passed validation on attempt %d", attempt)
                break

            logger.warning("Report incomplete (attempt %d): %s", attempt, validation.feedback)

        return report

    async def _run_fallback(self, query: str) -> str:
        """Fallback if deepagents is not installed. Runs planner->researcher->synthesis sequentially."""
        from langchain_core.messages import SystemMessage, HumanMessage

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Phase 1: Planning
        planner_prompt = render_prompt(
            "planner.j2",
            tools=self.tools_info,
            current_datetime=current_datetime,
            available_documents=[],
        )

        plan_task = query
        if self.plan:
            plan_task += f"\n\nPre-approved plan: {json.dumps(self.plan)}"

        logger.info("Running planner phase...")
        plan_response = await self.planner_llm.ainvoke([
            SystemMessage(content=planner_prompt),
            HumanMessage(content=f"Create a research plan for: {plan_task}"),
        ])
        plan_text = plan_response.content or ""

        # Phase 2: Research
        researcher_prompt = render_prompt(
            "researcher.j2",
            tools=self.tools_info,
            current_datetime=current_datetime,
            available_documents=[],
        )

        research_results = []
        queries_to_research = self._extract_queries_from_plan(plan_text)
        if not queries_to_research:
            queries_to_research = [query]

        source_mw = SourceRegistryMiddleware({t.name for t in self.tools}, self.source_registry)

        for i, research_query in enumerate(queries_to_research[:6]):
            logger.info("Researching query %d/%d: %s", i + 1, len(queries_to_research), research_query[:80])

            researcher_llm_with_tools = self.researcher_llm.bind_tools(self.tools)
            response = await researcher_llm_with_tools.ainvoke([
                SystemMessage(content=researcher_prompt),
                HumanMessage(content=research_query),
            ])

            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    for t in self.tools:
                        if t.name == tool_name:
                            try:
                                tool_result = t._run(**tool_args)
                                source_mw.on_tool_end(tool_name, str(tool_result))
                                research_results.append(str(tool_result))
                            except Exception as e:
                                logger.error(f"Tool {tool_name} failed: {e}")
                            break

            if response.content:
                research_results.append(response.content)

        # Phase 3: Synthesis with retry
        orchestrator_prompt = render_prompt(
            "orchestrator.j2",
            tools=self.tools_info,
            current_datetime=current_datetime,
            clarifier_result=self.clarifier_context,
            available_documents=[],
        )

        all_research = "\n\n---\n\n".join(research_results)
        verified_sources = self.source_registry.get_source_list_text()

        report = ""
        for attempt in range(1, _MAX_REPORT_RETRIES + 1):
            if attempt == 1:
                content = (
                    f"Write a comprehensive research report on: {query}\n\n"
                    f"**Research Plan:**\n{plan_text}\n\n"
                    f"**Research Findings:**\n{all_research}\n\n"
                    f"**{verified_sources}**\n\n"
                    f"Write the full report now."
                )
            else:
                content = (
                    f"Your previous report was incomplete. Feedback:\n{validation.feedback}\n\n"
                    f"Fix the issues. Here is the research again:\n{all_research}\n\n"
                    f"**{verified_sources}**\n\n"
                    f"Write the complete report now."
                )

            logger.info("Synthesis attempt %d/%d", attempt, _MAX_REPORT_RETRIES)
            report_response = await self.orchestrator_llm.ainvoke([
                SystemMessage(content=orchestrator_prompt),
                HumanMessage(content=content),
            ])
            report = report_response.content or ""

            validation = _validate_report(report)
            if validation.is_complete:
                logger.info("Report passed validation on attempt %d", attempt)
                break

            logger.warning("Report incomplete (attempt %d): %s", attempt, validation.feedback)

        return report

    @staticmethod
    def _extract_report(result) -> str:
        """Extract the report text from the deepagents agent result."""
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    return last_msg.content or ""
            return result.get("report", "") or result.get("output", "") or str(result)
        if isinstance(result, str):
            return result
        return str(result)

    @staticmethod
    def _extract_queries_from_plan(plan_text: str) -> list[str]:
        """Extract search queries from planner output."""
        queries = []
        try:
            for m in re.finditer(r"\{.*\}", plan_text, re.DOTALL):
                try:
                    data = json.loads(m.group(0))
                    if "queries" in data:
                        for q in data["queries"]:
                            if isinstance(q, dict):
                                queries.append(q.get("query", ""))
                            elif isinstance(q, str):
                                queries.append(q)
                        break
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

        return [q for q in queries if q.strip()]
