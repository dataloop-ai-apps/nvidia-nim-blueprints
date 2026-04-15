"""
Deep research agent for AIQ v2.

Wraps the `deepagents` library to run a multi-agent research system:
  - Orchestrator: coordinates research, writes final report
  - Planner: creates research plans with task analysis + TOC + queries
  - Researcher: gathers information using search tools

Uses an in-memory virtual file system for sub-agent communication.
Includes citation verification and source registry middleware.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

from langchain_core.tools import BaseTool
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


class SourceRegistryMiddleware:
    """Middleware that captures URLs from tool results into a SourceRegistry."""

    def __init__(self, registry: SourceRegistry):
        self.registry = registry

    def on_tool_result(self, tool_name: str, result: str) -> None:
        """Called after each tool execution to capture source URLs."""
        sources = extract_sources_from_tool_result(tool_name, result)
        for source in sources:
            self.registry.add(source)
        if sources:
            logger.info("Captured %d source(s) from %s", len(sources), tool_name)


class VerifiedSourcesTool(BaseTool):
    """Tool that returns the list of verified sources from the SourceRegistry."""

    name: str = "get_verified_sources"
    description: str = (
        "Returns the list of all verified source URLs collected during research. "
        "Use ONLY these URLs when writing citations in the report."
    )
    _registry: SourceRegistry = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, registry: SourceRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, query: str = "") -> str:
        return self._registry.get_source_list_text() or "No sources collected yet."

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class ReportValidationResult:
    """Result of report completeness validation."""

    def __init__(self, is_complete: bool, feedback: str = ""):
        self.is_complete = is_complete
        self.feedback = feedback


class DeepResearcherAgent:
    """
    Deep research agent using deepagents library.

    Runs orchestrator, planner, and researcher sub-agents in a single
    process with a virtual file system for coordination.
    """

    def __init__(
        self,
        orchestrator_llm: ChatNVIDIA,
        planner_llm: ChatNVIDIA,
        researcher_llm: ChatNVIDIA,
        tools: list[BaseTool],
        max_loops: int = 30,
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
        self._middleware = SourceRegistryMiddleware(self.source_registry)

        self.tools_info = [
            {"name": getattr(t, "name", str(t)), "description": getattr(t, "description", "")}
            for t in tools
        ]

    async def run(self, query: str) -> dict:
        """
        Execute deep research using deepagents.

        Returns dict with:
          - report: str (the final report)
          - sources: list[dict] (source registry as dicts)
          - validation: dict (report validation result)
        """
        try:
            report = await self._run_with_deepagents(query)
        except ImportError:
            logger.warning("deepagents not available, falling back to sequential implementation")
            report = await self._run_fallback(query)
        except Exception as e:
            logger.error(f"Deep research failed: {e}", exc_info=True)
            report = f"Deep research encountered an error: {e}"

        # Citation verification
        if self.source_registry.all_sources():
            verification = verify_citations(report, self.source_registry)
            report = verification.verified_report
            logger.info(
                "Citation verification: %d valid, %d removed",
                len(verification.valid_citations),
                len(verification.removed_citations),
            )

        # Sanitize
        sanitization = sanitize_report(report)
        report = sanitization.sanitized_report

        # Validate completeness
        validation = self._validate_report(report)

        return {
            "report": report,
            "sources": self.source_registry.to_dict(),
            "validation": {
                "is_complete": validation.is_complete,
                "feedback": validation.feedback,
            },
        }

    async def _run_with_deepagents(self, query: str) -> str:
        """Run deep research using the deepagents library."""
        from deepagents import create_deep_agent

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

        orchestrator_tools = [verified_sources_tool]
        sub_agent_tools = list(self.tools)

        def _on_tool_result(tool_name: str, result: str):
            self._middleware.on_tool_result(tool_name, result)

        agent = create_deep_agent(
            orchestrator_llm=self.orchestrator_llm,
            orchestrator_system_prompt=orchestrator_prompt,
            orchestrator_tools=orchestrator_tools,
            planner_llm=self.planner_llm,
            planner_system_prompt=planner_prompt,
            planner_tools=sub_agent_tools,
            researcher_llm=self.researcher_llm,
            researcher_system_prompt=researcher_prompt,
            researcher_tools=sub_agent_tools,
            max_loops=self.max_loops,
            on_tool_result=_on_tool_result,
        )

        task_description = query
        if self.plan:
            plan_json = json.dumps(self.plan, indent=2)
            task_description = (
                f"{query}\n\n"
                f"**Approved Research Plan:**\n```json\n{plan_json}\n```"
            )

        result = await agent.run(task_description)

        report = ""
        if isinstance(result, dict):
            report = result.get("report", "") or result.get("output", "") or str(result)
        elif isinstance(result, str):
            report = result
        else:
            report = str(result)

        return report

    async def _run_fallback(self, query: str) -> str:
        """
        Fallback sequential implementation if deepagents is not available.

        Runs planner -> researcher -> orchestrator sequentially using direct LLM calls.
        """
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Phase 1: Planning
        planner_prompt = render_prompt(
            "planner.j2",
            tools=self.tools_info,
            current_datetime=current_datetime,
            available_documents=[],
        )

        from langchain_core.messages import SystemMessage, HumanMessage

        plan_task = query
        if self.plan:
            plan_task += f"\n\nPre-approved plan: {json.dumps(self.plan)}"

        planner_messages = [
            SystemMessage(content=planner_prompt),
            HumanMessage(content=f"Create a research plan for: {plan_task}"),
        ]

        logger.info("Running planner phase...")
        plan_response = await self.planner_llm.ainvoke(planner_messages)
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

        for i, research_query in enumerate(queries_to_research[:6]):
            logger.info("Researching query %d/%d: %s", i + 1, len(queries_to_research), research_query[:80])

            researcher_messages = [
                SystemMessage(content=researcher_prompt),
                HumanMessage(content=research_query),
            ]

            researcher_llm_with_tools = self.researcher_llm.bind_tools(self.tools)

            response = await researcher_llm_with_tools.ainvoke(researcher_messages)

            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    for t in self.tools:
                        if t.name == tool_name:
                            try:
                                tool_result = t._run(**tool_args)
                                self._middleware.on_tool_result(tool_name, str(tool_result))
                                research_results.append(str(tool_result))
                            except Exception as e:
                                logger.error(f"Tool {tool_name} failed: {e}")
                            break

            if response.content:
                research_results.append(response.content)

        # Phase 3: Synthesis
        orchestrator_prompt = render_prompt(
            "orchestrator.j2",
            tools=self.tools_info,
            current_datetime=current_datetime,
            clarifier_result=self.clarifier_context,
            available_documents=[],
        )

        all_research = "\n\n---\n\n".join(research_results)
        verified_sources = self.source_registry.get_source_list_text()

        synthesis_messages = [
            SystemMessage(content=orchestrator_prompt),
            HumanMessage(content=(
                f"Write a comprehensive research report on: {query}\n\n"
                f"**Research Plan:**\n{plan_text}\n\n"
                f"**Research Findings:**\n{all_research}\n\n"
                f"**{verified_sources}**\n\n"
                f"Write the full report now."
            )),
        ]

        logger.info("Running synthesis/report writing phase...")
        report_response = await self.orchestrator_llm.ainvoke(synthesis_messages)
        return report_response.content or ""

    def _validate_report(self, report: str) -> ReportValidationResult:
        """Validate report completeness."""
        if not report or not report.strip():
            return ReportValidationResult(
                is_complete=False,
                feedback="Report is empty",
            )

        word_count = len(report.split())
        issues = []

        if word_count < 500:
            issues.append(f"Report is too short ({word_count} words, expected 3000+)")

        has_refs = bool(
            "## References" in report
            or "## Sources" in report
            or "**References" in report
        )
        if not has_refs:
            issues.append("Missing References/Sources section")

        heading_count = report.count("\n## ") + report.count("\n# ")
        if heading_count < 3:
            issues.append(f"Too few section headings ({heading_count}, expected 5+)")

        if issues:
            return ReportValidationResult(
                is_complete=False,
                feedback="; ".join(issues),
            )

        return ReportValidationResult(is_complete=True)

    @staticmethod
    def _extract_queries_from_plan(plan_text: str) -> list[str]:
        """Extract search queries from planner output."""
        queries = []
        try:
            json_match = None
            import re
            for m in re.finditer(r"\{.*\}", plan_text, re.DOTALL):
                try:
                    data = json.loads(m.group(0))
                    if "queries" in data:
                        json_match = data
                        break
                except json.JSONDecodeError:
                    continue

            if json_match and "queries" in json_match:
                for q in json_match["queries"]:
                    if isinstance(q, dict):
                        queries.append(q.get("query", ""))
                    elif isinstance(q, str):
                        queries.append(q)
        except Exception:
            pass

        return [q for q in queries if q.strip()]
