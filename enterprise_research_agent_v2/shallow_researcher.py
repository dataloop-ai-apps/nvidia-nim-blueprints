"""
Shallow research agent for AIQ v2.

Provides fast, bounded research with tool-calling using a LangGraph StateGraph.
Includes citation verification, synthesis anchoring, and escalation to deep research.

Architecture matches the NVIDIA AI-Q Blueprint v2.0.0 shallow_researcher.
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from enterprise_research_agent_v2.prompts import render_prompt
from enterprise_research_agent_v2.citation_verification import (
    EmptySourceRegistryError,
    Source,
    SourceRegistry,
    extract_sources_from_tool_result,
    sanitize_report,
    verify_citations,
)

logger = logging.getLogger("[AIQ-v2-ShallowResearcher]")


class ShallowResearchState(TypedDict, total=False):
    """State for the shallow research LangGraph."""
    messages: list[BaseMessage]
    tool_iterations: int


class ShallowResearcherAgent:
    """
    Fast bounded research agent using LangGraph with tool-calling.

    Performs quick lookups, generates cited answers, and can escalate
    to deep research when results are insufficient.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        max_tool_iterations: int = 5,
        max_llm_turns: int = 10,
    ):
        self.llm = llm
        self.tools = tools
        self.max_tool_iterations = max_tool_iterations
        self.max_llm_turns = max_llm_turns
        self.source_registry = SourceRegistry()
        self.tools_info = [
            {"name": getattr(t, "name", str(t)), "description": getattr(t, "description", "")}
            for t in tools
        ]
        self._graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph StateGraph with agent + tool nodes."""

        agent_self = self

        async def agent_node(state: ShallowResearchState) -> dict[str, Any]:
            messages = state.get("messages", [])
            iterations = state.get("tool_iterations", 0)

            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rendered_prompt = render_prompt(
                "shallow_researcher.j2",
                tools=agent_self.tools_info,
                user_info=None,
                current_datetime=current_datetime,
                available_documents=[],
            )
            system_message = SystemMessage(content=rendered_prompt)

            has_tool_results = any(isinstance(m, ToolMessage) for m in messages)

            if iterations >= agent_self.max_tool_iterations or has_tool_results:
                if iterations >= agent_self.max_tool_iterations:
                    logger.warning("Max iterations (%d) reached. Forcing synthesis.", iterations)
                synthesis_anchor = HumanMessage(
                    content=(
                        "Using the search results above, synthesize a comprehensive answer to the "
                        "user's question now. Cite sources inline with [1], [2], etc. and include a "
                        "'**References:**' section at the end listing each cited URL. "
                        "Do not attempt any further tool calls."
                    )
                )
                full_messages = [system_message] + list(messages) + [synthesis_anchor]
                response = await agent_self.llm.ainvoke(full_messages)
                return {"messages": [response], "tool_iterations": iterations}

            llm_with_tools = agent_self.llm.bind_tools(agent_self.tools, parallel_tool_calls=True)
            full_messages = [system_message] + list(messages)
            response = await llm_with_tools.ainvoke(full_messages)

            new_iterations = iterations
            if hasattr(response, "tool_calls") and response.tool_calls:
                new_iterations += len(response.tool_calls)
                logger.info("Added %d tool calls. Total: %d", len(response.tool_calls), new_iterations)

            return {"messages": [response], "tool_iterations": new_iterations}

        source_tool_names = {t.name for t in self.tools}

        async def tool_node_with_capture(state: ShallowResearchState) -> dict[str, Any]:
            """Execute tools and capture source URLs for citation verification."""
            tool_node = ToolNode(agent_self.tools)
            result = await tool_node.ainvoke(state)
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage) and msg.content:
                    tool_name = getattr(msg, "name", "") or ""
                    if tool_name not in source_tool_names:
                        continue
                    sources = extract_sources_from_tool_result(tool_name, str(msg.content))
                    for source in sources:
                        agent_self.source_registry.add(source)
                    if sources:
                        logger.info(
                            "Captured %d source(s) from %s",
                            len(sources), tool_name,
                        )
            return result

        builder = StateGraph(ShallowResearchState)
        builder.set_entry_point("agent")
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node_with_capture)
        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", "__end__": "__end__"},
        )
        builder.add_edge("tools", "agent")

        return builder.compile()

    async def run(self, query: str) -> dict:
        """
        Execute shallow research.

        Returns dict with:
          - answer: str (the research answer)
          - sources: list[dict] (source registry as dicts)
          - should_escalate: bool (whether to escalate to deep research)
        """
        self.source_registry.clear()

        state: ShallowResearchState = {
            "messages": [HumanMessage(content=query)],
            "tool_iterations": 0,
        }

        recursion_limit = (self.max_llm_turns * 2) + 10
        result = await self._graph.ainvoke(state, config={"recursion_limit": recursion_limit})

        messages = result.get("messages", [])
        if not messages:
            return {
                "answer": "An error occurred during research.",
                "sources": [],
                "should_escalate": True,
            }

        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        # Citation verification
        if self.source_registry.all_sources():
            verification = verify_citations(content, self.source_registry)
            content = verification.verified_report
            logger.info(
                "Citation verification: %d valid, %d removed",
                len(verification.valid_citations),
                len(verification.removed_citations),
            )
        else:
            logger.warning("No sources captured during shallow research")

        # Sanitize report
        sanitization = sanitize_report(content)
        content = sanitization.sanitized_report

        # Check if escalation is needed
        should_escalate = self._should_escalate(content)

        return {
            "answer": content,
            "sources": self.source_registry.to_dict(),
            "should_escalate": should_escalate,
        }

    @staticmethod
    def _should_escalate(content: str) -> bool:
        """Check if the shallow research answer warrants escalation to deep research."""
        if not content or not content.strip():
            return True

        lower = content.lower()
        escalation_signals = [
            "i don't have enough information",
            "unable to find",
            "need more research",
            "insufficient results",
            "could not find relevant",
            "no relevant results",
        ]
        return any(signal in lower for signal in escalation_signals)
