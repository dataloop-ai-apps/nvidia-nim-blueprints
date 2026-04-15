"""
Clarifier agent for AIQ v2.

Two-phase clarification and plan generation:
  Phase 1: Determine if the query needs clarification (research_clarification.j2)
  Phase 2: Generate a lightweight research plan (plan_generation.j2)

Designed to work with Dataloop AI Playground pipeline cycles:
  - Writes plan as annotation visible in chat
  - Stores metadata for plan-approval detection in next cycle
"""

import json
import logging
from typing import Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

from enterprise_research_agent_v2.prompts import render_prompt

logger = logging.getLogger("[AIQ-v2-Clarifier]")


class Clarifier:
    """Two-phase clarification and plan generation agent."""

    def __init__(self, llm: ChatNVIDIA):
        self.llm = llm

    def needs_clarification(
        self,
        query: str,
        clarifier_context: Optional[str] = None,
    ) -> dict:
        """
        Phase 1: Determine if the query needs clarification.

        Returns dict with:
          - needs_clarification: bool
          - clarification_question: str | None
        """
        system_content = render_prompt(
            "research_clarification.j2",
            clarifier_result=clarifier_context,
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]

        try:
            response = self.llm.invoke(messages)
            response_text = (response.content or "").strip()
            parsed = self._extract_json(response_text)

            if parsed and isinstance(parsed, dict):
                return {
                    "needs_clarification": bool(parsed.get("needs_clarification", False)),
                    "clarification_question": parsed.get("clarification_question"),
                }
        except Exception as e:
            logger.error(f"Clarification check failed: {e}")

        return {"needs_clarification": False, "clarification_question": None}

    def generate_plan(
        self,
        query: str,
        clarifier_context: Optional[str] = None,
        feedback_history: Optional[list[str]] = None,
    ) -> dict:
        """
        Phase 2: Generate a lightweight research plan.

        Returns dict with:
          - title: str
          - sections: list[str]
        """
        system_content = render_prompt(
            "plan_generation.j2",
            clarifier_context=clarifier_context,
            feedback_history=feedback_history or [],
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]

        try:
            response = self.llm.invoke(messages)
            response_text = (response.content or "").strip()
            parsed = self._extract_json(response_text)

            if parsed and isinstance(parsed, dict):
                return {
                    "title": parsed.get("title", "Research Plan"),
                    "sections": parsed.get("sections", []),
                }
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")

        return {
            "title": f"Research: {query[:80]}",
            "sections": [
                "Introduction and Background",
                "Key Findings",
                "Analysis",
                "Conclusion",
            ],
        }

    def run(
        self,
        query: str,
        clarifier_context: Optional[str] = None,
    ) -> dict:
        """
        Run the full two-phase clarification + plan generation.

        Returns dict with:
          - needs_clarification: bool
          - clarification_question: str | None (if clarification needed)
          - plan: dict | None (if no clarification needed, contains title + sections)
          - clarifier_log: str (log of the clarification process)
        """
        # Phase 1: Check if clarification is needed
        clarification = self.needs_clarification(query, clarifier_context)

        if clarification["needs_clarification"] and clarification["clarification_question"]:
            return {
                "needs_clarification": True,
                "clarification_question": clarification["clarification_question"],
                "plan": None,
                "clarifier_log": f"Clarification needed: {clarification['clarification_question']}",
            }

        # Phase 2: Generate plan
        plan = self.generate_plan(query, clarifier_context)

        plan_text = self._format_plan_for_chat(plan, query)

        return {
            "needs_clarification": False,
            "clarification_question": None,
            "plan": plan,
            "clarifier_log": f"Plan generated: {plan['title']}",
            "plan_text": plan_text,
        }

    @staticmethod
    def _format_plan_for_chat(plan: dict, query: str) -> str:
        """Format the plan as a readable message for AI Playground chat."""
        sections_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.get("sections", [])))
        return (
            f"I've created a research plan for your query.\n\n"
            f"**{plan.get('title', 'Research Plan')}**\n\n"
            f"**Planned Sections:**\n{sections_list}\n\n"
            f"Please reply with **'approved'** to start the deep research, "
            f"or provide feedback to adjust the plan."
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        import re
        cleaned = text
        while "<think>" in cleaned and "</think>" in cleaned:
            start = cleaned.find("<think>")
            end = cleaned.find("</think>") + len("</think>")
            cleaned = cleaned[:start] + cleaned[end:]
        if "<think>" in cleaned:
            cleaned = cleaned[:cleaned.find("<think>")]

        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None
