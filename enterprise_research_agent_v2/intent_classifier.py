"""
Intent classifier for AIQ v2.

Classifies user queries as:
  - meta: greetings, identity questions, casual chat -> direct response
  - research/shallow: simple factual queries -> shallow researcher
  - research/deep: complex multi-faceted queries -> clarifier -> deep researcher
"""

import json
import logging
from datetime import datetime
from typing import Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

from enterprise_research_agent_v2.prompts import render_prompt

logger = logging.getLogger("[AIQ-v2-IntentClassifier]")


class IntentClassifier:
    """Classifies user intent and determines research depth."""

    def __init__(self, llm: ChatNVIDIA, tools_info: Optional[list[dict]] = None):
        self.llm = llm
        self.tools_info = tools_info or []

    def classify(self, query: str, user_info: Optional[dict] = None) -> dict:
        """
        Classify query intent and depth.

        Returns dict with:
          - intent: "meta" | "research"
          - meta_response: str | None (response text if meta)
          - research_depth: "shallow" | "deep" | None
          - depth_reasoning: str | None
        """
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        system_content = render_prompt(
            "intent_classification.j2",
            query=query,
            current_datetime=current_datetime,
            user_info=user_info or {},
            tools=self.tools_info,
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]

        try:
            response = self.llm.invoke(messages)
            response_text = (response.content or "").strip()
            parsed = self._extract_json(response_text)

            if not parsed or not isinstance(parsed, dict):
                logger.warning("Failed to parse intent classifier response, defaulting to shallow research")
                return {
                    "intent": "research",
                    "meta_response": None,
                    "research_depth": "shallow",
                    "depth_reasoning": "Parse failed, defaulting to shallow",
                }

            intent = (parsed.get("intent") or "research").strip().lower()
            if intent not in ("meta", "research"):
                intent = "research"

            return {
                "intent": intent,
                "meta_response": parsed.get("meta_response"),
                "research_depth": (parsed.get("research_depth") or "shallow").strip().lower(),
                "depth_reasoning": parsed.get("depth_reasoning"),
            }

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": "research",
                "meta_response": None,
                "research_depth": "shallow",
                "depth_reasoning": f"Classification error: {e}",
            }

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract JSON from LLM response, handling think tags and markdown."""
        cleaned = text
        # Strip <think> tags
        while "<think>" in cleaned and "</think>" in cleaned:
            start = cleaned.find("<think>")
            end = cleaned.find("</think>") + len("</think>")
            cleaned = cleaned[:start] + cleaned[end:]
        if "<think>" in cleaned:
            cleaned = cleaned[:cleaned.find("<think>")]

        # Try direct parse
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        brace_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None
