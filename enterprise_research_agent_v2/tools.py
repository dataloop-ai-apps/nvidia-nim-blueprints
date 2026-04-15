"""
LangChain tool wrappers for AIQ v2 research agents.

Provides tools for:
  - Tavily web search
  - Serper academic paper search (Google Scholar)
  - Dataloop RAG pipeline (wrapped as a LangChain tool)
"""

import json
import logging
import os
import re
import time
from typing import Optional

import requests
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

logger = logging.getLogger(__name__)


class TavilySearchTool(BaseTool):
    """Web search using Tavily API."""

    name: str = "web_search_tool"
    description: str = "Search the web for real-time information. Returns relevant content and URLs."
    max_results: int = Field(default=5)
    relevance_threshold: float = Field(default=0.6)
    _client: object = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, api_key: str, max_results: int = 5, **kwargs):
        super().__init__(max_results=max_results, **kwargs)
        from tavily import TavilyClient
        self._client = TavilyClient(api_key=api_key)

    def _run(self, query: str) -> str:
        try:
            result = self._client.search(
                query,
                max_results=self.max_results,
                include_raw_content=True,
                topic="general",
            )
            filtered = [
                r for r in result.get("results", [])
                if float(r.get("score", 0)) > self.relevance_threshold
            ]
            if not filtered:
                return "No relevant results found."

            output_parts = []
            for r in filtered:
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                content = r.get("content", "")[:1000]
                output_parts.append(f"**{title}**\nURL: {url}\n{content}\n")
            return "\n---\n".join(output_parts)
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return f"Search failed: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


class TavilyAdvancedSearchTool(BaseTool):
    """Advanced web search using Tavily API with raw content extraction."""

    name: str = "advanced_web_search_tool"
    description: str = (
        "Advanced web search with full content extraction. "
        "Use for deeper research when standard search is insufficient."
    )
    max_results: int = Field(default=2)
    _client: object = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, api_key: str, max_results: int = 2, **kwargs):
        super().__init__(max_results=max_results, **kwargs)
        from tavily import TavilyClient
        self._client = TavilyClient(api_key=api_key)

    def _run(self, query: str) -> str:
        try:
            result = self._client.search(
                query,
                max_results=self.max_results,
                include_raw_content=True,
                search_depth="advanced",
                topic="general",
            )
            results = result.get("results", [])
            if not results:
                return "No results found."

            output_parts = []
            for r in results:
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                raw = r.get("raw_content", r.get("content", ""))[:3000]
                output_parts.append(f"**{title}**\nURL: {url}\n{raw}\n")
            return "\n---\n".join(output_parts)
        except Exception as e:
            logger.error(f"Tavily advanced search error: {e}")
            return f"Search failed: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


class SerperPaperSearchTool(BaseTool):
    """Academic paper search using Serper API (Google Scholar)."""

    name: str = "paper_search_tool"
    description: str = (
        "Search academic papers and scientific publications via Google Scholar. "
        "Use for scientific or technical validation."
    )
    max_results: int = Field(default=5)
    api_key: str = Field(default="")

    def __init__(self, api_key: str, max_results: int = 5, **kwargs):
        super().__init__(api_key=api_key, max_results=max_results, **kwargs)

    def _run(self, query: str) -> str:
        try:
            response = requests.post(
                "https://google.serper.dev/scholar",
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": self.max_results},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("organic", [])
            if not results:
                return "No academic papers found."

            output_parts = []
            for r in results:
                title = r.get("title", "Untitled")
                link = r.get("link", "")
                snippet = r.get("snippet", "")
                year = r.get("year", "")
                cited_by = r.get("citedBy", {}).get("total", "")
                publication = r.get("publication", "")
                output_parts.append(
                    f"**{title}** ({year})\n"
                    f"Publication: {publication}\n"
                    f"URL: {link}\n"
                    f"Cited by: {cited_by}\n"
                    f"{snippet}\n"
                )
            return "\n---\n".join(output_parts)
        except Exception as e:
            logger.error(f"Serper paper search error: {e}")
            return f"Paper search failed: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


class DataloopRAGTool(BaseTool):
    """Wraps a Dataloop RAG pipeline as a LangChain tool for knowledge retrieval."""

    name: str = "knowledge_search"
    description: str = (
        "Search internal knowledge base and uploaded documents. "
        "Use for questions about internal data, uploaded files, or enterprise documents."
    )
    rag_pipeline_id: str = Field(default="")
    dataset_id: str = Field(default="")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, rag_pipeline_id: str, dataset_id: str, **kwargs):
        super().__init__(
            rag_pipeline_id=rag_pipeline_id,
            dataset_id=dataset_id,
            **kwargs,
        )

    def _run(self, query: str) -> str:
        try:
            import dtlpy as dl

            rag_pipeline = dl.pipelines.get(pipeline_id=self.rag_pipeline_id)
            if rag_pipeline.status != "Installed":
                return "Knowledge base is not available."

            dataset = dl.datasets.get(dataset_id=self.dataset_id)

            safe_name = re.sub(r"[^\w\s-]", "", query)[:30].strip().replace(" ", "_")
            prompt_item = dl.PromptItem(name=f"rag_{safe_name}_{int(time.time())}")
            prompt_item.add(
                message={
                    "role": "user",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": query}],
                }
            )

            rag_prompt_item = dataset.items.upload(
                prompt_item,
                remote_path="/.dataloop/aiq_v2_rag/",
                overwrite=True,
            )

            execution = rag_pipeline.execute(
                execution_input={
                    "item": {
                        "item_id": rag_prompt_item.id,
                        "dataset_id": dataset.id,
                    }
                }
            )

            max_wait = 300
            poll_interval = 5
            elapsed = 0
            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval
                success, response = dl.client_api.gen_request(
                    req_type="get",
                    path=f"/pipelines/{self.rag_pipeline_id}/executions/{execution.id}",
                )
                if success:
                    status = response.json().get("status", "")
                    if status in ("success", "completed"):
                        break
                    elif status in ("failed", "error"):
                        return "Knowledge search failed."

            rag_prompt_item = dl.items.get(item_id=rag_prompt_item.id)
            annotations = rag_prompt_item.annotations.list()
            if annotations:
                answer = str(annotations[-1].coordinates)
                if answer.strip() and answer.strip() != query.strip():
                    return f"**Knowledge Base Result:**\n{answer}"

            return "No relevant results found in knowledge base."
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return f"Knowledge search failed: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


def build_tools(
    tavily_api_key: Optional[str] = None,
    serper_api_key: Optional[str] = None,
    rag_pipeline_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> list[BaseTool]:
    """Build the list of available LangChain tools based on configured API keys."""
    tools = []

    if tavily_api_key:
        tools.append(TavilySearchTool(api_key=tavily_api_key))
        tools.append(TavilyAdvancedSearchTool(api_key=tavily_api_key))
    else:
        logger.warning("No Tavily API key - web search disabled")

    if serper_api_key:
        tools.append(SerperPaperSearchTool(api_key=serper_api_key))
    else:
        logger.info("No Serper API key - paper search disabled")

    if rag_pipeline_id and dataset_id:
        tools.append(DataloopRAGTool(
            rag_pipeline_id=rag_pipeline_id,
            dataset_id=dataset_id,
        ))
    else:
        logger.info("No RAG pipeline configured - knowledge search disabled")

    return tools
