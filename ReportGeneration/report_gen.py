import dtlpy as dl
import asyncio
from typing import List, Optional, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import logging

# Import necessary libraries for report generation
from tavily import TavilyClient, AsyncTavilyClient

logger = logging.getLogger(['ReportGenerator'])

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.",)
    description: str = Field(description="Brief overview of the main topics and concepts to be covered in this section.",)
    research: bool = Field(description="Whether to perform web research for this section of the report.")
    content: str = Field(description="The content of the section.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.",)

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(description="List of search queries.",)

class ReportGenerator(dl.BaseServiceRunner):
    def __init__(self, model: Optional[dl.Model] = None):
        # Initialize with default API keys
        os.environ["TAVILY_API_KEY"] = ""

        # Initialize clients
        self.tavily_client = TavilyClient()
        self.tavily_async_client = AsyncTavilyClient()

    def _prepare_report_query(self, topic: str, report_structure: str, number_of_queries: int) -> str:
        """Prepare prompt for generating search queries for report planning"""
        report_planner_query_writer_instructions = """You are an expert technical writer, helping to plan a report. 

        The report will be focused on the following topic:

        {topic}

        The report structure will follow these guidelines:

        {report_organization}

        Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the report sections. 

        The query should:

        1. Be related to the topic 
        2. Help satisfy the requirements specified in the report organization

        Make the query specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure."""
        
        return report_planner_query_writer_instructions.format(
            topic=topic,
            report_organization=report_structure,
            number_of_queries=number_of_queries
        )

    def _prepare_section_generation(self, topic: str, report_structure: str, context: str) -> str:
        """Prepare prompt for generating report sections"""
        report_planner_instructions = """You are an expert technical writer, helping to plan a report.

        Your goal is to generate the outline of the sections of the report. 

        The overall topic of the report is:

        {topic}

        The report should follow this organization: 

        {report_organization}

        You should reflect on this information to plan the sections of the report: 

        {context}

        Now, generate the sections of the report. Each section should have the following fields:

        - Name - Name for this section of the report.
        - Description - Brief overview of the main topics and concepts to be covered in this section.
        - Research - Whether to perform web research for this section of the report.
        - Content - The content of the section, which you will leave blank for now.

        Consider which sections require web research. For example, introduction and conclusion will not require research because they will distill information from other parts of the report."""
        
        return report_planner_instructions.format(
            topic=topic,
            report_organization=report_structure,
            context=context
        )

    def _prepare_section_query(self, section_description: str, number_of_queries: int) -> str:
        """Prepare prompt for generating search queries for a section"""
        query_writer_instructions = """Your goal is to generate targeted web search queries that will gather comprehensive information for writing a technical report section.

        Topic for this section:
        {section_topic}

        When generating {number_of_queries} search queries, ensure they:
        1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
        2. Include specific technical terms related to the topic
        3. Target recent information by including year markers where relevant (e.g., "2024")
        4. Look for comparisons or differentiators from similar technologies/approaches
        5. Search for both official documentation and practical implementation examples

        Your queries should be:
        - Specific enough to avoid generic results
        - Technical enough to capture detailed implementation information
        - Diverse enough to cover all aspects of the section plan
        - Focused on authoritative sources (documentation, technical blogs, academic papers)"""
        
        return query_writer_instructions.format(
            section_topic=section_description,
            number_of_queries=number_of_queries
        )

    def _prepare_section_writing(self, section_name: str, section_description: str, context: str) -> str:
        """Prepare prompt for writing a section"""
        section_writer_instructions = """You are an expert technical writer crafting one section of a technical report.

        Topic for this section:
        {section_topic}

        Guidelines for writing:

        1. Technical Accuracy:
        - Include specific version numbers
        - Reference concrete metrics/benchmarks
        - Cite official documentation
        - Use technical terminology precisely

        2. Length and Style:
        - Strict 150-200 word limit
        - No marketing language
        - Technical focus
        - Write in simple, clear language
        - Start with your most important insight in **bold**
        - Use short paragraphs (2-3 sentences max)

        3. Structure:
        - Use ## for section title (Markdown format)
        - Only use ONE structural element IF it helps clarify your point:
        * Either a focused table comparing 2-3 key items (using Markdown table syntax)
        * Or a short list (3-5 items) using proper Markdown list syntax:
            - Use `*` or `-` for unordered lists
            - Use `1.` for ordered lists
            - Ensure proper indentation and spacing
        - End with ### Sources that references the below source material formatted as:
        * List each source with title, date, and URL
        * Format: `- Title : URL`

        3. Writing Approach:
        - Include at least one specific example or case study
        - Use concrete details over general statements
        - Make every word count
        - No preamble prior to creating the section content
        - Focus on your single most important point

        4. Use this source material to help write the section:
        {context}

        5. Quality Checks:
        - Exactly 150-200 words (excluding title and sources)
        - Careful use of only ONE structural element (table or list) and only if it helps clarify your point
        - One specific example / case study
        - Starts with bold insight
        - No preamble prior to creating the section content
        - Sources cited at end"""
        
        return section_writer_instructions.format(
            section_title=section_name,
            section_topic=section_description,
            context=context
        )

    def _prepare_final_section(self, section_name: str, section_description: str, context: str) -> str:
        """Prepare prompt for writing a final section (intro/conclusion)"""
        final_section_writer_instructions = """You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

        Section to write: 
        {section_topic}

        Available report content:
        {context}

        1. Section-Specific Approach:

        For Introduction:
        - Use # for report title (Markdown format)
        - 50-100 word limit
        - Write in simple and clear language
        - Focus on the core motivation for the report in 1-2 paragraphs
        - Use a clear narrative arc to introduce the report
        - Include NO structural elements (no lists or tables)
        - No sources section needed

        For Conclusion/Summary:
        - Use ## for section title (Markdown format)
        - 100-150 word limit
        - For comparative reports:
            * Must include a focused comparison table using Markdown table syntax
            * Table should distill insights from the report
            * Keep table entries clear and concise
        - For non-comparative reports: 
            * Only use ONE structural element IF it helps distill the points made in the report:
            * Either a focused table comparing items present in the report (using Markdown table syntax)
            * Or a short list using proper Markdown list syntax:
            - Use `*` or `-` for unordered lists
            - Use `1.` for ordered lists
            - Ensure proper indentation and spacing
        - End with specific next steps or implications
        - No sources section needed

        3. Writing Approach:
        - Use concrete details over general statements
        - Make every word count
        - Focus on your single most important point

        4. Quality Checks:
        - For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
        - For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
        - Markdown format
        - Do not include word count or any preamble in your response"""
        
        return final_section_writer_instructions.format(
            section_title=section_name,
            section_topic=section_description,
            context=context
        )

    async def generate_report_pipeline(self, topic: str, report_structure: str, number_of_queries: int = 2,
                                     tavily_topic: str = "general", tavily_days: Optional[int] = None) -> str:
        """Pipeline function that runs each step sequentially"""
        
        # Step 1: Generate initial queries
        query_prompt = self._prepare_report_query(topic, report_structure, number_of_queries)
        queries = await self._get_llm_response(query_prompt, Queries)
        
        # Step 2: Search for initial context
        search_docs = await self.tavily_search_async(
            [q.search_query for q in queries.queries],
            tavily_topic,
            tavily_days
        )
        source_str = self.deduplicate_and_format_sources(search_docs, max_tokens_per_source=1000, include_raw_content=True)
        
        # Step 3: Generate sections
        sections_prompt = self._prepare_section_generation(topic, report_structure, source_str)
        sections = await self._get_llm_response(sections_prompt, Sections)
        
        # Step 4: Process research sections
        completed_research = []
        for section in [s for s in sections.sections if s.research]:
            # Generate queries for section
            query_prompt = self._prepare_section_query(section.description, number_of_queries)
            section_queries = await self._get_llm_response(query_prompt, Queries)
            
            # Search for section content
            section_docs = await self.tavily_search_async(
                [q.search_query for q in section_queries.queries],
                tavily_topic,
                tavily_days
            )
            section_sources = self.deduplicate_and_format_sources(section_docs, max_tokens_per_source=5000, include_raw_content=True)
            
            # Generate section content
            content_prompt = self._prepare_section_writing(section.name, section.description, section_sources)
            section.content = await self._get_llm_response(content_prompt)
            completed_research.append(section)
        
        # Step 5: Process non-research sections
        completed_non_research = []
        for section in [s for s in sections.sections if not s.research]:
            prompt = self._prepare_final_section(section.name, section.description, self.format_sections(completed_research))
            section.content = await self._get_llm_response(prompt)
            completed_non_research.append(section)
        
        # Step 6: Combine all sections in original order
        all_sections = {**{s.name: s for s in completed_research},
                       **{s.name: s for s in completed_non_research}}
        ordered_sections = [all_sections[s.name] for s in sections.sections]
        
        return "\n\n".join([s.content for s in ordered_sections])