import dtlpy as dl
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient, AsyncTavilyClient
from langsmith import traceable
import logging

logger = logging.getLogger('[ReportGeneration]')

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
    def load(self, local_path: str):
        if os.environ.get("TAVILY_API_KEY", None) is None:
            raise ValueError(f"Missing Tavily API key.")
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError(f"Missing NVIDIA API key.")
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY", None)
        self.nvidia_api_key = os.environ.get("NGC_API_KEY", None)

        
        # Initialize clients and models
        self.llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0)
        self.tavily_client = TavilyClient()
        self.tavily_async_client = AsyncTavilyClient()

    # Utility functions
    def deduplicate_and_format_sources(self, search_response, max_tokens_per_source, include_raw_content=True):
        """
        Takes either a single search response or list of responses from Tavily API and formats them.
        Limits the raw_content to approximately max_tokens_per_source.
        include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
        """
        # Convert input to list of results
        if isinstance(search_response, dict):
            sources_list = search_response['results']
        elif isinstance(search_response, list):
            sources_list = []
            for response in search_response:
                if isinstance(response, dict) and 'results' in response:
                    sources_list.extend(response['results'])
                else:
                    sources_list.extend(response)
        else:
            raise ValueError("Input must be either a dict with 'results' or a list of search results")
        
        # Deduplicate by URL
        unique_sources = {}
        for source in sources_list:
            if source['url'] not in unique_sources:
                unique_sources[source['url']] = source
        
        # Format output
        formatted_text = "Sources:\n\n"
        for i, source in enumerate(unique_sources.values(), 1):
            formatted_text += f"Source {source['title']}:\n===\n"
            formatted_text += f"URL: {source['url']}\n===\n"
            formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
            if include_raw_content:
                # Using rough estimate of 4 characters per token
                char_limit = max_tokens_per_source * 4
                # Handle None raw_content
                raw_content = source.get('raw_content', '')
                if raw_content is None:
                    raw_content = ''
                    print(f"Warning: No raw_content found for source {source['url']}")
                if len(raw_content) > char_limit:
                    raw_content = raw_content[:char_limit] + "... [truncated]"
                formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                    
        return formatted_text.strip()

    def format_sections(self, sections: List[Section]) -> str:
        """Format a list of sections into a string"""
        formatted_str = ""
        for idx, section in enumerate(sections, 1):
            formatted_str += f"""{'='*60}
                                Section {idx}: {section.name}
                                {'='*60}
                                Description:
                                {section.description}
                                Requires Research: 
                                {section.research}
                                Content:
                                {section.content if section.content else '[Not yet written]'}

                                """
        return formatted_str

    @traceable
    def tavily_search(self, query):
        """Search the web using the Tavily API."""
        return self.tavily_client.search(query, 
                             max_results=5, 
                             include_raw_content=True)

    @traceable
    async def tavily_search_async(self, search_queries, tavily_topic, tavily_days):
        """Performs concurrent web searches using the Tavily API."""
        search_tasks = []
        for query in search_queries:
            if tavily_topic == "news":
                search_tasks.append(
                    self.tavily_async_client.search(
                        query,
                        max_results=5,
                        include_raw_content=True,
                        topic="news",
                        days=tavily_days
                    )
                )
            else:
                search_tasks.append(
                    self.tavily_async_client.search(
                        query,
                        max_results=5,
                        include_raw_content=True,
                        topic="general"
                    )
                )

        # Execute all searches concurrently
        search_docs = await asyncio.gather(*search_tasks)
        return search_docs

    async def generate_report_plan(self, topic, report_structure, number_of_queries, tavily_topic, tavily_days=None):
        """Generate a plan for the report"""
        # Convert JSON object to string if necessary
        if isinstance(report_structure, dict):
            report_structure = str(report_structure)

        # Prompt to generate a search query to help with planning the report outline
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

        # Prompt generating the report outline
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

        # Generate search query
        structured_llm = self.llm.with_structured_output(Queries)
        
        # Format system instructions
        system_instructions_query = report_planner_query_writer_instructions.format(
            topic=topic, 
            report_organization=report_structure, 
            number_of_queries=number_of_queries
        )
        
        # Generate queries  
        results = structured_llm.invoke([
            SystemMessage(content=system_instructions_query),
            HumanMessage(content="Generate search queries that will help with planning the sections of the report.")
        ])
        
        # Web search
        query_list = [query.search_query for query in results.queries]
        search_docs = await self.tavily_search_async(query_list, tavily_topic, tavily_days)

        # Deduplicate and format sources
        source_str = self.deduplicate_and_format_sources(search_docs, max_tokens_per_source=1000, include_raw_content=True)

        # Format system instructions
        system_instructions_sections = report_planner_instructions.format(
            topic=topic, 
            report_organization=report_structure, 
            context=source_str
        )

        # Generate sections 
        structured_llm = self.llm.with_structured_output(Sections)
        report_sections = structured_llm.invoke([
            SystemMessage(content=system_instructions_sections),
            HumanMessage(content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields.")
        ])
        
        return report_sections.sections

    async def generate_section_content(self, section, number_of_queries, tavily_topic, tavily_days=None):
        """Generate content for a single section"""
        # Query writer instructions
        query_writer_instructions = """Your goal is to generate targeted web search queries that will gather comprehensive information for writing a technical report section.

        Topic for this section:
        {section_topic}

        When generating {number_of_queries} search queries, ensure they:
        1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
        2. Include specific technical terms related to the topic
        3. Target recent information by including year markers where relevant (e.g., "2025")
        4. Look for comparisons or differentiators from similar technologies/approaches
        5. Search for both official documentation and practical implementation examples

        Your queries should be:
        - Specific enough to avoid generic results
        - Technical enough to capture detailed implementation information
        - Diverse enough to cover all aspects of the section plan
        - Focused on authoritative sources (documentation, technical blogs, academic papers)"""

        # Section writer instructions
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

        # Generate queries
        structured_llm = self.llm.with_structured_output(Queries)
        
        # Format system instructions for queries
        system_instructions = query_writer_instructions.format(
            section_topic=section.description, 
            number_of_queries=number_of_queries
        )
        
        # Generate queries
        queries = structured_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Generate search queries on the provided topic.")
        ])
        
        # Web search
        query_list = [query.search_query for query in queries.queries]
        search_docs = await self.tavily_search_async(query_list, tavily_topic, tavily_days)
        
        # Deduplicate and format sources
        source_str = self.deduplicate_and_format_sources(search_docs, max_tokens_per_source=5000, include_raw_content=True)
        
        # Format system instructions for section writing
        system_instructions = section_writer_instructions.format(
            section_title=section.name,
            section_topic=section.description,
            context=source_str
        )
        
        # Generate section content
        section_content = self.llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Generate a report section based on the provided sources.")
        ])
        
        # Update section content
        section.content = section_content.content
        return section

    async def write_final_section(self, section, completed_sections_content):
        """Write a final section (intro or conclusion) based on other completed sections"""
        # Final section writer instructions
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
        
        # Format system instructions
        system_instructions = final_section_writer_instructions.format(
            section_title=section.name,
            section_topic=section.description,
            context=completed_sections_content
        )
        
        # Generate section content
        section_content = self.llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Generate a report section based on the provided content.")
        ])
        
        # Update section content
        section.content = section_content.content
        return section

    async def generate_report(self, topic: str, report_structure: str, number_of_queries: int = 2, 
                       tavily_topic: str = "general", tavily_days: Optional[int] = None) -> str:
        """
        Generate a report on the given topic using the specified structure.
        
        Args:
            topic: The topic of the report
            report_structure: The structure of the report
            number_of_queries: Number of web search queries to perform per section
            tavily_topic: Type of Tavily search to perform ('general' or 'news')
            tavily_days: Number of days to look back for news articles (only used when tavily_topic='news')
            
        Returns:
            The generated report as a string
        """
        # Generate report plan
        sections = await self.generate_report_plan(
            topic=topic,
            report_structure=report_structure,
            number_of_queries=number_of_queries,
            tavily_topic=tavily_topic,
            tavily_days=tavily_days
        )
        
        # Process sections that require research
        research_sections = []
        non_research_sections = []
        
        for section in sections:
            if section.research:
                research_sections.append(section)
            else:
                non_research_sections.append(section)
        
        # Generate content for research sections
        completed_research_sections = []
        for section in research_sections:
            completed_section = await self.generate_section_content(
                section=section,
                number_of_queries=number_of_queries,
                tavily_topic=tavily_topic,
                tavily_days=tavily_days
            )
            completed_research_sections.append(completed_section)
        
        # Format completed research sections for context
        completed_sections_content = self.format_sections(completed_research_sections)
        
        # Generate content for non-research sections
        completed_non_research_sections = []
        for section in non_research_sections:
            completed_section = await self.write_final_section(
                section=section,
                completed_sections_content=completed_sections_content
            )
            completed_non_research_sections.append(completed_section)
        
        # Combine all sections in the original order
        all_completed_sections = {}
        for section in completed_research_sections + completed_non_research_sections:
            all_completed_sections[section.name] = section
        
        # Maintain original order
        ordered_sections = [all_completed_sections[section.name] for section in sections]
        
        # Compile final report
        final_report = "\n\n".join([section.content for section in ordered_sections])
        
        return final_report

    
    def _extract_parameters_from_prompt(self, prompt_text_variable: str):
        """Extract parameters from the prompt text"""
        lines = prompt_text_variable.strip().split('\n')
        
        # Initialize parameters
        params = {
            'topic': '',
            'report_structure': '',
            'number_of_queries': 2,  # Default value
            'tavily_topic': 'general',  # Default value
            'tavily_days': None  # Default value
        }
        
        # Extract topic (first line after "Topic:")
        for i, line in enumerate(lines):
            if line.strip().startswith('Topic:'):
                params['topic'] = line.replace('Topic:', '').strip()
                break
        
        # Extract structure (everything between "Structure:" and "Number of queries:")
        structure_lines = []
        capture_structure = False
        for line in lines:
            if line.strip().startswith('Structure:'):
                capture_structure = True
                continue
            elif line.strip().startswith('Number of queries:'):
                capture_structure = False
                continue
            
            if capture_structure:
                structure_lines.append(line)
        
        params['report_structure'] = '\n'.join(structure_lines).strip()
        
        # Extract number of queries
        for line in lines:
            if line.strip().startswith('Number of queries:'):
                try:
                    params['number_of_queries'] = int(line.replace('Number of queries:', '').strip())
                except ValueError:
                    logger.warning("Could not parse number of queries, using default value")
                break
        
        # Extract Tavily topic
        for line in lines:
            if line.strip().startswith('Tavily Topic:'):
                try:
                    params['tavily_topic'] = line.replace('Tavily Topic:', '').strip()
                except ValueError:
                    logger.warning("Could not parse Tavily Topic, using default 'general' value")
                break
        
        # Extract Tavily days
        for line in lines:
            if line.strip().startswith('Tavily Days:'):
                try:
                    days_value = line.replace('Tavily Days:', '').strip()
                    params['tavily_days'] = int(days_value) if days_value.isdigit() else None
                except ValueError:
                    logger.warning("Could not parse Tavily days, using default value")
                break
        
        return params
    
    def report_planning(self, item: dl.Item):
        return item
    
    def report_sections(self, item: dl.Item):
        return item, item, item, item

    def research_agents(self, item: dl.Item):
        return item
    
    def search_tavily(self, item: dl.Item):
        return item
    
    def report_writing(self, item: dl.Item, intro: dl.Item, body: dl.Item, conclusion: dl.Item):
        """
        Run the report generation service on a Dataloop item.
        
        Args:
            item: The item to process
            number_of_queries: Number of web search queries to perform per section
            tavily_days: Number of days to look back for news articles (only used when tavily_topic='news')
            
        Returns:
            The generated report as a string
        """
        prompt_item = dl.PromptItem.from_item(item)
        prompt_text = prompt_item.to_json()['prompts']['1'][0]['value']
        
        params = self._extract_parameters_from_prompt(prompt_text)
        if params['topic'] == '' or params['report_structure'] == '':
            raise ValueError("Topic and report structure are required. Please refer to the documentation for the correct format.")
        
        # Generate the report
        report = asyncio.run(self.generate_report(
            topic= params['topic'] ,
            report_structure=params['report_structure'],
            number_of_queries=params['number_of_queries'],
            tavily_topic=params['tavily_topic'],
            tavily_days=params['tavily_days']
        ))
        
        prompt_item.add(
            prompt_key='1', 
            message={
                "role": "assistant",
                "content": [{
                    "mimetype": dl.PromptType.TEXT,
                    "value": report
                }]
            },
            model_info={
                'name': 'llama_3.3_70b_instruct',
                'confidence': 1.0,
                'model_id': 'llama_3.3_70b_instruct-1'
            }
        )
        return report
    
    def test_report_generation(self):
        # Generate report
        self.load(local_path="")
        report = self.final_report(item=dl.items.get(item_id="67ceba78f89ab3cb0989e022"))
        return report
    
if __name__ == "__main__":
    runner = ReportGenerator()
    runner.test_report_generation()
