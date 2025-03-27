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
import json
import re

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
    def __init__(self):
        if os.environ.get("TAVILY_API_KEY", None) is None:
            raise ValueError(f"Missing Tavily API key.")
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY", None)

        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        self.tavily_async_client = AsyncTavilyClient(api_key=self.tavily_api_key)

        self.all_completed_sections = {}

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

    def tavily_search(self, search_queries, tavily_topic, tavily_days):
        """Performs web searches using the Tavily API."""
        search_results = []
        for query in search_queries:
            if tavily_topic == "news":
                result = self.tavily_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    topic="news",
                    days=tavily_days
                )
            else:
                result = self.tavily_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    topic="general"
                )
            search_results.append(result)
        
        return search_results

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
    
    def report_search_queries(self, item: dl.Item):
        """
        First node in the pipeline - Extract parameters and generate report plan queries
        
        Args:
            item: Dataloop item containing the prompt
            
        Returns:
            item: Updated item with report planning prompt
        """            
        # Extract parameters from the prompt
        prompt_item = dl.PromptItem.from_item(item)
        prompt_text = prompt_item.to_json()['prompts']['1'][0]['value']
        self.params = self._extract_parameters_from_prompt(prompt_text)

        # Validate parameters
        if self.params['topic'] == '' or self.params['report_structure'] == '':
            raise ValueError("Topic and report structure are required. Please refer to the documentation for the correct format.")
        
        # Create report planner query writer prompt
        report_planner_query_writer_instructions = f"""You are an expert technical writer, helping to plan a report. 

        The report will be focused on the following topic:

        {self.params['topic']}

        The report structure will follow these guidelines:

        {self.params['report_structure']}

        Your goal is to generate {self.params['number_of_queries']} search queries that will help gather comprehensive information for planning the report sections. 

        The query should:

        1. Be related to the topic 
        2. Help satisfy the requirements specified in the report organization

        Make the query specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
        
        Important:
        Each line should be a search query and nothing else."""
        
        prompt_item_search_queries = dl.PromptItem(name='report_search_queries')
        prompt1 = dl.Prompt(key='1')
        prompt1.add_element(
            mimetype=dl.PromptType.TEXT,
            value=report_planner_query_writer_instructions
        )
        prompt_item_search_queries.prompts.append(prompt1)
        item_search_queries = item.dataset.items.upload(prompt_item_search_queries, overwrite=True, remote_path="/.dataloop")

        item.metadata.setdefault('user', {})
        item.metadata['user']['item_search_queries'] = item_search_queries.id
        item.update(True)
        item_search_queries.metadata.setdefault('user', {})
        item_search_queries.metadata['user']['main_item'] = item.id
        item_search_queries.update(True)  
        
        return item_search_queries
    
    def search_tavily(self, item: dl.Item):
        queries = item.annotations.list()[0].coordinates
        query_list = [line.strip() for line in queries.split('\n') if line.strip()]
        search_docs = self.tavily_search(query_list, self.params['tavily_topic'], self.params['tavily_days'])
        # Deduplicate and format sources
        source_str = self.deduplicate_and_format_sources(search_docs, max_tokens_per_source=1000, include_raw_content=True)
        return item, source_str
    
    def report_planning(self, item: dl.Item, source_str: str):
        """
        Process the LLM's search queries and generate report sections
        """
        # Prompt generating the report outline
        report_planner_instructions = f"""You are an expert technical writer, helping to plan a report.

        Your goal is to generate the outline of the sections of the report. 

        The overall topic of the report is:

        {self.params['topic']}

        The report should follow this organization: 

        {self.params['report_structure']}

        You should reflect on this information to plan the sections of the report: 

        {source_str}

        Now, generate the sections of the report. Each section should have the following fields:

        - Name - Name for this section of the report.
        - Description - Brief overview of the main topics and concepts to be covered in this section.
        - Research - Whether to perform web research for this section of the report.
        - Content - The content of the section, which you will leave blank for now.

        Consider which sections require web research. For example, introduction and conclusion will not require research because they will distill information from other parts of the report.
        
        Important:
        Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields. 
        The sections should be in this JSON format (without any additional text):
        
        sections: [
          section1: 
            name: "Introduction"
            description: "Brief overview of the topic and its importance"
            research: false
            content: ""
          section2:
            name: "Section Name"
            description: "Description of what this section covers"
            research: true
            content: ""
        ]
        
        Ensure your response can be parsed as valid JSON with the proper structure."""
        prompt_item_report_planning = dl.PromptItem(name='report_planning')
        prompt1 = dl.Prompt(key='1')
        prompt1.add_element(
            mimetype=dl.PromptType.TEXT,
            value=report_planner_instructions
        )
        prompt_item_report_planning.prompts.append(prompt1)
        item_report_planning = item.dataset.items.upload(prompt_item_report_planning, overwrite=True, remote_path="/.dataloop")

        main_item = dl.items.get(item_id=item.metadata['user']['main_item'])
        main_item.metadata['user']['item_report_planning'] = item_report_planning.id
        main_item.update(True)

        item_report_planning.metadata.setdefault('user', {})
        item_report_planning.metadata['user']['main_item'] = main_item.id
        item_report_planning.update(True)
        return item_report_planning
    
    def report_sections(self, item: dl.Item):
        """
        Process the LLM's search queries and generate report sections
        """
        sections_str = item.annotations.list()[0].coordinates

        # Extract the sections array directly from the string
        sections_match = re.search(r'sections:\s*\[(.*?)\]', sections_str, re.DOTALL)
        sections = []

        if sections_match:
            sections_content = sections_match.group(1)
            # Split by section objects (each starting with {)
            section_objects = re.findall(r'\s*{\s*(.*?)\s*}', sections_content, re.DOTALL)
            
            for section_str in section_objects:
                section = {}
                # Extract name
                name_match = re.search(r'"name":\s*"([^"]*)"', section_str)
                if name_match:
                    section['name'] = name_match.group(1)
                
                # Extract description
                desc_match = re.search(r'"description":\s*"([^"]*)"', section_str)
                if desc_match:
                    section['description'] = desc_match.group(1)
                
                # Extract research flag
                research_match = re.search(r'"research":\s*(true|false)', section_str)
                if research_match:
                    section['research'] = research_match.group(1) == 'true'
                
                # Extract content
                content_match = re.search(r'"content":\s*"([^"]*)"', section_str)
                if content_match:
                    section['content'] = content_match.group(1)
                
                sections.append(section)

        research_sections_prompt_items = []
        non_research_sections_prompt_items = []

        main_item = dl.items.get(item_id=item.metadata['user']['main_item'])
        main_item.metadata.setdefault('user', {})
        main_item.metadata['user']['sections'] = sections
        # Process sections that require research
        for i, section in enumerate(sections):
            if section.get('research', False):
                # Query writer instructions
                query_writer_instructions = f"""Your goal is to generate targeted web search queries that will gather comprehensive information for writing a technical report section.

                Topic for this section:
                {section['name']}

                Description:
                {section['description']}

                When generating 2 search queries, ensure they:
                1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
                2. Include specific technical terms related to the topic
                3. Target recent information by including year markers where relevant (e.g., "2023")
                4. Look for comparisons or differentiators from similar technologies/approaches
                5. Search for both official documentation and practical implementation examples

                Your queries should be:
                - Specific enough to avoid generic results
                - Technical enough to capture detailed implementation information
                - Diverse enough to cover all aspects of the section plan
                - Focused on authoritative sources (documentation, technical blogs, academic papers)
                
                Important:
                Each line should be a search query and nothing else."""

                prompt_item_research = dl.PromptItem(name=f'section_{i}')
                prompt1 = dl.Prompt(key='1')
                prompt1.add_element(
                    mimetype=dl.PromptType.TEXT,
                    value=query_writer_instructions
                )
                prompt_item_research.prompts.append(prompt1)
                item_research = item.dataset.items.upload(prompt_item_research, overwrite=True, remote_path="/.dataloop")

                main_item.metadata.setdefault('user', {})
                main_item.metadata['user'][f'section_item_{i}'] = item_research.id
                main_item.update(True)
                item_research.metadata.setdefault('user', {})
                item_research.metadata['user']['main_item'] = main_item.id
                item_research.update(True)
                research_sections_prompt_items.append(item_research)
            else:
                final_section_writer_instructions = f"""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

                Section to write: 
                {section['name']}

                Available report content:
                {section['description']}

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
                - Do not include word count or any preamble in your response
                
                Important:
                Generate a report section based on the provided content."""

                prompt_item_non_research = dl.PromptItem(name=f'section_{i}')
                prompt1 = dl.Prompt(key='1')
                prompt1.add_element(
                    mimetype=dl.PromptType.TEXT,
                    value=final_section_writer_instructions
                ) 
                prompt_item_non_research.prompts.append(prompt1)
                item_non_research = item.dataset.items.upload(prompt_item_non_research, overwrite=True, remote_path="/.dataloop")
                
                main_item.metadata.setdefault('user', {})
                main_item.metadata['user'][f'section_item_{i}'] = item_non_research.id
                main_item.update(True)
                item_non_research.metadata.setdefault('user', {})
                item_non_research.metadata['user']['main_item'] = main_item.id
                item_non_research.update(True)
                non_research_sections_prompt_items.append(item_non_research)
        
        return research_sections_prompt_items, non_research_sections_prompt_items
        
    def write_final_section_research(self, item: dl.Item, source_str: str):
        """
        Process the LLM's search queries and generate report sections
        """
        # Get section name and description from the item metadata
        main_item = dl.items.get(item_id=item.metadata['user']['main_item'])
        sections = eval(main_item.metadata['user']['sections'])
        section_number = None
        if item.name:
            match = re.search(r'section_(\d+)', item.name)
            if match:
                section_number = match.group(1)
            else:
                # Try to extract number if it's in a different format
                match = re.findall(r'\d+', item.name)
                if match:
                    section_number = match[0]
        
        if section_number is None:
            logger.warning(f"Could not extract section number from item name: {item.name}")
        section_topic = eval(sections[int(section_number)])['name']
        context = source_str
        
        # Section writer instructions
        section_writer_instructions = f"""You are an expert technical writer crafting one section of a technical report.

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
        - Sources cited at end
        
        Important:
        Generate a report section based on the provided content."""
        
        prompt_item_research = dl.PromptItem(name=f'research_section_{item.name}')
        prompt1 = dl.Prompt(key='1')
        prompt1.add_element(
            mimetype=dl.PromptType.TEXT,
            value=section_writer_instructions
        )
        prompt_item_research.prompts.append(prompt1)
        item_research = item.dataset.items.upload(prompt_item_research, overwrite=True, remote_path="/.dataloop")

        main_item = dl.items.get(item_id=item.metadata['user']['main_item'])
        main_item.metadata.setdefault('user', {})
        main_item.metadata['user'][f'item_research_{item.name}'] = item_research.id
        main_item.update()
        item_research.metadata.setdefault('user', {})
        item_research.metadata['user']['main_item'] = main_item.id
        item_research.update()
        return item_research
    
    def gather_sections(self, item: dl.Item):

        section = item.annotations.list()[0].coordinates
        self.all_completed_sections[section.name] = section

        main_item = dl.items.get(item_id=item.metadata['user']['main_item'])
        # Maintain original order
        ordered_sections = [self.all_completed_sections[section.name] for section in main_item.metadata['user']['sections']]
        
        # Compile final report
        final_report = "\n\n".join([section.content for section in ordered_sections])
        
        prompt_item = dl.PromptItem.from_item(main_item)
        prompt_item.add(
            prompt_key='1', 
            message={
                "role": "assistant",
                "content": [{
                    "mimetype": dl.PromptType.TEXT,
                    "value": final_report
                }]
            },
            model_info={
                'name': 'llama_3.3_70b_instruct',
                'confidence': 1.0,
                'model_id': 'llama_3.3_70b_instruct-1'
            }
        )
        return main_item