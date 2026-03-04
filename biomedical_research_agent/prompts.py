# Prompt templates for the Biomedical AI-Q Research Agent
# Virtual Screening prompts adapted from NVIDIA Biomedical AI-Q Research Agent Blueprint:
# https://github.com/NVIDIA-AI-Blueprints/biomedical-aiq-research-agent

CHECK_VIRTUAL_SCREENING = """Using report topic and report organization as a guide identify whether there is intention to do virtual screening.
Virtual screening is a computational technique used in drug discovery to identify potential drug candidates from large libraries of molecules.

# Report topic
{topic}

# Report organization
{report_organization}

# Instructions
1. From the report topic and report organization, determine if virtual screening would be helpful for what the user wants to research.
2. If the report topic is not a disease or medical condition, then the intention to do virtual screening would be 'no'.
3. If the report topic is a disease or medical condition, such as cystic fibrosis, and the report organization contains mentions of proposing new/novel small molecule therapies, mentions of intentions to do virtual screening, or mentions of the target protein and a recent or novel small molecule therapy, then the intention to do virtual screening would be 'yes'.
4. Output a binary intention 'yes' or 'no' to indicate whether the virtual screening is intended.

**Output example**
```json
{{
    "intention": "yes"
}}
```"""

CHECK_PROTEIN_MOLECULE_FOUND = """Using the current knowledge sources to identify whether the two ingredients needed for virtual screening are found already.
The two ingredients are: target protein related to the condition or disease, and a recent small molecule therapy for the condition or disease.
If either ingredient is missing, write a follow-up question for the missing ingredient(s).

# Report topic
{topic}

# Knowledge Sources
{knowledge_sources}

# Instructions
1. Focus on whether both of the two ingredients had already been found in the knowledge sources.
2. Ensure the follow-up question is self-contained and includes necessary context for web search.
3. If both of the ingredients are found in the current knowledge base, return the target protein and recent small molecule therapy. If the recent small molecule therapy is a combination of multiple molecules, pick only one molecule. For example, if the recent small molecule therapy is "A combination of Elexacaftor, Tezacaftor, and Ivacaftor", pick any one of the three molecules in the combination, and only return one molecule. Make sure you return a valid molecule name, and not a branded name for therapy that may contain multiple molecules such as Alyftrek. Format your response as a JSON object with the following keys:
- target_protein: the target protein for the disease or condition, be as succinct as possible, one word is ideal
- recent_small_molecule_therapy: a recent small molecule therapy that has been found in research, one molecule name only, be as succinct as possible, one word is ideal

**Output example**
```json
{{
    "target_protein": "CFTR",
    "recent_small_molecule_therapy": "Ivacaftor"
}}
```
4. If at least one ingredient is missing, return a query on one ingredient, format your response as a JSON object with the following keys:
- query: Write a specific follow up question to identify the missing ingredient (target protein or recent small molecule therapy)
- rationale: Describe what information is missing or needs clarification

**Output example**
```json
{{
    "query": "What is the target protein related to [specific condition or disease]?",
    "report_section": "Virtual Screening Details",
    "rationale": "The knowledge sources lack information about the target protein"
}}
```"""

COMBINE_VS_INTO_REPORT = """Given the intended report structure and existing report draft below, add one additional section exactly titled "Running Virtual Screening for Novel Small Molecule Therapies" into the intended report structure, immediately after the Abstract or Introduction section (whichever comes first).
Make sure to preserve the existing report draft and its existing format including sections, headings etc. Do not delete any content or sections from the existing report draft.

# Report structure
{report_organization}

# Report draft
{report}

# Virtual Screening Related Queries
{vs_queries}
{vs_queries_results}

# Virtual Screening Process and Output Information
{vs_info}

# Instructions
1. Preserve and copy over the original report draft and structure exactly as they are. Do not delete any content or sections from the existing report draft.
2. Add one additional self-contained section "Running Virtual Screening for Novel Small Molecule Therapies" immediately after the Abstract or Introduction section. The additional section should reflect the virtual screening process for proposing novel small-molecule therapies. In this section, first give a summary of Virtual Screening Related Queries from the provided context. Then copy over the provided Virtual Screening Process and Output Information exactly as-is, without deleting any content including lists, numbers, confidence scores, success, directory name. Do not remove or simplify any content from the source Virtual Screening Process and Output Information, copy over the content word-for-word.
3. Do not include information from Virtual Screening Related Queries or Virtual Screening Process and Output Information in other sections. These info should be self-contained in "Running Virtual Screening for Novel Small Molecule Therapies" section.
4. You should use proper markdown syntax when appropriate, as the text you generate will be rendered in markdown. Do NOT wrap the report in markdown blocks (e.g triple backticks).
5. Do not include any source citations, as these will be added to the report in post processing.
"""
