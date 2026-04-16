# NVIDIA AIQ Enterprise Research Agent v2

Dataloop implementation of the [NVIDIA AI-Q Blueprint v2.0.0](https://github.com/NVIDIA-AI-Blueprints/aiq).

## Features

- **Two-Tier Research**: Automatic routing between fast shallow research and comprehensive deep research based on query complexity.
- **Intent Classification**: Uses `nemotron-3-nano-30b-a3b` to classify queries as meta (casual chat), shallow research, or deep research.
- **Shallow Research**: Bounded tool-calling with LangGraph for quick factual answers with citation verification.
- **Deep Research**: Multi-agent system using `deepagents` (orchestrator/planner/researcher sub-agents) for comprehensive, long-form research reports.
- **Human-in-the-Loop (HITL)**: For deep research, the agent proposes a plan and waits for user approval in the AI Playground chat before proceeding.
- **Citation Verification**: 5-level URL matching against a source registry, with report sanitization to remove hallucinated URLs.
- **RAG Integration**: Optional Dataloop RAG pipeline wrapped as a LangChain tool for knowledge base queries.
- **Report Writer**: Dedicated NIM `gpt-oss-120b` predict node for deep research report formatting with streaming output (shallow answers are returned directly).

## Pipeline Architecture

```
Input -> [Init] -> [Intent Classifier]
                     |-- meta -> annotate response (end)
                     |-- shallow -> [Shallow Researcher]
                     |                |-- answer -> annotate response (end)
                     |                '-- escalate -> [Clarifier]
                     '-- deep -> [Clarifier]
                                   |-- plan_pending -> (present plan, wait for approval)
                                   '-- (next cycle, approved) -> [Deep Researcher] -> [GPT-OSS 120B Report Writer]
```

## Pipeline Nodes

| Node | Function | Description |
|------|----------|-------------|
| Init | `init_research` | Validates config, detects plan approval cycles |
| Intent Classifier | `classify_intent` | Routes queries by intent and depth |
| Shallow Researcher | `shallow_research` | Fast bounded research with LangGraph |
| Clarifier | `clarify_and_plan` | Two-phase clarification + plan generation |
| Deep Researcher | `deep_research` | Multi-agent research using deepagents |
| Report Writer | NIM predict | GPT-OSS 120B for final report formatting |

## Models Used

| Role | Model | Usage |
|------|-------|-------|
| Intent Classifier | `nemotron-3-nano-30b-a3b` | Fast classification via inline ChatNVIDIA |
| Shallow Researcher | `nemotron-3-nano-30b-a3b` | Bounded research with tool-calling |
| Clarifier | `nemotron-3-nano-30b-a3b` | Clarification and plan generation |
| Orchestrator | `gpt-oss-120b` | Deep research coordination and synthesis |
| Planner | `gpt-oss-120b` | Research plan generation |
| Researcher (deep) | `nemotron-3-nano-30b-a3b` | Information gathering for deep research |
| Report Writer | `gpt-oss-120b` | Final deep research report formatting (dedicated NIM node) |

## Pipeline Variables

| Variable | Type | Description |
|----------|------|-------------|
| `report_writer_model` | Model | NIM GPT-OSS 120B model for report generation |
| `rag_pipeline_id` | String | Optional RAG pipeline ID for knowledge base integration |

## HITL Plan Approval Flow

1. User asks a complex question in AI Playground
2. Pipeline creates a cycle, Intent Classifier routes to "deep"
3. Clarifier generates a research plan and presents it as a chat annotation
4. User reads the plan in the chat and replies (e.g. "approved", "looks good", or feedback)
5. The reply creates a new pipeline cycle
6. Init node detects `plan_pending` metadata and reads the latest message
7. Any response is treated as **approval by default** — the user's message is passed along so the deep researcher can incorporate any additional instructions
8. Only explicit rejection ("no", "reject", "cancel", "stop", "abort") re-routes to the Clarifier for a revised plan

## Usage

This pipeline is designed to be used from the **Dataloop AI Playground**. Select the pipeline in the AI Playground dropdown and start a conversation — the agent will automatically classify your query and route it through the appropriate research path.

## AI Playground Timeout

The AI Playground has a **5-minute response timeout**. Shallow research and meta queries complete well within this limit, but deep research (which involves multi-agent orchestration + the NIM report writer) typically takes **15–25 minutes**.

When deep research is triggered:

- The **plan presentation** will appear in the chat normally (takes a few seconds)
- After approving the plan, the AI Playground may show a **"Failed to get response"** error — this is expected and does not mean the pipeline failed
- The pipeline continues running in the background

To view the completed deep research report:

1. **From the pipeline**: Open the pipeline execution, find the Report Writer node output, and click the output item to see the full report
2. **From the dataset**: Go to the `ai-playground-history` dataset, find the prompt item for your conversation, and open it — the report will be in the last response


## RAG Integration (Optional)

To enable knowledge base retrieval from your own documents, you can connect a RAG pipeline:

1. Create a RAG pipeline from the **NVIDIA RAG Blueprint** template (`nim-rag-bp`). See the [NVIDIA RAG Pipeline documentation](../multimodal_rag/nvidia_rag_pipeline/README.md) for setup instructions.
2. Configure the RAG pipeline with your dataset and embedding/reranking models, then install and activate it.
3. Copy the RAG pipeline's ID.
4. In this pipeline's variables, paste the ID into the **`rag_pipeline_id`** field.

When configured, the research agents will use the RAG pipeline as an additional tool (`knowledge_search`) alongside web search, allowing them to retrieve information from your uploaded documents and internal knowledge base.

If left empty, the agent uses web search only.

## Deployment

1. Go to the **Dataloop Marketplace** and find **AI Agent for Enterprise Research v2** under the Pipelines tab
2. Install the pipeline into your project
3. Configure the pipeline variables:
   - **`report_writer_model`**: NIM GPT-OSS 120B model (already selected)
   - **`rag_pipeline_id`**: (optional) paste a RAG pipeline ID for knowledge base integration
4. Ensure the required integrations are configured in your project (`NGC_API_KEY`, `TAVILY_API_KEY`)
5. Publish the pipeline
6. Open the **AI Playground**, select this pipeline, and start asking questions
