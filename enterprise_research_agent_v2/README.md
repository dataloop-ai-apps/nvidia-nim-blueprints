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
- **Report Writer**: Dedicated NIM `gpt-oss-120b` predict node for final report formatting with streaming output.

## Pipeline Architecture

```
Input -> [Init] -> [Intent Classifier]
                     |-- meta -> annotate response (end)
                     |-- shallow -> [Shallow Researcher]
                     |                |-- answer -> [GPT-OSS 120B Report Writer]
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
| Shallow Researcher | `llama-3.3-nemotron-super-49b-v1.5` | Research with tool-calling |
| Clarifier | `nemotron-3-nano-30b-a3b` | Clarification and plan generation |
| Orchestrator | `llama-3.3-nemotron-super-49b-v1.5` | Deep research coordination |
| Planner | `llama-3.3-nemotron-super-49b-v1.5` | Research plan generation |
| Researcher | `llama-3.3-nemotron-super-49b-v1.5` | Information gathering |
| Report Writer | `gpt-oss-120b` | Final report (dedicated NIM node) |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NGC_API_KEY` | Yes | NVIDIA NGC API key for NIM endpoints |
| `TAVILY_API_KEY` | Yes | Tavily API key for web search |
| `SERPER_API_KEY` | No | Serper API key for academic paper search |
| `NVIDIA_BASE_URL` | No | Override NIM API base URL |
| `INTENT_MODEL` | No | Override intent classifier model |
| `SHALLOW_MODEL` | No | Override shallow researcher model |
| `CLARIFIER_MODEL` | No | Override clarifier model |
| `ORCHESTRATOR_MODEL` | No | Override deep research orchestrator model |
| `PLANNER_MODEL` | No | Override deep research planner model |
| `RESEARCHER_MODEL` | No | Override deep research researcher model |

## Pipeline Variables

| Variable | Type | Description |
|----------|------|-------------|
| `report_writer_model` | Model | NIM GPT-OSS 120B model for report generation |
| `rag_pipeline_id` | String | Optional RAG pipeline ID for knowledge base integration |

## HITL Plan Approval Flow

1. User asks a complex question in AI Playground
2. Pipeline creates a cycle, Intent Classifier routes to "deep"
3. Clarifier generates a research plan and presents it as a chat annotation
4. User reads the plan in the chat and replies with "approved" or feedback
5. The reply creates a new pipeline cycle
6. Init node detects `plan_pending` metadata and the approval/feedback
7. If approved: routes directly to Deep Researcher
8. If feedback: routes back through Intent Classifier -> Clarifier with the feedback

## Deployment

1. Build the Docker image using the provided Dockerfile
2. Install the app DPK (`dataloop.json`)
3. Create a pipeline from the template (`pipeline/dataloop.json`)
4. Configure the pipeline variables (RAG pipeline ID, report writer model)
5. Set the required environment variables as Dataloop integrations
6. Install and activate the pipeline
