# AI Agent for Enterprise Research

## Quick setup

1. **(Optional) Prepare a RAG knowledge base:**
   - Run the [Preprocessing Multimodal PDF RAG](../multimodal_rag/preprocessing_multimodal_rag/README.md) pipeline to create a dataset of embedded chunks from your documents.
   - Install and configure the [NVIDIA RAG Pipeline](../multimodal_rag/nvidia_rag_pipeline/README.md) so it can retrieve information from those chunks.
2. Install this pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
3. Add your **NVIDIA NGC API Key** and **Tavily API Key** in [Data Governance](https://docs.dataloop.ai/docs/overview-1).
4. Set pipeline variables (see [Variables](#variables) below): **report_writer_model** and optionally **rag_pipeline_id**.
5. Create a prompt item with your research topic and run the pipeline or use AI playground with your pipeline.

For architecture, components, and troubleshooting, see the sections below.

### Variables

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| **report_writer_model** | Model | Yes | LLM for generating the final report. Recommended: NIM Llama 3.3 70B Instruct. |
| **rag_pipeline_id** | String | No | Pipeline ID of a configured [NVIDIA RAG Pipeline](../multimodal_rag/nvidia_rag_pipeline/README.md) instance. Enables RAG-first search with LLM-as-judge relevancy checking. Leave empty for web-search-only mode. |

**Getting the RAG pipeline ID:** Open your installed NVIDIA RAG Pipeline in the Dataloop platform, copy the pipeline ID from the URL or pipeline settings, and paste it into the `rag_pipeline_id` variable when configuring the research agent pipeline.

**Getting the model ID:** When you run or edit the pipeline, the Model variable shows a model selector. Choose the recommended model; the selected value is the model ID (e.g. `nim-llama-3-3-70b-instruct.models.nim-llama-3-3-70b-instruct`). You can also find model IDs in your project Models page.

---

## Overview

The AI Agent for Enterprise Research is a Dataloop implementation of the [NVIDIA AIQ Research Assistant Blueprint](https://build.nvidia.com/nvidia/aiq). It automates deep research on any topic using a Plan-Execute-Reflect agentic loop, producing comprehensive, publication-ready long-form reports.

The agent generates search queries, retrieves information from both a RAG knowledge base (if configured) and web search, summarizes findings, reflects on gaps, and iterates until the research is complete. A final NIM Llama model formats the accumulated research into a polished report.

## Prerequisites

### Required

- **NVIDIA NGC API Key**: For the Nemotron reasoning model and Llama report writer
- **Tavily API Key**: For web search fallback

### Optional (for RAG-enhanced research)

Setting up RAG enables the agent to search your own document corpus before falling back to web search. This requires two pipelines to be installed and configured beforehand:

1. **Preprocessing Pipeline**: Run the [Preprocessing Multimodal PDF RAG](../multimodal_rag/preprocessing_multimodal_rag/README.md) pipeline on your documents to create embedded chunks in a dataset.

2. **NVIDIA RAG Pipeline**: Install and configure the [NVIDIA RAG Pipeline](../multimodal_rag/nvidia_rag_pipeline/README.md) with:
   - `retrieval_dataset` pointing to the embedded chunks dataset from step 1
   - `retrieval_embed_model` matching the embedding model used in preprocessing
   - `gen_ai_model` set to a suitable LLM for RAG responses

   Once the RAG pipeline is installed and active, copy its **pipeline ID** and paste it into the `rag_pipeline_id` variable of the research agent pipeline.

> Without a RAG pipeline configured, the agent operates in **web-search-only mode** using Tavily, which is fully functional for general research topics.

## NIM Models

| Model | Purpose |
|-------|---------|
| **nvidia/llama-3.3-nemotron-super-49b-v1.5** | Reasoning: query generation, summarization, reflection, relevancy checking (temperature 0.5, max 5000 tokens) |
| **meta/llama-3.3-70b-instruct** | Report writing: final report formatting (temperature 0.0, max 20000 tokens) |

## Usage

### 1. Install the Pipeline

Install from the Dataloop Marketplace. If you want RAG, also install the preprocessing and RAG pipelines first.

### 2. Configure Variables

Set `report_writer_model` to your Llama model. Optionally set `rag_pipeline_id` to enable RAG retrieval.

### 3. Create a Prompt Item

Create a prompt item in your dataset with the following format:

```
Topic: The impact of autonomous AI agents on enterprise workflows
Report Organization:
1. What are autonomous AI agents and how do they differ from traditional automation?
2. Key enterprise use cases
3. Architectural patterns and frameworks
4. Risks, governance, and reliability challenges
5. Market trajectory and adoption outlook
Number of queries: 4
Number of reflections: 2
```

All fields are optional except the topic. Defaults: 3 queries, 2 reflections, auto-generated report organization.

You can also provide a simple question without any structure:

```
What are the environmental and economic trade-offs of nuclear fusion vs solar energy?
```

### 4. Run the Pipeline

Execute the pipeline with your prompt item. The agent will research the topic and produce a comprehensive report as an annotation on the original prompt item.

## Troubleshooting

### No RAG results used

- Verify `rag_pipeline_id` is set correctly and the RAG pipeline is installed and active.
- Check that the RAG pipeline's retrieval dataset contains relevant embedded chunks.
- The LLM-as-judge may determine RAG results are not relevant (expected if the topic doesn't match your documents). Web search fallback will be used automatically.

### Short or generic report output

- Ensure the Llama model's `max_tokens` is set high enough (recommended: 20000).
- Check that the model's system prompt includes report-writing instructions.
- Verify that `nearestItems` context is reaching the model by checking the prompt item metadata.

### Pipeline execution fails

- Verify NVIDIA NGC API key and Tavily API key are configured in Data Governance.
- Check that all required services are running.
- Review service logs for error details.

## Contributing

We welcome contributions! Please submit bug reports or feature requests through the appropriate channels.
