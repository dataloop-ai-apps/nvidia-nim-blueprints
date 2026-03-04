# Biomedical AI-Q Research Agent

## Quick setup

1. **(Optional) Prepare a RAG knowledge base:**
   - Run the [Preprocessing Multimodal PDF RAG](../multimodal_rag/preprocessing_multimodal_rag/README.md) pipeline to create a dataset of embedded chunks from your biomedical documents.
   - Install and configure the [NVIDIA RAG Pipeline](../multimodal_rag/nvidia_rag_pipeline/README.md) so it can retrieve information from those chunks.
2. Install this pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
3. Add your **NVIDIA NGC API Key** and **Tavily API Key** in [Data Governance](https://docs.dataloop.ai/docs/overview-1).
4. Set pipeline variables (see [Variables](#variables) below): **report_writer_model** and optionally **rag_pipeline_id**.
5. Create a prompt item with your biomedical research topic and run the pipeline.

For architecture, components, and troubleshooting, see the sections below.

### Variables

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| **report_writer_model** | Model | Yes | LLM for generating the final report. Recommended: NIM Llama 3.3 70B Instruct. |
| **rag_pipeline_id** | String | No | Pipeline ID of a configured [NVIDIA RAG Pipeline](../multimodal_rag/nvidia_rag_pipeline/README.md) instance. Enables RAG-first search with LLM-as-judge relevancy checking. Leave empty for web-search-only mode. |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NGC_API_KEY` | Yes | — | NVIDIA NGC API key for NIM model access |
| `TAVILY_API_KEY` | Yes | — | Tavily API key for web search |
| `MOLMIM_ENDPOINT_URL` | No | NVIDIA hosted API | MolMIM NIM endpoint for molecule generation. Defaults to `https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate` |
| `DIFFDOCK_ENDPOINT_URL` | No | NVIDIA hosted API | DiffDock NIM endpoint for molecular docking. Defaults to `https://health.api.nvidia.com/v1/biology/mit/diffdock` |

---

## Overview

The Biomedical AI-Q Research Agent is a Dataloop implementation of the [NVIDIA Biomedical AI-Q Research Agent Blueprint](https://build.nvidia.com/nvidia/biomedical-aiq-research-agent/blueprintcard). It extends the [AI Agent for Enterprise Research](../enterprise_research_agent/README.md) with virtual screening capabilities for biomedical drug discovery.

The agent performs deep research on biomedical topics using a Plan-Execute-Reflect agentic loop. When the research topic involves a disease or condition and the user requests novel therapeutic proposals, the agent automatically triggers a virtual screening branch that:

1. Identifies a **target protein** and **seed small-molecule therapy** from the research results
2. Looks up the protein 3D structure via [RCSB PDB](https://www.rcsb.org/) and the molecule's SMILES string via [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
3. Calls **MolMIM** (NVIDIA BioNeMo NIM) to generate novel candidate molecules
4. Calls **DiffDock** (NVIDIA BioNeMo NIM) to predict how those molecules dock against the target protein
5. Integrates the virtual screening results into the final research report

For non-biomedical topics, the pipeline behaves identically to the enterprise research agent — the virtual screening branch is skipped automatically.

## Features

- **Deep Research**: Plan-Execute-Reflect loop with RAG-first search, LLM-as-judge relevancy checking, and web search fallback.
- **Virtual Screening**: Automatic detection of drug-discovery intent with MolMIM molecule generation and DiffDock molecular docking.
- **Protein & Molecule Resolution**: Automated lookup of protein structures (RCSB PDB) and molecular SMILES strings (PubChem) from names extracted by the LLM.
- **Artifact Persistence**: DiffDock output files (`.mol` ligand positions, confidence scores CSV) are uploaded to the Dataloop dataset for downstream analysis.
- **Report Generation**: Final report formatted by NIM Llama 3.3 70B Instruct with virtual screening results seamlessly integrated.

## Components

### Pipeline Nodes

| Node | Description |
|------|-------------|
| **Init** | Validates configuration and optional RAG pipeline connection |
| **Biomedical AIQ Agent** | Orchestrator: generates queries, summarizes, reflects, checks VS intent, and routes between nodes |
| **Biomedical Research** | RAG-first search with LLM-as-judge + Tavily web search fallback (parallel query processing) |
| **Virtual Screening** | Resolves protein/molecule, calls MolMIM + DiffDock, uploads artifacts to Dataloop |
| **NIM Llama 3.3 70B - Report Writer** | Formats the final research report from the accumulated draft |

### Pipeline Flow

```
Input → [Init] → [Biomedical Agent]
                    |-- "research"          → [Research] → [Agent] (cycle)
                    |-- "virtual_screening" → [VS Node]  → [Agent]
                    '-- "generate_report"   → [NIM Llama 3.3 70B] (end)
```

### NIM Models

| Model | Purpose |
|-------|---------|
| **nvidia/llama-3.3-nemotron-super-49b-v1.5** | Reasoning: query generation, summarization, reflection, relevancy checking, VS intent detection, protein/molecule identification |
| **meta/llama-3.3-70b-instruct** | Report writing: final report formatting |
| **nvidia/molmim** | BioNeMo NIM: controlled generation of novel small molecules from a seed SMILES |
| **mit/diffdock** | BioNeMo NIM: prediction of 3D molecular docking poses and confidence scores |

## Usage

### 1. Install the Pipeline

Install from the Dataloop Marketplace. If you want RAG-enhanced retrieval over your own biomedical documents, also install the preprocessing and RAG pipelines first.

### 2. Configure Variables

Set `report_writer_model` to your Llama model. Optionally set `rag_pipeline_id` to enable RAG retrieval.

### 3. Create a Prompt Item

Create a prompt item in your dataset. To trigger virtual screening, include biomedical topic and mention novel therapies:

```
Topic: Cystic Fibrosis

Report Organization:
You are a medical researcher who specializes in cystic fibrosis.
Discuss advancements made in gene therapy for cystic fibrosis.
Discuss the efficacy of gene vs cell based therapies.
Propose novel small molecule therapies for the disease.
Include an abstract, and a section on each topic.
Format your answer in paragraphs.
Consider all (and only) relevant data.
Give a factual report with cited sources.

Number of queries: 5
Number of reflections: 2
```

For general (non-biomedical) research, simply provide any topic — virtual screening will be skipped automatically:

```
Topic: The impact of autonomous AI agents on enterprise workflows
```

### 4. Run the Pipeline

Execute the pipeline with your prompt item. The agent will:
1. Generate research queries and gather information from RAG and/or web
2. Summarize findings and reflect on gaps (iterating as configured)
3. Check if virtual screening is appropriate for the topic
4. If yes: identify protein + molecule, run MolMIM and DiffDock, integrate results
5. Produce a comprehensive report as an annotation on the original prompt item

## Prerequisites

### Required

- **NVIDIA NGC API Key**: For the Nemotron reasoning model, Llama report writer, MolMIM, and DiffDock
- **Tavily API Key**: For web search fallback

### Optional

- **RAG Pipeline**: For searching your own biomedical document corpus (see [Enterprise Research Agent](../enterprise_research_agent/README.md) for RAG setup details)
- **Self-hosted MolMIM / DiffDock**: Set `MOLMIM_ENDPOINT_URL` and `DIFFDOCK_ENDPOINT_URL` environment variables to point to your own NIM deployments instead of the NVIDIA hosted API

## Troubleshooting

### Virtual screening not triggered

- The LLM decides whether VS is appropriate based on the topic and report organization. Ensure your prompt mentions a disease/condition and requests novel small molecule therapies or drug discovery.
- Check service logs for the `Virtual screening intended: True/False` log line.

### Protein or molecule not found

- The agent searches research results and performs follow-up queries to identify the target protein and seed molecule. If your RAG corpus lacks this information and web search doesn't find it, VS will be skipped gracefully.
- Check logs for `VS ingredients found` or `VS intended but could not find protein/molecule`.

### MolMIM or DiffDock call fails

- Verify your NGC API key has access to the BioNeMo NIMs.
- If using self-hosted endpoints, ensure the URLs in `MOLMIM_ENDPOINT_URL` / `DIFFDOCK_ENDPOINT_URL` are correct and the services are running.
- DiffDock has a 5-minute timeout — large proteins may require more time.

### No RAG results used

- Verify `rag_pipeline_id` is set correctly and the RAG pipeline is installed and active.
- The LLM-as-judge may determine RAG results are not relevant; web search fallback will be used automatically.

### Short or generic report output

- Ensure the Llama model's `max_tokens` is set high enough.
- Verify that `nearestItems` context is reaching the model by checking the prompt item metadata.

## Contributing

We welcome contributions! Please submit bug reports or feature requests through the appropriate channels.
