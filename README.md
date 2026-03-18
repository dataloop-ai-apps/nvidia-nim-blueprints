# NVIDIA NIM Blueprints for Dataloop

## Quick setup

1. Open the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace) and find the blueprint you want (Report Generation, PDF to Podcast, Multimodal RAG, or Video Search & Summarization).
2. Install the pipeline into your project.
3. Add the required API keys in your organization’s [Data Governance](https://docs.dataloop.ai/docs/overview-1) (e.g. NVIDIA NGC, and Tavily or ElevenLabs where needed).
4. Configure and run the pipeline from your project.

For per-blueprint steps and requirements, see the README for each app in [Blueprints](#blueprints) below, or the detailed [Installation](#installation) section.

---

A collection of NVIDIA NIM-powered blueprints for the Dataloop Platform. These blueprints leverage NVIDIA's AI models to enable advanced GenAI workflows including document processing, report generation, and multimodal RAG pipelines.

## Blueprints

| Blueprint | Description |
|-----------|-------------|
| [Report Generation](report_generation/README.md) | Automated report creation with web research and LLM-powered content generation |
| [PDF to Podcast](pdf_to_podcast/README.md) | Transform PDF documents into podcast-ready audio content |
| [Multimodal RAG - Preprocessing](multimodal_rag/preprocessing_multimodal_rag/README.md) | PDF extraction and embedding pipeline for RAG |
| [Multimodal RAG - Retrieval](multimodal_rag/nvidia_rag_pipeline/README.md) | Document retrieval and response generation with human-in-the-loop |
| [VSS - Preprocessing](video_search_summarization/README.md) | Ingest videos, describe with VILA VLM, transcribe with Parakeet ASR, embed for vector search, and build a knowledge graph |
| [VSS - Retrieval](video_search_summarization/README.md#2-retrieval-pipeline) | Query the knowledge graph and vector store, then generate answers with an LLM |

## Blueprint Overviews

### Video Search & Summarization (VSS)

Implemented as two pipelines that work together, similar to the Multimodal RAG blueprint:

**1. Preprocessing Pipeline** — processes video content into searchable text and a knowledge graph. Splits videos into time-based chunks and runs two branches in parallel:

- **Visual branch**: Wraps each sub-video in a prompt → describes it with VILA 1.5 3B → extracts the text response
- **Audio branch**: Extracts audio → transcribes with NVIDIA Parakeet CTC 0.6B ASR

Both branches clone results to a target dataset, which then fans out to embedding (for vector search) and graph entity extraction (for knowledge graph construction).

```
                              ┌─ Video to Prompt → VILA VLM → Prompt to Text ─┐
Dataset → Video to Videos ────┤                                               ├──→ Clone to Dataset ─┬─→ Embedding
                              └─ Audio Extract → Parakeet ASR ────────────────┘                      │
                                                                                                     └─→ Text to Prompt → Graph Entity Extraction → Add Chunk to Graph
```

**2. Retrieval Pipeline** — answers questions by combining graph RAG with vector search:

```
Prompt → Embedding → Retriever → Graph Query → Response LLM
```

**Required API Keys:** NVIDIA NGC API Key

### Report Generation

Automates comprehensive report creation on any topic. The pipeline uses NVIDIA NIM's Llama 3.3 70B model combined with Tavily web search to research, plan, and generate well-structured reports with minimal user input.

**Key Features:**
- Automated report planning and section generation
- Web research integration via Tavily API
- Customizable report structures
- Support for news or general research modes

### PDF to Podcast

Transforms PDF documents into audio content using text-to-speech technology. Supports both monologue and dialogue formats with customizable speakers and duration.

**Key Features:**
- PDF text extraction and processing
- Monologue or two-host dialogue generation
- High-quality audio synthesis via ElevenLabs
- Configurable podcast duration and focus

### Multimodal RAG Pipeline

Based on NVIDIA's [Multimodal RAG](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag) blueprint. On the Dataloop platform, this blueprint is implemented as two separate pipelines that work together:

1. **Preprocessing Pipeline**: Extracts text and images from PDFs, generates chunks, and creates embeddings using NVIDIA NIM models (YOLOX, PaddleOCR, Llama 3.2 NeMoRetriever 1B VLM Embed v1).

2. **Retrieval Pipeline**: Retrieves relevant documents based on queries and generates responses using Llama 3.1 405B, with human-in-the-loop validation.

## Installation

1. Access the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace)
2. Search for the desired blueprint (Report Generation, PDF to Podcast, Multimodal RAG, or Video Search & Summarization)
3. Install the pipeline to your project
4. Configure required API keys in your organization's [Data Governance](https://docs.dataloop.ai/docs/overview-1)

## Requirements

| Blueprint | Required API Keys |
|-----------|-------------------|
| Report Generation | NVIDIA NGC API Key, Tavily API Key |
| PDF to Podcast | NVIDIA NGC API Key, ElevenLabs API Key |
| Multimodal RAG | NVIDIA NGC API Key |
| Video Search & Summarization | NVIDIA NGC API Key |

## NVIDIA Models Used

- **VILA 1.5 3B** - Video description via vision-language understanding (VSS)
- **Llama 3.2 NeMoRetriever 1B VLM Embed v1** - Text embeddings for vector search (VSS, Multimodal RAG)
- **Llama 3.3 70B Instruct** - Report planning and content generation
- **Llama 3.1 405B Instruct** - RAG response generation, PDF to Podcast script generation
- **Llama 3.1 70B Instruct** - PDF to Podcast content processing
- **Llama 3.1 8B Instruct** - Knowledge graph entity extraction (VSS preprocessing), RAG response generation (VSS retrieval), PDF to Podcast content processing
- **YOLOX Page Elements** - PDF layout analysis
- **PaddleOCR** - Optical character recognition

For more information on NVIDIA NIMs, visit [NVIDIA Build](https://build.nvidia.com/explore/discover).

## Important Note on NVIDIA NIM Model Usage

While the code in this repository is open-sourced, users of NVIDIA NIM models must adhere to NVIDIA's software license agreements:

- [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
- [Product-Specific Terms for AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/)

By using NVIDIA NIM models, you acknowledge and agree to comply with these terms.

## Contributions, Bugs and Issues

We welcome anyone to help us improve these blueprints.
For bug reports or feature requests, please open an issue in this repository.

---
