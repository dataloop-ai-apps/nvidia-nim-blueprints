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
| Video Search & Summarization | Ingest videos, split into chunks, describe with VILA VLM, transcribe audio with Parakeet ASR, embed, and store for search & Q&A |

## Blueprint Overviews

### Video Search & Summarization

Processes video content for semantic search and Q&A. The pipeline is 100% composed from external Dataloop DPKs — no custom node code lives in this repository. It splits videos into time-based chunks and runs two branches in parallel:

- **Visual branch**: Wraps each sub-video in a prompt → describes it with NVILA-8B (a Vision Language Model with native video understanding) → extracts the text response
- **Audio branch**: Extracts audio from each sub-video → transcribes with NVIDIA Parakeet CTC 0.6B ASR

Both branches clone results to a target dataset and embed them with Llama 3.2 NeMoRetriever 300M Embed v2 for downstream vector search.

**Pipeline flow:**

```
                        ┌─ Video to Prompt → VILA VLM → Prompt to Text ─┐
Video ─→ Video to Videos┤                                               ├─→ Clone to Dataset → Embedding
                        └─ Audio Extract → Parakeet ASR ────────────────┘
```

**Reused DPKs:**

| DPK | Node(s) |
|-----|---------|
| `video-utils-splitting` | Video to Videos — FFmpeg stream-copy splitting into 30 s chunks |
| `llm-tools-frames-to-prompt` | Video to Prompt — wraps a video item in a PromptItem |
| `vila-model-adapter` | VILA VLM — NVILA-8B vision-language inference on native video |
| `prompt_to_text` | Prompt to Text — extracts the assistant response as a text item |
| `audio-utils` | Audio Extract — FFmpeg audio extraction to WAV |
| `parakeet-ctc-0-6b-asr` | Parakeet ASR — NVIDIA Parakeet CTC 0.6B speech-to-text |
| `nim-llama-3-2-nemoretriever-300m-embed-v2` | Embedding — text embeddings for vector search |

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

1. **Preprocessing Pipeline**: Extracts text and images from PDFs, generates chunks, and creates embeddings using NVIDIA NIM models (YOLOX, PaddleOCR, E5-V5).

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

- **NVILA-8B** - Video description via native video understanding (VSS)
- **Parakeet CTC 0.6B ASR** - Speech-to-text transcription (VSS)
- **Llama 3.2 NeMoRetriever 300M Embed v2** - Text embeddings for vector search (VSS)
- **Llama 3.3 70B Instruct** - Report planning and content generation
- **Llama 3.1 405B Instruct** - RAG response generation, PDF to Podcast script generation
- **Llama 3.1 70B Instruct** - PDF to Podcast content processing
- **Llama 3.1 8B Instruct** - PDF to Podcast content processing
- **YOLOX Page Elements** - PDF layout analysis
- **PaddleOCR** - Optical character recognition
- **E5-V5 Embeddings** - Document embeddings for retrieval

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
