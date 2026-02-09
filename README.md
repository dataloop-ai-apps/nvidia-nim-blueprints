# NVIDIA NIM Blueprints for Dataloop

---

A collection of NVIDIA NIM-powered blueprints for the Dataloop Platform. These blueprints leverage NVIDIA's AI models to enable advanced GenAI workflows including document processing, report generation, and multimodal RAG pipelines.

## Blueprints

| Blueprint | Description |
|-----------|-------------|
| [Report Generation](report_generation/README.md) | Automated report creation with web research and LLM-powered content generation |
| [PDF to Podcast](pdf_to_podcast/README.md) | Transform PDF documents into podcast-ready audio content |
| [Multimodal RAG - Preprocessing](multimodal_rag/preprocessing_multimodal_rag/README.md) | PDF extraction and embedding pipeline for RAG |
| [Multimodal RAG - Retrieval](multimodal_rag/nvidia_rag_pipeline/README.md) | Document retrieval and response generation with human-in-the-loop |

## Blueprint Overviews

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
2. Search for the desired blueprint (Report Generation, PDF to Podcast, or Multimodal RAG)
3. Install the pipeline to your project
4. Configure required API keys in your organization's [Data Governance](https://docs.dataloop.ai/docs/overview-1)

## Requirements

| Blueprint | Required API Keys |
|-----------|-------------------|
| Report Generation | NVIDIA NGC API Key, Tavily API Key |
| PDF to Podcast | NVIDIA NGC API Key, ElevenLabs API Key |
| Multimodal RAG | NVIDIA NGC API Key |

## NVIDIA Models Used

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
