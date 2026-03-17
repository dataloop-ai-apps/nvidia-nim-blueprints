# Video Search & Summarization Blueprint

## Quick setup

1. Install the pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
2. Add your **NVIDIA NGC API Key** in [Data Governance](https://docs.dataloop.ai/docs/overview-1).
3. Create a pipeline from the **Video Search and Summarization NVIDIA Blueprint** template.
4. Set the **source dataset** (Dataset node) and **target dataset** (Clone to Dataset node).
5. Optionally update model variables in the pipeline Variables panel (see [Variables](#variables) below).
6. Upload videos to your source dataset and execute the pipeline.

### Variables

| Variable | Type | Default model | Purpose |
|----------|------|---------------|---------|
| **vlm_model** | Model | VILA 1.5 3B (vila-model-adapter) | Vision Language Model for describing video segments |
| **embedding_model** | Model | Llama 3.2 NeMoRetriever 1B VLM Embed v1 | Text embeddings for vector search |
| **graph_llm_model** | Model | Llama 3.1 8B Instruct (guided JSON) | Entity and relationship extraction for the knowledge graph |

---

## Overview

The Video Search & Summarization (VSS) pipeline processes video content for semantic search, Q&A, and knowledge-graph-based retrieval. The pipeline is 100% composed from external Dataloop DPKs — no custom node code lives in this repository.

It splits videos into time-based chunks and runs two parallel branches to extract information:

- **Visual branch**: Wraps each sub-video in a prompt, describes it with VILA 1.5 3B (a compact Vision Language Model optimized for image and video understanding), and extracts the text response.
- **Audio branch**: Extracts audio from each sub-video and transcribes it with NVIDIA Parakeet CTC 0.6B ASR.

Both branches merge at a Clone to Dataset node, which then fans out to:

- **Embedding**: Embeds the text with Llama 3.2 NeMoRetriever 1B VLM Embed v1 for downstream vector search.
- **Graph RAG**: Converts text to a prompt, extracts entities and relationships using Llama 3.1 8B Instruct with guided JSON, and stores them in a knowledge graph.

## Pipeline flow

```
                              ┌─ Video to Prompt → VILA VLM → Prompt to Text ─┐
Dataset → Video to Videos ────┤                                               ├──→ Clone to Dataset ─┬─→ Embedding
                              └─ Audio Extract → Parakeet ASR ────────────────┘                      │
                                                                                                     └─→ Text to Prompt → Graph Entity Extraction → Add Chunk to Graph
```
## Requirements

- NVIDIA NGC API Key

## Contributing

We welcome contributions! Please submit bug reports or feature requests through the appropriate channels.
