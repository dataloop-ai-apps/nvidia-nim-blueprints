# Video Search & Summarization Blueprint

This blueprint consists of two pipelines that work together:

1. **Preprocessing Pipeline** — ingests videos, generates descriptions and transcripts, embeds them, and builds a knowledge graph.
2. **Retrieval Pipeline** — queries the knowledge graph and vector store to answer questions about the processed videos.

---

## 1. Preprocessing Pipeline

### Quick setup

1. Install the pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
2. Add your **NVIDIA NGC API Key** in [Data Governance](https://docs.dataloop.ai/docs/overview-1).
3. Create a pipeline from the **Video Search and Summarization NVIDIA Blueprint** template.
4. Set the **source dataset** (Dataset node) and **target dataset** (Clone to Dataset node).
5. Optionally update model variables in the pipeline Variables panel.
6. Upload videos to your source dataset and execute the pipeline.

### Variables

| Variable | Type | Default model | Purpose |
|----------|------|---------------|---------|
| **vlm_model** | Model | VILA 1.5 3B (vila-model-adapter) | Vision Language Model for describing video segments |
| **embedding_model** | Model | Llama 3.2 NeMoRetriever 1B VLM Embed v1 | Text embeddings for vector search |
| **graph_llm_model** | Model | Llama 3.1 8B Instruct (guided JSON) | Entity and relationship extraction for the knowledge graph |

### Overview

The preprocessing pipeline processes video content for semantic search, Q&A, and knowledge-graph-based retrieval. It is 100% composed from external Dataloop DPKs — no custom node code lives in this repository.

It splits videos into time-based chunks and runs two parallel branches to extract information:

- **Visual branch**: Wraps each sub-video in a prompt, describes it with VILA 1.5 3B (a compact Vision Language Model optimized for image and video understanding), and extracts the text response.
- **Audio branch**: Extracts audio from each sub-video and transcribes it with NVIDIA Parakeet CTC 0.6B ASR.

Both branches merge at a Clone to Dataset node, which then fans out to:

- **Embedding**: Embeds the text with Llama 3.2 NeMoRetriever 1B VLM Embed v1 for downstream vector search.
- **Graph RAG**: Converts text to a prompt, extracts entities and relationships using Llama 3.1 8B Instruct with guided JSON, and stores them in a knowledge graph.

### Pipeline flow

```
                              ┌─ Video to Prompt → VILA VLM → Prompt to Text ─┐
Dataset → Video to Videos ────┤                                               ├──→ Clone to Dataset ─┬─→ Embedding
                              └─ Audio Extract → Parakeet ASR ────────────────┘                      │
                                                                                                     └─→ Text to Prompt → Graph Entity Extraction → Add Chunk to Graph
```

### Pipeline nodes

| # | Node | Type | DPK | Purpose |
|---|------|------|-----|---------|
| 1 | Dataset | storage | pipeline-utils | Source dataset — entry point for video items |
| 2 | Video to Videos | custom | video-utils-splitting | FFmpeg stream-copy splitting into 15 s chunks |
| 3 | Video to Prompt | custom | llm-tools-frames-to-prompt | Wraps a video item in a PromptItem for VLM inference |
| 4 | VILA VLM | ml | vila-model-adapter | VILA 1.5 3B vision-language inference on video |
| 5 | Prompt to Text | custom | prompt_to_text | Extracts the assistant response as a text item |
| 6 | Audio Extract | custom | audio-utils | FFmpeg audio extraction to WAV (16 kHz) |
| 7 | Parakeet ASR | custom | parakeet-ctc-0-6b-asr | NVIDIA Parakeet CTC 0.6B speech-to-text |
| 8 | Clone to Dataset | storage | pipeline-utils | Clones visual and audio text items into target dataset |
| 9 | Embedding | ml | nim-llama-3-2-nemoretriever-1b-vlm-embed-v1 | Text embeddings for vector search |
| 10 | Text to Prompt | custom | txt_to_prompt | Wraps text items in a PromptItem for LLM inference |
| 11 | Graph Entity Extraction | ml | nim-llama-3-1-8b-instruct | Extracts entities and relationships using guided JSON |
| 12 | Add Chunk to Graph | custom | graph-rag | Stores extracted entities/relationships in the knowledge graph |

### Target dataset structure

| Path | Content |
|------|---------|
| `/` | Text items (VLM descriptions + ASR transcripts) — used by embedding and retrieval |
| `/graph_rag/prompts/` | Intermediate prompt items for graph entity extraction |
| `/graph_rag/knowledge_graph.json` | The knowledge graph data (NetworkX) |
| `/graph_rag/knowledge_graph.png` | Auto-generated graph visualization |

---

## 2. Retrieval Pipeline

### Quick setup

1. Install the retrieval pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
2. Create a pipeline from the **VSS Retrieval NVIDIA Blueprint** template.
3. Set **retrieval_dataset** to the target dataset from the preprocessing pipeline.
4. Optionally adjust model variables and **k_nearest_items**.
5. Execute the pipeline with prompt items containing your questions.

### Variables

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| **embed_model** | Model | Llama 3.2 NeMoRetriever 1B VLM Embed v1 | Query embedding (must match the preprocessing embedding model) |
| **gen_ai_model** | Model | Llama 3.1 8B Instruct | Generates the final answer from retrieved context |
| **retrieval_dataset** | Dataset | — (user must set) | The chunked dataset with embeddings and knowledge graph |
| **k_nearest_items** | Integer | 30 | Number of nearest items to retrieve via vector search |

### Overview

The retrieval pipeline answers questions about processed videos by combining two retrieval strategies:

1. **Vector search**: Embeds the user query and retrieves the k nearest text chunks from the dataset.
2. **Graph RAG**: Searches the knowledge graph for matching entities and relationships, adding structured context to the prompt.

The Response LLM then generates an answer using both vector-retrieved passages and graph context.

### Pipeline flow

```
Prompt → Embedding → Retriever → Graph Query → Response LLM
```

### Pipeline nodes

| # | Node | Type | DPK | Purpose |
|---|------|------|-----|---------|
| 1 | Embedding Model | ml | nim-llama-3-2-nemoretriever-1b-vlm-embed-v1 | Embeds the user query |
| 2 | Retriever Prompt | custom | llm-tools-retriever | Finds k nearest embedded chunks via vector search |
| 3 | Graph Query | custom | graph-rag | Searches the knowledge graph and adds graph context to the prompt |
| 4 | Response LLM | ml | nim-llama-3-1-8b-instruct | Generates the final answer from all retrieved context |

---

## Requirements

- NVIDIA NGC API Key

## Contributing

We welcome contributions! Please submit bug reports or feature requests through the appropriate channels.
