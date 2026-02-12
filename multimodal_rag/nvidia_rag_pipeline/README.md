# NVIDIA RAG Pipeline

## Quick setup

1. Run the [Preprocessing Multimodal PDF RAG](../preprocessing_multimodal_rag/README.md) pipeline first so you have a dataset of embedded chunks.
2. Install this pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
3. Add your **NVIDIA NGC API Key** in [Data Governance](https://docs.dataloop.ai/docs/overview-1).
4. Set pipeline variables (see [Variables and model IDs](#variables-and-model-ids) below): **retrieval_dataset**, **retrieval_embed_model** (must match preprocessing), **embed_model**, **gen_ai_model**, and optionally **k_nearest_items**.
5. Create prompt items with your questions and run the pipeline to get RAG answers.

For configuration, components, and troubleshooting, see the sections below.

### Variables and model IDs

| Variable | Type | Recommended / value | Purpose |
|----------|------|--------------------|---------|
| **retrieval_dataset** | Dataset | Output dataset from [preprocessing](../preprocessing_multimodal_rag/README.md) | Dataset of embedded chunks to search |
| **retrieval_embed_model** | Model | NVIDIA NV-EmbedQA E5 v5 (same as preprocessing) | Must match the embedding model used in preprocessing |
| **embed_model** | Model | NVIDIA NV-EmbedQA E5 v5 | Embeds user questions for retrieval |
| **gen_ai_model** | Model | Llama 3.1 405B Instruct | Generates the final RAG response |
| **k_nearest_items** | Integer | 30 (default) | Number of chunks to retrieve |

**Getting the model ID:** When you run or edit the pipeline, each Model variable shows a model selector. Choose the recommended model (or another from your project); the selected value is the model ID (e.g. `nim-nv-embedqa-e5-v5.models.nim-nv-embedqa-e5-v5`). You can also find model IDs in your project under **Develop** → **AI Library** (or **Models**). For **retrieval_embed_model**, use the exact same model ID as in the preprocessing pipeline so vector spaces match.

---

## Overview

The NVIDIA RAG Pipeline is the second stage of the two-stage RAG system. It accepts user queries, retrieves relevant document chunks using vector similarity search, and generates responses using an LLM with retrieved context.

This pipeline integrates with Dataloop's AI Playground and uses NIM models for embedding queries and generating responses. It includes human-in-the-loop validation by storing responses in a prompts dataset for review.

For more details, visit the NVIDIA blueprint page: [Build an Enterprise RAG pipeline](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline) and look for: `Retrieval Pipeline`.

## Prerequisites

- **NVIDIA NGC API Key**: Required for all NIM model services
- **Completed Extraction Pipeline**: Run the [Preprocessing Multimodal PDF RAG Blueprint](../preprocessing_multimodal_rag/README.md) first
- **Retrieval Dataset**: The output dataset from the extraction pipeline containing embedded chunks
- **Prompts Dataset**: A dataset to store user queries and generated responses

> **Critical**: The `retrieval_embed_model` must be the same model used in the extraction pipeline (`nv-embedqa-e5v5.models.nv-embedqa-e5v5`). Using a different embedding model will cause semantic search failures because the vector spaces won't align.

## Features

- **Query Embedding**: Converts user questions into vectors using the same model as document embeddings
- **Vector Similarity Search**: Retrieves the K nearest document chunks to the query
- **Context-Aware Generation**: Uses retrieved chunks as context for the LLM response
- **Human-in-the-Loop**: Stores responses for validation and refinement

## Components

### NIM Models

| Model | Purpose |
|-------|---------|
| **nv-embedqa-e5v5** | Embeds user queries for similarity search |
| **llama-3.1-405b-instruct** | Generates responses using retrieved context |

### Pipeline Nodes

#### 1. Dataset (Storage Node)
- **Function**: `clone_item`
- **Service**: pipeline-utils
- **Purpose**: Receives prompt items from the prompts dataset

#### 2. nv-embedqa-e5v5 (Embedding Model)
- **Type**: ML
- **Function**: `embed`
- **Purpose**: Generates embeddings for the user query
- **Output**: Vector representation of the query

#### 3. Retriever Prompt
- **Type**: Custom
- **Function**: `query_nearest_items_prompt`
- **Service**: retriever-service
- **Package**: llm-tools-retriever
- **Purpose**: Finds K nearest document chunks using vector similarity
- **Inputs**:
  - `item`: The prompt item
  - `embeddings`: Query embedding from previous node
  - `embedder`: Retrieval embedding model (must match extraction)
  - `dataset`: Retrieval dataset (from extraction pipeline)
  - `k`: Number of nearest items to retrieve

#### 4. llama-3.1-405b-instruct (Response Generation)
- **Type**: ML
- **Function**: `predict`
- **Purpose**: Generates a response using retrieved context
- **Output**: Item with response annotations

## Configuration Variables

| Variable | Type | Description |
|----------|------|-------------|
| `embed_model` | Model | Embedding model for user queries |
| `gen_ai_model` | Model | LLM for response generation |
| `retrieval_dataset` | Dataset | Dataset containing embedded chunks (from extraction pipeline) |
| `retrieval_embed_model` | Model | Embedding model used for retrieval dataset (must match extraction) |
| `k_nearest_items` | Integer | Number of chunks to retrieve (default: 30) |

## Usage

### 1. Install the Blueprint

Install the pipeline from the Dataloop Marketplace.

### 2. Configure the Retrieval Dataset

Set the `retrieval_dataset` variable to the ID of the dataset output by the extraction pipeline.

### 3. Configure the Embedding Model

Set the `retrieval_embed_model` to `nv-embedqa-e5v5.models.nv-embedqa-e5v5` (must match the extraction pipeline).

### 4. Configure Model Services

Ensure your NVIDIA NGC API key is configured in the model services:
- nv-embedqa-e5v5
- llama-3.1-405b-instruct

### 5. Create Prompt Items

Create prompt items in your prompts dataset containing user questions.

### 6. Run the Pipeline

Execute the pipeline. Each prompt item will be processed, retrieving relevant chunks and generating a response.

## Troubleshooting

### No Relevant Results Retrieved

- Verify `retrieval_dataset` points to the correct embedded chunks dataset
- Confirm `retrieval_embed_model` matches the model used during extraction
- Check that the retrieval dataset contains items with embedding annotations

### Poor Response Quality

- Increase `k_nearest_items` to provide more context
- Verify the extraction pipeline processed documents correctly
- Check that prompt items contain clear, well-formed questions

### Pipeline Execution Fails

- Verify NVIDIA NGC API key is valid and configured
- Check that all required services are running
- Ensure the retrieval dataset is accessible

### Embedding Model Mismatch Error

The retrieval embedding model **must** be the same as the one used in the extraction pipeline. If you used a different model during extraction, you'll need to re-run the extraction pipeline with the correct model, or update this pipeline to use the matching model.

## Contributing

We welcome contributions! Please see our contributing guidelines for more information on how to get involved.
