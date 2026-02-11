# Preprocessing Multimodal PDF RAG Blueprint

## Overview

The Preprocessing Multimodal PDF RAG pipeline is the first stage of the two-stage RAG system. It extracts content from PDF documents—including text and embedded images—applies OCR, chunks the text, and generates vector embeddings for semantic search.

This pipeline uses the [RAG PDF Processor](https://github.com/dataloop-ai-apps/rag-multimodal-processors) app to handle text extraction, OCR, and chunking in a single node, followed by NVIDIA NIM embedding generation.

For more details, visit the NVIDIA blueprint page: [Build an Enterprise RAG pipeline](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline) and look for: `Extraction Pipeline`.

## Pipeline Architecture

```
Dataset (Input) → PDF to Chunks → Dataset (Chunks) → nv-embedqa-e5v5 → (Embedded Items)
```

The pipeline consists of 4 nodes:

1. **Dataset (Input)** — Source dataset containing PDF documents
2. **PDF to Chunks** — Extracts text, applies OCR on embedded images, and splits into chunks
3. **Dataset (Chunks)** — Intermediate dataset storing the generated text chunks
4. **nv-embedqa-e5v5** — Generates vector embeddings for each text chunk

## Prerequisites

- **NVIDIA NGC API Key**: Required for the NIM embedding model service
- **Source Dataset**: A Dataloop dataset containing PDF documents to process
- **Chunk Dataset**: A Dataloop dataset where text chunks will be stored (intermediate)
- **Target Dataset**: The chunk dataset with embeddings, used as input for the retrieval pipeline

> **Important**: This pipeline produces outputs required by the [NVIDIA RAG Pipeline](../nvidia_rag_pipeline/README.md). Run this extraction pipeline first, then use the output dataset and embedding model as inputs for the retrieval pipeline.

## Features

- **PDF Text Extraction**: Extracts text using PyMuPDF with optional markdown-aware extraction (`pymupdf4llm`)
- **OCR on Embedded Images**: Extracts text from images embedded in PDFs using EasyOCR
- **Text Cleaning**: Optional text normalization and deep cleaning
- **Flexible Chunking**: Multiple chunking strategies (recursive, fixed-size, sentence, paragraph, single chunk)
- **Embedding Generation**: Creates 1024-dimensional vector embeddings using nv-embedqa-e5v5

## Components

### Apps & Models

| Component | Purpose |
|-----------|---------|
| **RAG PDF Processor** | Extracts text from PDFs, applies OCR, and creates text chunks |
| **nv-embedqa-e5v5** | Generates vector embeddings for text chunks (NVIDIA NIM) |

### Pipeline Nodes

#### 1. Dataset (Input — Storage Node)
- **Function**: `clone_item`
- **Service**: pipeline-utils
- **Purpose**: Manages item flow from source dataset

#### 2. PDF to Chunks
- **Type**: Custom
- **Function**: `run`
- **Service**: pdf-processor-service (from `rag-pdf-processor` app)
- **Purpose**: Processes PDF files into text chunks with optional OCR
- **Default Config**:
  - `use_markdown_extraction`: true — preserves document structure
  - `ocr_from_images`: true — extracts text from embedded images
  - `ocr_integration_method`: append_to_page — appends OCR text after each page
  - `chunking_strategy`: recursive — respects semantic boundaries
  - `max_chunk_size`: 300
  - `chunk_overlap`: 40
  - `to_correct_spelling`: false

#### 3. Dataset (Chunks — Storage Node)
- **Function**: `clone_item`
- **Service**: pipeline-utils
- **Purpose**: Stores generated text chunks in an intermediate dataset

#### 4. nv-embedqa-e5v5
- **Type**: ML (Embeddings)
- **Function**: `embed`
- **Purpose**: Generates 1024-dimensional vector embeddings for each text chunk
- **Model**: `nvidia/nv-embedqa-e5-v5`

## Usage

### 1. Install the Blueprint

Install the pipeline from the Dataloop Marketplace.

### 2. Configure Datasets

- **Source Dataset**: Select the dataset containing your PDF documents
- **Chunk Dataset**: Select or create a dataset for intermediate text chunks
- **Embedding output**: Chunks in the intermediate dataset will be embedded in place

### 3. Configure Model Service

Ensure your NVIDIA NGC API key is configured in the embedding model service:
- nv-embedqa-e5v5

### 4. Configure PDF Processing (Optional)

The PDF to Chunks node supports the following configuration options:

| Option | Default | Description |
|--------|---------|-------------|
| `use_markdown_extraction` | `true` | Use ML-enhanced markdown extraction for better structure preservation |
| `ocr_from_images` | `true` | Extract text from embedded images using OCR |
| `ocr_integration_method` | `append_to_page` | How to integrate OCR text (`append_to_page`, `separate_chunks`, `combine_all`) |
| `chunking_strategy` | `recursive` | Chunking method (`recursive`, `fixed-size`, `nltk-sentence`, `nltk-paragraphs`, `1-chunk`) |
| `max_chunk_size` | `300` | Maximum characters per chunk |
| `chunk_overlap` | `40` | Overlapping characters between consecutive chunks |
| `to_correct_spelling` | `false` | Apply deep text cleaning and normalization |

### 5. Run the Pipeline

Execute the pipeline. Items from the source dataset will be processed, chunked, and embedded.

### 6. Next Steps

After completion, use the output dataset as the `retrieval_dataset` input for the [NVIDIA RAG Pipeline](../nvidia_rag_pipeline/README.md). The embedding model to use is `nv-embedqa-e5v5.models.nv-embedqa-e5v5`.

## Troubleshooting

### Pipeline Execution Fails

- Verify NVIDIA NGC API key is valid and configured in the embedding model service
- Check that source dataset contains valid PDF files
- Ensure sufficient compute resources are allocated to services

### Empty Output Dataset

- Verify PDFs contain extractable text (not scanned images without OCR)
- If PDFs contain scanned pages, enable `ocr_from_images` in the PDF to Chunks node
- Check pipeline logs for extraction errors

### Embedding Errors

- Confirm the nv-embedqa-e5v5 model service is running
- Verify text chunks are being generated (check the intermediate chunks dataset)

## Contributing

We welcome contributions! Please see our contributing guidelines for more information on how to get involved.
