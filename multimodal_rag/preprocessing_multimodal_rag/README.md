# Preprocessing Multimodal PDF RAG Blueprint

## Quick setup

1. Install the pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
2. Add your **NVIDIA NGC API Key** in [Data Governance](https://docs.dataloop.ai/docs/overview-1).
3. Set **source** and **target** datasets, and the four model variables (see [Variables and model IDs](#variables-and-model-ids) below).
4. Run the pipeline. When it finishes, use the target dataset and embedding model as inputs for the [NVIDIA RAG Pipeline](../nvidia_rag_pipeline/README.md) (retrieval stage).

For prerequisites, components, and troubleshooting, see the sections below.

### Variables and model IDs

| Variable | Type | Recommended model | Purpose |
|----------|------|-------------------|---------|
| **nv-yolox-page-elements-v1** | Model | NVIDIA YOLOX Page Elements v1 | Page element detection (tables, figures, text blocks) |
| **university-at-buffalo-cached** | Model | University at Buffalo CachedTransformer | Table detection and extraction |
| **baidu-paddleocr** | Model | Baidu PaddleOCR | OCR on chart/images |
| **nv-embedqa-e5v5** | Model | NVIDIA NV-EmbedQA E5 v5 | Vector embeddings for chunks (must match RAG retrieval) |

**Getting the model ID:** When you run or edit the pipeline, each Model variable shows a model selector. Choose the recommended model (or another from your project); the selected value is the model ID (e.g. `nv-embedqa-e5v5.models.nv-embedqa-e5v5`). You can also find model IDs in your project under **Develop** → **AI Library** (or **Models**). Dataset variables: pick the source dataset (PDFs) and target dataset (for embedded chunks) from the dataset selector.

---

## Overview

The Preprocessing Multimodal PDF RAG pipeline is the first stage of the two-stage RAG system. It extracts content from PDF documents—including text, tables, and charts—and generates vector embeddings for semantic search.

This pipeline handles multimodal content, using specialized NIM models to detect and process different page elements (text blocks, tables, charts) before embedding them for retrieval.

For more details, visit the NVIDIA blueprint page: [Build an Enterprise RAG pipeline](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline) and look for: `Extraction Pipeline`.

## Prerequisites

- **NVIDIA NGC API Key**: Required for all NIM model services
- **Source Dataset**: A Dataloop dataset containing PDF documents to process
- **Target Dataset**: A Dataloop dataset where embedded chunks will be stored

> **Important**: This pipeline produces outputs required by the [NVIDIA RAG Pipeline](../nvidia_rag_pipeline/README.md). Run this extraction pipeline first, then use the output dataset and embedding model as inputs for the retrieval pipeline.

## Features

- **PDF to Text Conversion**: Extracts text content from PDF documents
- **PDF to Image Conversion**: Converts PDF pages to images for visual element processing
- **Page Element Detection**: Uses YOLOX to identify tables, charts, and text blocks
- **Table Extraction**: Processes tables using the CACHED model
- **Chart OCR**: Extracts text from charts using PaddleOCR
- **Text Chunking**: Splits extracted text into configurable chunks with overlap
- **Embedding Generation**: Creates vector embeddings using nv-embedqa-e5v5

## Components

### NIM Models

| Model | Purpose |
|-------|---------|
| **nv-yolox-page-elements-v1** | Detects page elements (tables, charts, text blocks) |
| **university-at-buffalo-cached** | Extracts structured data from tables |
| **baidu-paddleocr** | Performs OCR on chart images |
| **nv-embedqa-e5v5** | Generates vector embeddings for text chunks |

### Pipeline Nodes

#### 1. Dataset (Storage Node)
- **Function**: `clone_item`
- **Service**: pipeline-utils
- **Purpose**: Manages item flow from source dataset

#### 2. PDF to Txt
- **Type**: Custom
- **Function**: `pdf_extraction`
- **Service**: pdf-to-txt-v2
- **Purpose**: Converts PDF files into text format
- **Config**:
  - `remote_path_for_extractions`: /extracted_from_pdfs
  - `extract_images`: false

#### 3. PDF to Image
- **Type**: Custom
- **Function**: `pdf_item_to_images`
- **Purpose**: Converts PDF pages into images for visual processing

#### 4. nv-yolox-page-elements-v1
- **Type**: ML
- **Function**: `predict`
- **Purpose**: Detects page elements and routes items based on element type

#### 5. baidu-paddleocr
- **Type**: ML
- **Function**: `predict`
- **Purpose**: Performs OCR on chart images (filtered by `metadata.user.originalAnnotationLabel = "chart"`)

#### 6. university-at-buffalo-cached
- **Type**: ML
- **Function**: `predict`
- **Purpose**: Extracts structured data from detected tables

#### 7. Post-Processing (Code Node)
- **Type**: Code
- **Function**: `post_processing`
- **Purpose**: Aggregates OCR text and cached responses into text files

#### 8. Text to Chunks
- **Type**: Custom
- **Function**: `create_chunks`
- **Purpose**: Splits text into chunks with configurable size and overlap

#### 9. nv-embedqa-e5v5
- **Type**: ML
- **Function**: `embed`
- **Purpose**: Generates vector embeddings for each text chunk

## Usage

### 1. Install the Blueprint

Install the pipeline from the Dataloop Marketplace.

### 2. Configure Datasets

- **Source Dataset**: Select the dataset containing your PDF documents
- **Target Dataset**: Select or create a dataset to store the embedded chunks

### 3. Configure Model Services

Ensure your NVIDIA NGC API key is configured in the model services:
- nv-yolox-page-elements-v1
- baidu-paddleocr
- university-at-buffalo-cached
- nv-embedqa-e5v5

### 4. Run the Pipeline

Execute the pipeline. Items from the source dataset will be processed, and embedded chunks will be stored in the target dataset.

### 5. Next Steps

After completion, use the output dataset as the `retrieval_dataset` input for the [NVIDIA RAG Pipeline](../nvidia_rag_pipeline/README.md). The embedding model to use is `nv-embedqa-e5v5.models.nv-embedqa-e5v5`.

## Troubleshooting

### Pipeline Execution Fails

- Verify NVIDIA NGC API key is valid and configured in all model services
- Check that source dataset contains valid PDF files
- Ensure sufficient compute resources are allocated to services

### Empty Output Dataset

- Verify PDFs contain extractable text (not scanned images without OCR)
- Check pipeline logs for extraction errors

### Embedding Errors

- Confirm the nv-embedqa-e5v5 model service is running
- Verify text chunks are being generated (check intermediate outputs)

## Contributing

We welcome contributions! Please see our contributing guidelines for more information on how to get involved.
