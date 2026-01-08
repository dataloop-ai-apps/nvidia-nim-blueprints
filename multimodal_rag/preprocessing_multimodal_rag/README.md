# Preprocessing Multimodal PDF RAG Blueprint

## Overview

The "Preprocessing Multimodal PDF RAG" is an NVIDIA blueprint designed to facilitate the preprocessing stage of a multimodal PDF Retrieval-Augmented Generation (RAG) pipeline. This pipeline is capable of handling both text and image data, making it suitable for a variety of applications that require the extraction and processing of information from PDF documents.

For more details, visit the NVIDIA blueprint page: [Build an Enterprise RAG pipeline](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline).
And look for: `Etraction Pipeline`.

## Prerequisites

- This blueprint is designed to be used in conjunction with `Retrival Pipeline` in [NVIDIA RAG Pipeline](../nvidia_rag_pipeline/README.md). \
  Please run this `Extraction Pipeline` first and use the final embeddings Dataset and Model as inputs for the 
  `Retrival Pipeline` **Retrival Dataset** and **Retrival Embedding Model**.

## Features

- **PDF to Text Conversion**: Converts PDF documents into text format for further processing.
- **Text Chunking**: Splits the extracted text into manageable chunks for efficient processing.
- **PDF to Image Conversion**: Converts PDF pages into images for visual data extraction.
- **Crop Annotations**: Extracts and processes annotations from images.

## Components

### 1. NIM Models
- ***yolox-page-elements-v1***
- ***university-at-buffalo-cached***
- ***baidu-paddleocr***
- ***nvidia-embedqa-e5v5***

### 2. PDF to Txt
- **Type**: Custom
- **Function**: `pdf_extraction`
- **Service**: `pdf-to-txt-v2`
- **Description**: Converts PDF files into text format, with options for image extraction and remote path configuration.

### 3. Text to Chunks
- **Type**: Custom
- **Function**: `create_chunks`
- **Description**: Generates text chunks from the extracted text with configurable chunk size and overlap.

### 4. Crop Annotations
- **Type**: Function
- **Function**: `crop_annotations`
- **Description**: Processes and extracts annotations from images.

### 5. PDF to Image
- **Type**: Custom
- **Function**: `pdf_item_to_images`
- **Description**: Converts PDF documents into image format.

## Usage

1. **Install the Blueprint**: Install the pipeline from Dataloop Marketplace.

2. **Set up the Pipeline**: Choose the 2 datasets:
   - **Source Dataset**: The dataset containing the PDF documents to be processed.
   - **Target Dataset**: The dataset containing the extracted chunks with the embeddings.

3. **Run the Pipeline**: Make sure you have inserted your nim api key to the models services and run the pipeline.


## Contributing

We welcome contributions! Please see our [contributing guidelines] for more information on how to get involved.
