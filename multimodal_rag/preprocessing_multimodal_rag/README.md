# Preprocessing Multimodal PDF RAG Blueprint

## Overview

The "Preprocessing Multimodal PDF RAG" is an NVIDIA blueprint designed to facilitate the preprocessing stage of a multimodal PDF Retrieval-Augmented Generation (RAG) pipeline. This pipeline is capable of handling both text and image data, making it suitable for a variety of applications that require the extraction and processing of information from PDF documents.

For more details, visit the NVIDIA blueprint page: [Build an Enterprise RAG pipeline](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline).
And look for: `Etraction Pipeline`.

## Prerequisites

- This blueprint is designed to be used in conjunction with `Retrival Pipeline` in [NVIDIA RAG Pipeline](../nvidia_rag_pipeline/README.md).

## Features

- **PDF to Text Conversion**: Converts PDF documents into text format for further processing.
- **Text Chunking**: Splits the extracted text into manageable chunks for efficient processing.
- **PDF to Image Conversion**: Converts PDF pages into images for visual data extraction.
- **Image to Prompt Conversion**: Transforms images into prompts for further analysis.
- **Crop Annotations**: Extracts and processes annotations from images.

## Components

### 1. NIM Models
- ***yolox-page-elements-v1***
- ***google-deplot***
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

### 4. Image to Prompt
- **Type**: Code
- **Function**: `run`
- **Description**: Converts images into prompts for further processing.

### 5. Crop Annotations
- **Type**: Function
- **Function**: `crop_annotations`
- **Description**: Processes and extracts annotations from images.

### 6. PDF to Image
- **Type**: Custom
- **Function**: `pdf_item_to_images`
- **Description**: Converts PDF documents into image format.

## Usage

1. **Install the Blueprint**: Install the pipeline from Dataloop Marketplace.

2. **Run the Pipeline**: Choose your dataset, insert your nim api key to the models services and run the pipeline.


## Contributions, Bugs and Issues

We welcome anyone to help us improve this app.  
[Here](CONTRIBUTING.md) are detailed instructions to help you report a bug or submit a feature request.
 