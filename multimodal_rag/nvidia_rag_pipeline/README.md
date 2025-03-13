# NVIDIA RAG Pipeline

## Overview

The NVIDIA RAG pipeline is a blueprint designed to enhance the Retrieval-Augmented Generation (RAG) process. It efficiently interacts with models using Dataloop's AI playground and leveraging NIM models for generating embeddings and responses. The pipeline incorporates a retriever service to fetch relevant documents from the source dataset. The model's responses are stored in the prompts dataset and are subsequently sent to a labeling task, enabling human-in-the-loop validation and refinement.

For more details, visit the [NVIDIA blueprint page](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag/blueprintcard).

## Features

- **Efficient Model Interaction**: Utilizes Dataloop's AI Playground for seamless model communication.
- **Embedding Generation**: Leverages NIM models to generate embeddings and responses.
- **Document Retrieval**: Incorporates a retriever service to fetch relevant documents.
- **Human-in-the-Loop**: Stores model responses in a prompt dataset and sends them to a labeling task for validation.

## Components

### 1. Dataset
- **Type**: Storage
- **Function**: `dataset_handler`
- **Service Name**: pipeline-utils
- **Description**: Manages dataset storage and retrieval.

### 2. Embedding Model
- **Type**: ML
- **Function**: `embed`
- **Package**: nv-embedqa-e5v5
- **Description**: Generates embeddings for input data.

### 3. Retriever Service
- **Type**: Custom
- **Function**: `query_nearest_items_prompt`
- **Package**: llm-tools-retriever
- **Description**: Retrieves relevant documents based on embeddings.

### 4. Response Generation
- **Type**: ML
- **Function**: `predict`
- **Package**: nim-api-llama3-1-405b-instruct-meta
- **Description**: Generates responses using the Llama model.

### 5. Labeling Task
- **Type**: Task
- **Function**: `move_to_task`
- **Service Name**: pipeline-utils
- **Description**: Sends responses to a labeling task for human validation.

## Usage

1. **Install the Blueprint**: Install the pipeline from Dataloop Marketplace.

2. **Run the Pipeline**: Choose your dataset, insert your nim api key to the models services and run the pipeline.

## Contributing

We welcome contributions! Please see our [contributing guidelines] for more information on how to get involved.

