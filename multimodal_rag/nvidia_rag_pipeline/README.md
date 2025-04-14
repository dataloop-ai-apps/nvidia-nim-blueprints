# NVIDIA RAG Pipeline

## Overview

The NVIDIA RAG pipeline is a blueprint designed to enhance the Retrieval-Augmented Generation (RAG) process. It efficiently interacts with models using Dataloop's AI playground and leveraging NIM models for generating embeddings and responses. The pipeline incorporates a retriever service to fetch relevant documents from the source dataset. The model's responses are stored in the prompts dataset and are subsequently sent to a labeling task, enabling human-in-the-loop validation and refinement.

For more details, visit the NVIDIA blueprint page: [Build an Enterprise RAG pipeline](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline).
And look for: `Retrival Pipeline`.

## Prerequisites

- Run the `Extraction Pipeline` in [Preprocessing Multimodal PDF RAG Blueprint](../preprocessing_multimodal_rag/README.md),
  and use the Final Embeddings Dataset and Model as inputs for this pipeline **Retrival Dataset** and **Retrival Embedding Model**.
- **Prompts Dataset** that contains the prompts to be sent for the LLM model.

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

2. **Set up the Pipeline**: Choose the appropriate **Retrival Dataset** and **Embedding Model** for the **Retriever Prompt** Service,
   and choose the **Prompts Dataset** that will store prompts for the LLM model.

3. **Run the Pipeline**: Make sure you have inserted your nim api key to the models services and run the pipeline.

## Contributing

We welcome contributions! Please see our [contributing guidelines] for more information on how to get involved.

