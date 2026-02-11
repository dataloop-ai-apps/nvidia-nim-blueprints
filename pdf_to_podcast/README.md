# PDF to Podcast Blueprint

## Quick setup

1. Install the pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
2. Add your **NVIDIA NGC API Key** and **ElevenLabs API Key** in [Data Governance](https://docs.dataloop.ai/docs/overview-1).
3. Set pipeline variables: **reasoning LLM**, **iterative LLM**, **json podcast LLM**, **json convo LLM** (see [Variables and model IDs](#variables-and-model-ids) below).
4. Configure format (monologue or dialogue), duration, and voice in the pipeline settings.
5. Upload a PDF and run the pipeline to generate podcast audio.

For components, usage details, and requirements, see the sections below.

### Variables and model IDs

| Variable | Type | Recommended model | Purpose |
|----------|------|-------------------|---------|
| **reasoning LLM** | Model | Llama 3.1 405B Instruct | Summarization, outline, and main content |
| **iterative LLM** | Model | Llama 3.1 70B Instruct | Combining dialogue segments |
| **json podcast LLM** | Model | Llama 3.1 8B Instruct | Structured podcast outline (JSON) |
| **json convo LLM** | Model | Llama 3.1 8B Instruct | Dialogue/monologue to JSON |

**Getting the model ID:** When you run or edit the pipeline, each Model variable shows a model selector. Choose the recommended model (or another from your project); the selected value is the model ID (e.g. `nim-llama-3-1-405b-instruct.models.nim-llama-3-1-405b-instruct`). You can also find model IDs in your project under **Develop** → **AI Library** (or **Models**).

---

## Overview

The PDF to Podcast blueprint is designed to transform PDF documents into audio content using advanced text-to-speech technology. This pipeline automates the process of extracting text from PDFs, processing it into natural-sounding segments, and converting them into high-quality audio files, making written content more accessible in audio format.

## Features

- **PDF Text Extraction**: Efficiently extracts text content from PDF documents.
- **Text Processing**: Processes extracted text into natural, speech-friendly segments.
- **Text-to-Speech Conversion**: Converts processed text into high-quality audio using advanced TTS models.
- **Audio File Generation**: Creates podcast-ready audio files from the converted text.
- **Flexible Output Options**: Supports various audio formats and quality settings.

## Components

### 1. PDF Processing

- **Description**: Handles PDF document parsing and text extraction.

### 2. Podcast Transcript Generation

- **Description**: Converts extracted text into a transcript with natural speech.

### 3. Text-to-Speech Service

- **Description**: Converts text segments into natural-sounding audio with ElevenLabs.

## Usage

1. **Install the Blueprint**: Install the pipeline from Dataloop Marketplace.

2. **Configure Settings**: Set up your preferred:
   - Conversation format (1 or 2 podcast hosts)
   - Guided instructions to help focus model responses
   - Duration of podcast (in minutes)
   - Voice selection

3. **Run the Pipeline**: Upload your PDF document and run the pipeline to generate the audio output.

## Requirements

- NVIDIA NGC API Key
- ElevenLabs API Key
