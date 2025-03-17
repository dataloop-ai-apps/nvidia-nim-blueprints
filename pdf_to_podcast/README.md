# PDF to Podcast Blueprint

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

## Contributing

We welcome contributions! Please see our [contributing guidelines] for more information on how to get involved.
