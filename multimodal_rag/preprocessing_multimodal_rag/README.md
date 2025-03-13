# Preprocessing Multimodal PDF RAG Blueprint

## Overview

The "Preprocessing Multimodal PDF RAG" is an NVIDIA blueprint designed to facilitate the preprocessing stage of a multimodal PDF Retrieval-Augmented Generation (RAG) pipeline. This pipeline is capable of handling both text and image data, making it suitable for a variety of applications that require the extraction and processing of information from PDF documents.

## Features

- **PDF to Text Conversion**: Converts PDF documents into text format for further processing.
- **Text Chunking**: Splits the extracted text into manageable chunks for efficient processing.
- **PDF to Image Conversion**: Converts PDF pages into images for visual data extraction.
- **Image to Prompt Conversion**: Transforms images into prompts for further analysis.
- **Crop Annotations**: Extracts and processes annotations from images.

## Components

### 1. Dataset
- **Type**: Storage
- **Function**: `clone_item`
- **Service Name**: pipeline-utils
- **Description**: Manages dataset storage and retrieval.

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

1. **Clone the Repository**: Clone the repository from the following URL:
   ```
   git clone https://github.com/dataloop-ai-apps/nvidia-nim-blueprints.git
   ```

2. **Set Up Environment**: Ensure you have the necessary dependencies installed. You can find them listed in the `dependencies` section of the `dataloop.json` file.

3. **Run the Pipeline**: Follow the instructions in the repository's README to execute the preprocessing pipeline on your multimodal PDF data.

## License

This project is licensed under the terms of the [applicable license]. Please refer to the LICENSE file for more details.

## Contributing

We welcome contributions! Please see our [contributing guidelines] for more information on how to get involved.

## Support

For any questions or issues, please contact [support contact information]. 