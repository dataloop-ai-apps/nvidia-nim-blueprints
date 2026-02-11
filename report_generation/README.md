# Report Generation Blueprint

## Quick setup

1. Install the pipeline from the [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace).
2. Add your **NVIDIA NGC API Key** and **Tavily API Key** in [Data Governance](https://docs.dataloop.ai/docs/overview-1).
3. Create a prompt with your topic and optional structure (see [Usage](#usage) below for format).
4. Run the pipeline; it will research and generate the report.

For details, requirements, and troubleshooting, see the sections below.

---

## Overview

The "Report Generation" is an NVIDIA blueprint designed to automate the creation of comprehensive reports on any given topic. This pipeline leverages advanced language models and web search capabilities to research, plan, and generate well-structured reports with minimal user input.

## Features

- **Automated Report Planning**: Intelligently plans report sections based on the provided topic and structure.
- **Web Research Integration**: Performs targeted web searches to gather relevant information for each section.
- **Section-Based Generation**: Creates specialized content for each section of the report.
- **Flexible Report Structure**: Supports customizable report structures to fit various needs.
- **News and General Research**: Can focus on either general information or recent news articles.

## Components

### 1. Pipeline Nodes
- **Report Planning Node**: Plans the overall structure of the report.
- **Search Tavily Node**: Performs web searches using the Tavily API.
- **Report Sections Node**: Organizes the report into logical sections.
- **Research Agents Node**: Conducts in-depth research for each section.
- **Report Writing Node**: Compiles the final report from all researched sections.

### 2. Models and APIs
- **NVIDIA NIM API**: Uses Llama 3.3 70B Instruct model for high-quality text generation.
- **Tavily API**: Provides web search capabilities for research.

## Usage

1. **Install the Blueprint**: Install the pipeline from Dataloop Marketplace.

2. **Configure the Report**: Create a prompt with the following format:
   ```
   Topic: [Your report topic]
   Structure: [Desired report structure]
   Number of queries: [Number of web searches per section, default: 2]
   Tavily Topic: [general or news, default: general]
   Tavily Days: [Number of days for news search, optional]
   ```

3. **Run the Pipeline**: The pipeline will automatically research, plan, and generate a comprehensive report on your topic.

## Requirements

- NVIDIA NGC API Key
- Tavily API Key

## Contributions, Bugs and Issues

We welcome anyone to help us improve this blueprint.  
Please submit bug reports or feature requests through the appropriate channels.