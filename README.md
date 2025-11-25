# Document Processing & Contextual Chunking Pipeline

An intelligent document processing system that extracts content from various file formats, generates semantic chunks with contextual summaries, and handles images using AI vision capabilities.

## Overview

This pipeline processes documents through two main stages:

1. **Document Processing** (`process_documents.py`): Extracts structured content from documents using Unstructured.io API
2. **Contextual Chunking** (`contextual_chunking.py`): Creates semantically meaningful chunks with AI-generated context using Azure OpenAI embeddings and AWS Bedrock Claude

## Features

### Document Processing
- Multi-format support: PDF, PPTX, DOCX, HTML, TXT, MD, CSV
- High-resolution content extraction with layout detection
- Table structure inference with HTML preservation
- Image extraction as base64
- Parallel processing for speed

### Contextual Chunking
- **Semantic chunking** using text-embedding-3-large (Azure OpenAI)
- **AI-generated context** using Claude on AWS Bedrock
- **Image descriptions** using Claude Vision
- **Table preservation** with HTML structure
- **Contextual retrieval** following Anthropic's best practices
- Automatic topic boundary detection
- Configurable chunk sizes (400-1200 tokens)

## Installation

### Prerequisites
- Python 3.13+
- uv (Python package manager)
- Azure OpenAI account
- AWS Bedrock access
- Unstructured.io API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd data-ingest
```

2. Install dependencies:
```bash
uv sync
```

3. Configure environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` with your credentials:
```bash
# AWS Bedrock Configuration (for Claude)
AWS_ACCESS_KEY_ID=<your-aws-access-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret-key>
AWS_DEFAULT_REGION=us-east-1
AWS_BEDROCK_CLAUDE_MODEL=<your-claude-model-arn>

# Azure OpenAI Configuration (for embeddings)
AZURE_OPENAI_ENDPOINT=<your-azure-endpoint>
AZURE_OPENAI_API_KEY=<your-azure-api-key>

# Unstructured API Configuration
UNSTRUCTURED_API_KEY=<your-unstructured-api-key>
UNSTRUCTURED_API_URL=<your-unstructured-api-url>
```

## Usage

### Step 1: Process Documents

Place your source documents in `dataset/src/` directory, then run:

```bash
uv run src/process_documents.py
```

**What it does:**
- Reads all files from `dataset/src/`
- Processes each file with optimal settings based on file type
- Extracts text, tables (as HTML), and images (as base64)
- Saves structured JSON to `dataset/res/`

**Output:** JSON files with structured elements in `dataset/res/`

### Step 2: Generate Contextual Chunks

```bash
uv run src/contextual_chunking.py
```

**What it does:**
- Loads processed JSON from `dataset/res/`
- Generates image descriptions using Claude Vision
- Creates semantic embeddings for all content
- Groups content into coherent chunks (400-1200 tokens)
- Generates contextual summaries using Claude
- Saves enriched chunks to `dataset/chunks/`

**Output:** JSON files with contextualized chunks in `dataset/chunks/`

## Configuration

### Document Processing Settings

Edit `process_documents.py` to adjust:

```python
processor = DocumentProcessing(
    max_workers=5  # Parallel processing threads
)
```

### Chunking Settings

Edit `contextual_chunking.py` to adjust:

```python
chunker = ContextualChunker(
    # Azure OpenAI (for embeddings)
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    embedding_model="text-embedding-3-large",

    # AWS Bedrock (for Claude)
    aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    claude_model=os.getenv("AWS_BEDROCK_CLAUDE_MODEL"),

    # Chunking parameters
    similarity_threshold=0.75,  # Topic boundary detection (0-1)
    min_chunk_size=400,         # Minimum tokens per chunk
    max_chunk_size=1200,        # Maximum tokens per chunk
    max_workers=2               # Parallel processing (reduce to 1 if rate limited)
)
```

## Output Format

### Chunk Structure

Each chunk in `dataset/chunks/*.json` contains:

```json
{
  "chunk_id": "document_name_chunk_0",
  "original_content": "Raw text or [IMAGE] description",
  "contextualized_content": "Context summary + original content",
  "metadata": {
    "document": "document_name",
    "chunk_index": 0,
    "chunk_type": "text|table|image",
    "section_title": "Section name",
    "page_number": 1,
    "element_count": 3,
    "has_image": true,
    "image_base64": "base64_encoded_image_data",
    "image_description": "AI-generated description",
    "filetype": "png"
  },
  "token_count": 450
}
```

### Chunk Types

- **text**: Regular text content grouped semantically
- **table**: Standalone table chunks with HTML structure preserved
- **image**: Images with AI-generated descriptions and base64 data

## How It Works

### Semantic Chunking Process

1. **Element Processing**
   - Text elements: Used as-is
   - Images: Claude Vision generates detailed descriptions
   - Tables: HTML structure preserved

2. **Embedding Generation**
   - All elements embedded using Azure text-embedding-3-large
   - Creates 3072-dimensional vectors

3. **Semantic Grouping**
   - Calculates cosine similarity between consecutive elements
   - Creates chunk boundaries when similarity drops below threshold (0.75)
   - Respects min/max token limits

4. **Context Generation**
   - Entire document provided as context to Claude
   - Claude generates 50-100 token summary situating each chunk
   - Follows [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) methodology

5. **Special Handling**
   - Tables: Standalone chunks (preserve structure)
   - Images: Standalone chunks (with descriptions + base64)
   - Titles: Tracked for section context

## Retrieval Strategy

### Text-Based Search
Use the `contextualized_content` field for embedding-based retrieval:
- Chunks include context for better search accuracy
- Image descriptions are searchable
- Table structure preserved in HTML

### Image Retrieval
When retrieving image chunks:
1. Search using the AI-generated description
2. Decode `metadata.image_base64` to display the actual image
3. Use `metadata.filetype` for proper rendering

## Troubleshooting

### Rate Limiting Errors

**Problem:** `ThrottlingException` from AWS Bedrock

**Solution:**
1. Reduce `max_workers` to 1 in `contextual_chunking.py`:
```python
max_workers=1  # Sequential processing
```

2. Or disable parallel processing:
```python
chunker.process_all_documents(
    parallel=False  # Disable parallel processing
)
```

### Image Description Errors

**Problem:** `ValidationException` for media_type

**Solution:** The system automatically maps file types to supported formats (jpeg, png, gif, webp). If you see this error, the image format may not be supported by Claude Vision.

**Problem:** Empty content or index errors

**Solution:** The image may be:
- Too large (reduce image size in source documents)
- Corrupted base64 data
- Violating content policies

The system will automatically fallback to `[Image: description unavailable]`

### Memory Issues

**Problem:** Out of memory during processing

**Solution:**
1. Process fewer documents at once
2. Reduce `max_workers`
3. Increase system memory

## Directory Structure

```
data-ingest/
├── src/
│   ├── process_documents.py      # Step 1: Document extraction
│   └── contextual_chunking.py    # Step 2: Chunking & enrichment
├── dataset/
│   ├── src/                       # Input: Place documents here
│   ├── res/                       # Output: Structured JSON
│   └── chunks/                    # Output: Contextual chunks
├── .env                           # Configuration (create from .env.example)
├── .env.example                   # Configuration template
├── pyproject.toml                 # Dependencies
└── README.md                      # This file
```

## Cost Considerations

### API Usage
- **Azure OpenAI Embeddings**: ~$0.13 per 1M tokens (text-embedding-3-large)
- **AWS Bedrock Claude**: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- **Unstructured.io**: Varies by plan

### Optimization Tips
1. Set `use_llm_context=False` to skip Claude context generation (faster, cheaper)
2. Reduce `max_workers` to avoid rate limits
3. Filter documents before processing
4. Disable image descriptions if not needed

## Advanced Usage

### Skip Context Generation

For faster processing without Claude context:

```python
chunker.process_all_documents(
    use_llm_context=False  # Skip Claude context generation
)
```

### Process Single Document

```python
chunker.process_document(
    json_path="dataset/res/document.json",
    output_path="dataset/chunks/document_chunks.json",
    use_llm_context=True,
    parallel=True
)
```

## References

- [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [Unstructured.io Documentation](https://docs.unstructured.io/)
- [Azure OpenAI Embeddings](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/embeddings)
- [AWS Bedrock Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude.html)

## Support

For issues and questions:
- Create an issue in the repository
- Check troubleshooting section above
- Review API documentation for rate limits and quotas
