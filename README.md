# Document Processing & Contextual Chunking Pipeline

An intelligent document processing system that extracts content from various file formats, generates semantic chunks with contextual summaries, indexes them into a vector database, and enables filtered semantic retrieval.

## Overview

This pipeline processes documents through three main stages:

1. **Document Processing** (`process_documents.py`): Extracts structured content from documents using Unstructured.io API
2. **Contextual Chunking** (`contextual_chunking.py`): Creates semantically meaningful chunks with AI-generated context using Amazon Titan embeddings and AWS Bedrock Claude
3. **Vector Indexing** (`indexing.py`): Indexes chunks into Pinecone vector database with full metadata for filtered retrieval

## Features

### Document Processing
- Multi-format support: PDF, PPTX, DOCX, HTML, TXT, MD, CSV
- High-resolution content extraction with layout detection
- Table structure inference with HTML preservation
- Image extraction as base64
- Parallel processing for speed

### Contextual Chunking
- **Semantic chunking** using Amazon Titan Embed Text v2
- **AI-generated context** using Claude on AWS Bedrock
- **Image descriptions** using Claude Vision
- **Table preservation** with HTML structure
- **Contextual retrieval** following Anthropic's best practices
- Automatic topic boundary detection
- Configurable chunk sizes (400-1200 tokens)
- Flexible embedding dimensions (256, 512, or 1024)

### Vector Indexing & Retrieval
- **Pinecone vector database** integration for scalable storage
- **Metadata filtering** for targeted search (document, page, type, section)
- **Batch processing** script for chunking multiple documents
- **Flexible retrieval patterns** with combined filters
- Filter by chunk type (text, table, image)
- Filter by page range, section, or document
- Search images by AI-generated descriptions

## Installation

### Prerequisites
- Python 3.13+
- uv (Python package manager)
- AWS Bedrock access (for Titan embeddings and Claude)
- Unstructured.io API key
- Pinecone account and API key (for vector indexing)

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
# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=us-east-1
AWS_BEDROCK_CLAUDE_MODEL=<YOUR_CLAUDE_MODEL_ARN_OR_ID>
AWS_BEDROCK_VISION_MODEL=<YOUR_VISION_MODEL_ARN_OR_ID>
AWS_BEDROCK_TITAN_EMBEDDING_MODEL=<YOUR_TITAN_EMBEDDING_MODEL_ARN_OR_ID>

# Pinecone Configuration
PINECONE_API_KEY=<YOUR_PINECONE_API_KEY>
PINECONE_INDEX_NAME=data-ingest
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Unstructured API Configuration
UNSTRUCTURED_API_KEY=<YOUR_UNSTRUCTURED_API_KEY>
UNSTRUCTURED_API_URL=<YOUR_UNSTRUCTURED_API_URL>
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

#### Option A: Process All Documents (Batch)

```bash
./chunk_all_documents.sh
```

**What it does:**
- Finds all JSON files in `dataset/res/`
- Processes each document sequentially with error handling
- Shows progress and provides summary at completion
- Reports any failed documents

#### Option B: Process Single Document

```bash
uv run src/contextual_chunking.py -s dataset/res/document.json
```

Or process all documents programmatically:

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

### Step 3: Index Chunks into Pinecone

```bash
uv run src/indexing.py
```

**What it does:**
- Loads all chunk files from `dataset/chunks/`
- Generates embeddings for contextualized content using Amazon Titan
- Creates or connects to Pinecone index
- Upserts vectors with full metadata (batch processing)
- Stores filterable metadata: document, chunk_type, page_number, section_title, etc.

**Output:** Chunks indexed in Pinecone with searchable metadata

**Note:** Image base64 data is NOT stored in Pinecone due to size limits. Use chunk_id to retrieve original chunk JSON if you need the image data.

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
    # AWS Bedrock (for both embeddings and Claude)
    aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    claude_model=os.getenv("AWS_BEDROCK_CLAUDE_MODEL"),

    # Titan embedding settings
    embedding_model=os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL"),
    embedding_dimensions=1024,  # Options: 256, 512, or 1024
    normalize_embeddings=True,  # Better for similarity calculations

    # Chunking parameters
    similarity_threshold=0.75,  # Topic boundary detection (0-1)
    min_chunk_size=400,         # Minimum tokens per chunk
    max_chunk_size=1200,        # Maximum tokens per chunk
    max_workers=2               # Parallel processing (reduce to 1 if rate limited)
)
```

### Indexing Settings

Edit `indexing.py` to adjust:

```python
indexer = PineconeIndexer(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding_dimensions=1024,  # Must match chunking dimensions
    normalize_embeddings=True,  # Should match chunking settings
    aws_region=os.getenv("AWS_DEFAULT_REGION"),
    metric="cosine",            # Best for normalized embeddings
    batch_size=100              # Vectors per batch upsert
)
```

## Retrieval & Querying

### Basic Usage

```bash
uv run src/retrieve.py
```

This runs several retrieval examples demonstrating different filtering strategies.

### Query Examples

```python
from src.indexing import PineconeIndexer
from src.retrieve import DocumentRetriever

# Initialize
indexer = PineconeIndexer(index_name="data-ingest")
retriever = DocumentRetriever(indexer)

# 1. Basic semantic search
results = retriever.search_all("blood transfusion protocol", top_k=5)

# 2. Search within specific document
results = retriever.search_by_document(
    query="emergency procedures",
    document="Module 1 - Introduction to Emergency Fresh Whole Blood Transfusion V1.6",
    top_k=5
)

# 3. Search only tables
results = retriever.search_tables_only("dosage guidelines", top_k=5)

# 4. Search by page range
results = retriever.search_by_page_range(
    query="safety precautions",
    min_page=5,
    max_page=15,
    top_k=5
)

# 5. Search only images
results = retriever.search_images_only("medical diagram", top_k=5)

# 6. Combined filters
results = retriever.search_with_combined_filters(
    query="treatment protocol",
    document="module1.pdf",
    chunk_type="text",
    min_page=1,
    max_page=20,
    has_image=False,
    top_k=5
)

# 7. Custom Pinecone filters
results = indexer.query_with_filters(
    query_text="comprehensive overview",
    filters={"token_count": {"$gte": 500}},  # Large chunks only
    top_k=5
)
```

### Available Filters

You can filter by any of these metadata fields:

- `document`: Document name (e.g., `"module1.pdf"`)
- `chunk_type`: Type of content (`"text"`, `"table"`, `"image"`)
- `chunk_index`: Chunk position in document (integer)
- `page_number`: Page number (integer)
- `section_title`: Section heading (string)
- `has_image`: Whether chunk contains an image (boolean)
- `token_count`: Number of tokens in chunk (integer)
- `element_count`: Number of elements in chunk (integer)

### Filter Operators

- `$eq`: Equal to
- `$ne`: Not equal to
- `$gt`: Greater than
- `$gte`: Greater than or equal to
- `$lt`: Less than
- `$lte`: Less than or equal to
- `$in`: In list
- `$nin`: Not in list
- `$and`: Logical AND
- `$or`: Logical OR

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
   - All elements embedded using Amazon Titan Embed Text v2
   - Creates vectors with configurable dimensions (256, 512, or 1024)

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

The system uses Pinecone for semantic search with metadata filtering:

### Contextual Retrieval
- All chunks are indexed using their `contextualized_content` (original content + AI-generated context)
- This improves retrieval accuracy by providing situating information
- Follows [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) best practices

### Text-Based Search
- Semantic search powered by Amazon Titan embeddings
- Chunks include AI-generated context for better search accuracy
- Image descriptions are searchable
- Table structure preserved and searchable

### Image Retrieval
When retrieving image chunks:
1. Search using the AI-generated description (stored in metadata)
2. Retrieve the original chunk JSON using `chunk_id` to access `image_base64`
3. Decode `image_base64` to display the actual image
4. Use `filetype` metadata for proper rendering

**Note:** Image base64 data is stored in chunk JSON files, not in Pinecone, due to size limitations.

### Filtered Search
Combine semantic search with metadata filters:
- Narrow results to specific documents, pages, or sections
- Filter by content type (text, table, image)
- Search within page ranges
- Filter by chunk characteristics (size, element count, etc.)

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

### Pinecone Index Issues

**Problem:** Dimension mismatch error when upserting

**Solution:** Ensure `embedding_dimensions` in `indexing.py` matches the value used in `contextual_chunking.py` (default: 1024)

**Problem:** Index not found

**Solution:** The script will automatically create the index on first run. Ensure your Pinecone API key and region are correctly configured in `.env`

**Problem:** Metadata size limit exceeded

**Solution:** The system automatically truncates large metadata fields. Image base64 data is NOT stored in Pinecone to avoid this issue. Retrieve original chunk JSON files if you need full data.

## Directory Structure

```
data-ingest/
├── src/
│   ├── process_documents.py      # Step 1: Document extraction
│   ├── contextual_chunking.py    # Step 2: Chunking & enrichment
│   ├── indexing.py                # Step 3: Vector indexing
│   └── retrieve.py                # Retrieval examples
├── dataset/
│   ├── src/                       # Input: Place documents here
│   ├── res/                       # Output: Structured JSON
│   └── chunks/                    # Output: Contextual chunks
├── chunk_all_documents.sh         # Batch processing script
├── .env                           # Configuration (create from .env.example)
├── .env.example                   # Configuration template
├── pyproject.toml                 # Dependencies
└── README.md                      # This file
```

## Cost Considerations

### API Usage
- **Amazon Titan Embeddings**: ~$0.02 per 1K input tokens (Titan Embed Text v2)
- **AWS Bedrock Claude**: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- **Pinecone**: Serverless pricing varies by cloud/region (~$0.095 per 1M queries)
- **Unstructured.io**: Varies by plan

### Optimization Tips
1. Set `use_llm_context=False` to skip Claude context generation (faster, cheaper)
2. Reduce `max_workers` to avoid rate limits
3. Filter documents before processing
4. Disable image descriptions if not needed
5. Use metadata filters to reduce vector search scope and costs

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

### Index Single Chunk File

```python
from src.indexing import PineconeIndexer

indexer = PineconeIndexer(index_name="data-ingest")
stats = indexer.index_from_file("dataset/chunks/document_chunks.json")
print(f"Indexed {stats['upserted']} chunks")
```

### Delete and Rebuild Index

```python
# Delete all vectors (keeps index structure)
indexer.delete_all_vectors()

# Or delete entire index (use with caution!)
indexer.delete_index()

# Recreate and reindex
indexer = PineconeIndexer(index_name="data-ingest")
indexer.index_from_directory("dataset/chunks")
```

### Check Index Statistics

```python
stats = indexer.get_index_stats()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Dimensions: {stats['dimensions']}")
print(f"Fullness: {stats['index_fullness']}")
```

## References

- [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Unstructured.io Documentation](https://docs.unstructured.io/)
- [Amazon Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
- [AWS Bedrock Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude.html)

## Support

For issues and questions:
- Create an issue in the repository
- Check troubleshooting section above
- Review API documentation for rate limits and quotas
