# Data Ingest

A document processing pipeline that extracts content from various file formats, generates semantic chunks with AI-powered contextual summaries, and indexes them into Pinecone for filtered retrieval.

## Features

- **Multi-format extraction** — PDF, DOCX, PPTX, HTML, TXT, CSV via Unstructured.io
- **Contextual chunking** — Semantic grouping with AI-generated context using Claude
- **Image understanding** — Automatic descriptions via Claude Vision
- **Vector indexing** — Pinecone integration with metadata filtering
- **Embedding caching** — LRU cache to avoid redundant API calls

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS account with Bedrock access (Titan embeddings + Claude)
- Pinecone account
- Unstructured.io API key

## Quick Start

```bash
# Clone and install
git clone git@github.com:jc2409/Data-Ingest.git
cd Data-Ingest
make install

# Configure credentials
cp .env.example .env
# Edit .env with your API keys

# Run pipeline
make pipeline    # Runs: process → chunk → index

# Or run steps individually
make process     # 1. Extract documents
make chunk       # 2. Generate chunks
make index       # 3. Index to Pinecone
```

## Project Structure

```
data-ingest/
├── src/
│   ├── __init__.py              # Package exports
│   ├── embedding.py             # Shared embedding client with caching
│   ├── process_documents.py     # Document extraction
│   ├── contextual_chunking.py   # Semantic chunking + context generation
│   ├── indexing.py              # Pinecone vector indexing
│   └── retrieve.py              # Retrieval examples
├── tests/
│   ├── conftest.py              # Test fixtures
│   ├── test_*.py                # Unit tests
│   └── test_integration.py      # Integration tests (real APIs)
├── dataset/
│   ├── src/                     # Input: source documents
│   ├── res/                     # Output: extracted JSON
│   └── chunks/                  # Output: contextual chunks
├── Makefile                     # Pipeline commands
├── .env.example                 # Environment template
└── pyproject.toml               # Dependencies
```

## Configuration

Create `.env` from `.env.example`:

```bash
# AWS Bedrock
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
AWS_BEDROCK_CLAUDE_MODEL=
AWS_BEDROCK_VISION_MODEL=
AWS_BEDROCK_TITAN_EMBEDDING_MODEL=

# Pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=data-ingest
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Unstructured
UNSTRUCTURED_API_KEY=
UNSTRUCTURED_API_URL=
```

## Usage

### Pipeline Commands

```bash
make help              # Show all commands
make process           # Extract documents from dataset/src/
make chunk             # Chunk all files in dataset/res/
make chunk FILE=x.json # Chunk single file
make index             # Index to Pinecone
make pipeline          # Run full pipeline
make stats             # Show index statistics
make clean             # Remove generated files
```

### Index & Query

```python
from src import PineconeIndexer, DocumentRetriever

# Index
indexer = PineconeIndexer(index_name="data-ingest")
indexer.index_from_directory("dataset/chunks")

# Query with filters
retriever = DocumentRetriever(indexer)
results = retriever.search_with_combined_filters(
    query="control system design",
    chunk_type="text",
    min_page=1,
    max_page=50,
    top_k=5
)
```

### Metadata Filters

| Field | Type | Description |
|-------|------|-------------|
| `document` | string | Source document name |
| `chunk_type` | string | `text`, `table`, or `image` |
| `page_number` | int | Page number |
| `section_title` | string | Section heading |
| `has_image` | bool | Contains image |
| `token_count` | int | Token count |

## Testing

```bash
make dev               # Install dev dependencies
make test              # Run unit tests
make test-integration  # Run integration tests (real APIs)
make coverage          # Run tests with coverage report
```

## API Reference

### EmbeddingClient

```python
from src import EmbeddingClient, cosine_similarity

client = EmbeddingClient(dimensions=1024)
emb = client.get_embedding("text")      # Returns numpy array (cached)
emb_list = client.get_embedding_list("text")  # Returns list
client.cache_info()                      # Cache statistics
```

### ContextualChunker

```python
from src import ContextualChunker

chunker = ContextualChunker(
    embedding_dimensions=1024,
    min_chunk_size=400,
    max_chunk_size=800
)
chunks = chunker.process_document(
    json_path="dataset/res/doc.json",
    output_path="dataset/chunks/doc_chunks.json",
    use_llm_context=True
)
```

### PineconeIndexer

```python
from src import PineconeIndexer

indexer = PineconeIndexer(index_name="data-ingest")
indexer.index_from_file("dataset/chunks/doc_chunks.json")
indexer.get_index_stats()
indexer.query_with_filters("query", filters={"chunk_type": {"$eq": "table"}})
```

## References

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Amazon Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
- [Unstructured.io](https://docs.unstructured.io/)
