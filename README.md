# Data Ingest

A document processing pipeline that extracts content from various file formats, generates semantic chunks with AI-powered contextual summaries, and indexes them into Pinecone for retrieval.

## Features

- **Multi-format extraction** — PDF, DOCX, PPTX, HTML, TXT, CSV via Unstructured.io
- **Contextual chunking** — Semantic grouping with AI-generated context using Claude
- **Overlapping chunks** — Configurable overlap for better boundary handling
- **Image understanding** — Automatic descriptions via Claude Vision
- **Vector indexing** — Pinecone integration with metadata filtering
- **Advanced retrieval** — Hybrid search, HyDE, query expansion, LLM reranking
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
make chunk             # Chunk with basic strategy (default)
make chunk-semantic    # Chunk with semantic strategy
make index             # Index to Pinecone
make pipeline          # Run full pipeline
make stats             # Show index statistics
make clean             # Remove generated files
```

### Python API

```python
from src import PineconeIndexer, DocumentRetriever

# Index chunks
indexer = PineconeIndexer(index_name="data-ingest")
indexer.index_from_directory("dataset/chunks")

# Basic search
retriever = DocumentRetriever(indexer)
results = retriever.search("control system design", top_k=5)

# Filtered search
results = retriever.search_combined(
    query="safety protocol",
    chunk_type="text",
    min_page=1,
    max_page=50
)

# Advanced retrieval
results = retriever.hyde_search("what is the protocol?")      # HyDE
results = retriever.search_expanded("query", num_expansions=3) # Query expansion
results = retriever.advanced_search("query", use_reranking=True) # Full pipeline
```

### Metadata Filters

| Field | Type | Description |
|-------|------|-------------|
| `document` | string | Source document name |
| `chunk_type` | string | `text`, `table`, or `image` |
| `page_number` | int | Page number |
| `section_title` | string | Section heading |
| `has_image` | bool | Contains image |

## Testing

```bash
make dev               # Install dev dependencies
make test              # Run unit tests
make test-integration  # Run integration tests (requires API credentials)
make coverage          # Run tests with coverage report
```

## Project Structure

```
data-ingest/
├── src/
│   ├── embedding.py             # Embedding client with caching
│   ├── process_documents.py     # Document extraction
│   ├── contextual_chunking.py   # Chunking + context generation
│   ├── indexing.py              # Pinecone vector indexing
│   └── retrieve.py              # Advanced retrieval methods
├── tests/                       # Unit and integration tests
├── dataset/
│   ├── src/                     # Input documents
│   ├── res/                     # Extracted JSON
│   └── chunks/                  # Contextual chunks
└── Makefile                     # Pipeline commands
```

## References

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Amazon Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
- [Unstructured.io](https://docs.unstructured.io/)
