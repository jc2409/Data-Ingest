"""
Data Ingest - Document Processing & Contextual Chunking Pipeline

An intelligent document processing system that extracts content from various
file formats, generates semantic chunks with contextual summaries, indexes
them into a vector database, and enables filtered semantic retrieval.
"""

from .embedding import EmbeddingClient, get_embedding, cosine_similarity
from .process_documents import DocumentProcessing
from .contextual_chunking import ContextualChunker, Chunk
from .indexing import PineconeIndexer
from .retrieve import DocumentRetriever, RetrievalResult, BM25

__all__ = [
    "DocumentProcessing",
    "ContextualChunker",
    "Chunk",
    "PineconeIndexer",
    "DocumentRetriever",
    "RetrievalResult",
    "BM25",
    "EmbeddingClient",
    "get_embedding",
    "cosine_similarity",
]
