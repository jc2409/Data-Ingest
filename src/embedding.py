"""
Shared embedding utilities for the data-ingest pipeline.
Provides cached embedding generation using Amazon Titan.
"""
import json
from functools import lru_cache
from typing import List, Tuple

import boto3
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


class EmbeddingClient:
    """
    Shared embedding client with caching support.

    Uses Amazon Titan Embed Text v2 for generating embeddings.
    Implements LRU caching to avoid re-embedding identical text.
    """

    def __init__(
        self,
        aws_region: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        normalize: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize the embedding client.

        Args:
            aws_region: AWS region for Bedrock
            model_id: Amazon Titan embedding model ID
            dimensions: Embedding dimensions (256, 512, or 1024)
            normalize: Whether to normalize embeddings
            cache_size: Maximum number of embeddings to cache
        """
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region
        )
        self.model_id = model_id
        self.dimensions = dimensions
        self.normalize = normalize

        # Create cached version of the embedding function
        self._get_embedding_cached = lru_cache(maxsize=cache_size)(
            self._get_embedding_uncached
        )

    def _get_embedding_uncached(self, text: str) -> Tuple[float, ...]:
        """
        Generate embedding without caching (internal use).
        Returns tuple for hashability in lru_cache.
        """
        try:
            request_body = {
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": self.normalize
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            return tuple(embedding)

        except Exception as e:
            print(f"Error generating embedding: {e}")
            return tuple([0.0] * self.dimensions)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text with caching.

        Args:
            text: Text to embed

        Returns:
            numpy array of embedding vector
        """
        embedding_tuple = self._get_embedding_cached(text)
        return np.array(embedding_tuple)

    def get_embedding_list(self, text: str) -> List[float]:
        """
        Generate embedding for text, returning as list.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        embedding_tuple = self._get_embedding_cached(text)
        return list(embedding_tuple)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._get_embedding_cached.cache_clear()

    def cache_info(self):
        """Get cache statistics."""
        return self._get_embedding_cached.cache_info()


# Convenience function for simple usage
_default_client = None


def get_embedding(
    text: str,
    aws_region: str = "us-east-1",
    model_id: str = "amazon.titan-embed-text-v2:0",
    dimensions: int = 1024,
    normalize: bool = True
) -> np.ndarray:
    """
    Convenience function to get embedding with default client.

    Uses a shared cached client for efficiency.
    """
    global _default_client

    if _default_client is None:
        _default_client = EmbeddingClient(
            aws_region=aws_region,
            model_id=model_id,
            dimensions=dimensions,
            normalize=normalize
        )

    return _default_client.get_embedding(text)
