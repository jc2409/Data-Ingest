"""Tests for the embedding module."""
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.embedding import EmbeddingClient, cosine_similarity


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_zero_vector_first(self):
        """Zero vector should return 0.0 similarity."""
        vec1 = np.zeros(3)
        vec2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_zero_vector_second(self):
        """Zero vector should return 0.0 similarity."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.zeros(3)
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.1, 2.1, 3.1])
        similarity = cosine_similarity(vec1, vec2)
        assert similarity > 0.99


class TestEmbeddingClient:
    """Tests for EmbeddingClient class."""

    @patch("boto3.client")
    def test_init(self, mock_boto_client):
        """Test client initialization."""
        client = EmbeddingClient(
            aws_region="us-west-2",
            model_id="amazon.titan-embed-text-v2:0",
            dimensions=512,
            normalize=True
        )

        mock_boto_client.assert_called_once_with(
            service_name="bedrock-runtime",
            region_name="us-west-2"
        )
        assert client.dimensions == 512
        assert client.normalize is True

    @patch("boto3.client")
    def test_get_embedding(self, mock_boto_client):
        """Test embedding generation."""
        sample_embedding = [0.1] * 1024
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({"embedding": sample_embedding})

        client_instance = MagicMock()
        client_instance.invoke_model.return_value = {"body": response_body}
        mock_boto_client.return_value = client_instance

        client = EmbeddingClient()
        result = client.get_embedding("test text")

        assert isinstance(result, np.ndarray)
        assert len(result) == 1024
        assert result[0] == pytest.approx(0.1)

    @patch("boto3.client")
    def test_get_embedding_list(self, mock_boto_client):
        """Test embedding generation returning list."""
        sample_embedding = [0.2] * 1024
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({"embedding": sample_embedding})

        client_instance = MagicMock()
        client_instance.invoke_model.return_value = {"body": response_body}
        mock_boto_client.return_value = client_instance

        client = EmbeddingClient()
        result = client.get_embedding_list("test text")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert result[0] == pytest.approx(0.2)

    @patch("boto3.client")
    def test_caching(self, mock_boto_client):
        """Test that embeddings are cached."""
        sample_embedding = [0.1] * 1024
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({"embedding": sample_embedding})

        client_instance = MagicMock()
        client_instance.invoke_model.return_value = {"body": response_body}
        mock_boto_client.return_value = client_instance

        client = EmbeddingClient()

        result1 = client.get_embedding("cached text")
        result2 = client.get_embedding("cached text")

        assert client_instance.invoke_model.call_count == 1
        np.testing.assert_array_equal(result1, result2)

    @patch("boto3.client")
    def test_cache_info(self, mock_boto_client):
        """Test cache statistics."""
        sample_embedding = [0.1] * 1024
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({"embedding": sample_embedding})

        client_instance = MagicMock()
        client_instance.invoke_model.return_value = {"body": response_body}
        mock_boto_client.return_value = client_instance

        client = EmbeddingClient()

        client.get_embedding("text1")
        client.get_embedding("text1")
        client.get_embedding("text2")

        info = client.cache_info()
        assert info.hits == 1
        assert info.misses == 2

    @patch("boto3.client")
    def test_clear_cache(self, mock_boto_client):
        """Test cache clearing."""
        sample_embedding = [0.1] * 1024
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({"embedding": sample_embedding})

        client_instance = MagicMock()
        client_instance.invoke_model.return_value = {"body": response_body}
        mock_boto_client.return_value = client_instance

        client = EmbeddingClient()

        client.get_embedding("text")
        client.clear_cache()

        info = client.cache_info()
        assert info.hits == 0
        assert info.misses == 0

    @patch("boto3.client")
    def test_error_handling(self, mock_boto_client):
        """Test error handling returns zero vector."""
        client_instance = MagicMock()
        client_instance.invoke_model.side_effect = Exception("API Error")
        mock_boto_client.return_value = client_instance

        client = EmbeddingClient(dimensions=512)
        result = client.get_embedding("test")

        assert len(result) == 512
        assert all(v == 0.0 for v in result)
