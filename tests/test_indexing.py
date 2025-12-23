"""Tests for the indexing module."""
import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch


class TestPineconeIndexer:
    """Tests for PineconeIndexer class."""

    @pytest.fixture
    def mock_indexer(self):
        """Create indexer with mocked Pinecone and AWS clients."""
        with patch("src.indexing.Pinecone") as mock_pinecone, \
             patch("src.indexing.EmbeddingClient") as mock_embed:

            mock_pc_instance = MagicMock()
            mock_pc_instance.list_indexes.return_value = []
            mock_index = MagicMock()
            mock_pc_instance.Index.return_value = mock_index
            mock_pinecone.return_value = mock_pc_instance

            mock_embed_instance = MagicMock()
            mock_embed_instance.get_embedding_list.return_value = [0.1] * 1024
            mock_embed.return_value = mock_embed_instance

            from src.indexing import PineconeIndexer

            indexer = PineconeIndexer(
                index_name="test-index",
                embedding_dimensions=1024,
                batch_size=10
            )

            indexer._mock_pc = mock_pc_instance
            indexer._mock_index = mock_index
            indexer._mock_embed = mock_embed_instance

            yield indexer

    def test_init_creates_index(self, mock_indexer):
        """Test that init creates index if not exists."""
        mock_indexer._mock_pc.create_index.assert_called_once()

    def test_init_uses_existing_index(self):
        """Test that init uses existing index."""
        with patch("src.indexing.Pinecone") as mock_pinecone, \
             patch("src.indexing.EmbeddingClient"):

            mock_pc_instance = MagicMock()
            mock_existing = MagicMock()
            mock_existing.name = "existing-index"
            mock_pc_instance.list_indexes.return_value = [mock_existing]
            mock_pinecone.return_value = mock_pc_instance

            from src.indexing import PineconeIndexer
            PineconeIndexer(index_name="existing-index")

            mock_pc_instance.create_index.assert_not_called()

    def test_prepare_vector(self, mock_indexer, sample_chunks):
        """Test preparing a chunk for Pinecone."""
        chunk = sample_chunks[0]
        result = mock_indexer.prepare_vector(chunk)

        assert result["id"] == "test_doc_chunk_0"
        assert len(result["values"]) == 1024
        assert result["metadata"]["document"] == "test_doc"
        assert result["metadata"]["chunk_type"] == "text"

    def test_prepare_vector_truncates_content(self, mock_indexer):
        """Test that large content is truncated."""
        long_content = "x" * 5000
        chunk = {
            "chunk_id": "test",
            "original_content": long_content,
            "contextualized_content": long_content,
            "metadata": {
                "document": "test",
                "chunk_index": 0,
                "chunk_type": "text",
                "page_number": 1,
                "element_count": 1,
                "has_image": False
            },
            "token_count": 1000
        }

        result = mock_indexer.prepare_vector(chunk)

        assert len(result["metadata"]["original_content"]) == 2000

    def test_index_chunks(self, mock_indexer, sample_chunks):
        """Test indexing multiple chunks."""
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 2
        mock_stats.dimension = 1024
        mock_indexer._mock_index.describe_index_stats.return_value = mock_stats

        result = mock_indexer.index_chunks(sample_chunks)

        assert result["upserted"] == 2
        assert result["total_vectors"] == 2
        mock_indexer._mock_index.upsert.assert_called()

    def test_index_from_file(self, mock_indexer, sample_chunks):
        """Test indexing from JSON file."""
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 2
        mock_stats.dimension = 1024
        mock_indexer._mock_index.describe_index_stats.return_value = mock_stats

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_chunks, f)
            temp_path = f.name

        try:
            result = mock_indexer.index_from_file(temp_path)
            assert result["upserted"] == 2
        finally:
            os.unlink(temp_path)

    def test_index_from_directory(self, mock_indexer, sample_chunks):
        """Test indexing from directory of files."""
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 4
        mock_stats.dimension = 1024
        mock_indexer._mock_index.describe_index_stats.return_value = mock_stats

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                path = os.path.join(tmpdir, f"doc{i}_chunks.json")
                with open(path, 'w') as f:
                    json.dump(sample_chunks, f)

            result = mock_indexer.index_from_directory(tmpdir)

            assert result["upserted"] == 4

    def test_query_with_filters(self, mock_indexer):
        """Test querying with metadata filters."""
        mock_match = MagicMock()
        mock_match.score = 0.95
        mock_match.metadata = {"document": "test"}

        mock_results = MagicMock()
        mock_results.matches = [mock_match]
        mock_indexer._mock_index.query.return_value = mock_results

        filters = {"document": {"$eq": "test"}}
        results = mock_indexer.query_with_filters(
            query_text="test query",
            filters=filters,
            top_k=5
        )

        assert len(results) == 1
        mock_indexer._mock_index.query.assert_called_once()
        call_kwargs = mock_indexer._mock_index.query.call_args[1]
        assert call_kwargs["filter"] == filters
        assert call_kwargs["top_k"] == 5

    def test_get_index_stats(self, mock_indexer):
        """Test getting index statistics."""
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 100
        mock_stats.dimension = 1024
        mock_stats.index_fullness = 0.5
        mock_indexer._mock_index.describe_index_stats.return_value = mock_stats

        result = mock_indexer.get_index_stats()

        assert result["total_vectors"] == 100
        assert result["dimensions"] == 1024
        assert result["index_fullness"] == 0.5

    def test_delete_all_vectors(self, mock_indexer):
        """Test deleting all vectors."""
        mock_indexer.delete_all_vectors()

        mock_indexer._mock_index.delete.assert_called_once_with(delete_all=True)

    def test_delete_index(self, mock_indexer):
        """Test deleting entire index."""
        mock_indexer.delete_index()

        mock_indexer._mock_pc.delete_index.assert_called_once_with("test-index")
