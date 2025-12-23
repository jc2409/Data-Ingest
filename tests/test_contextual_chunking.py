"""Tests for the contextual chunking module."""
import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from src.contextual_chunking import ContextualChunker, Chunk


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk instance."""
        chunk = Chunk(
            chunk_id="test_chunk_0",
            original_content="Original text",
            contextualized_content="Context: Original text",
            metadata={"page": 1},
            token_count=10
        )

        assert chunk.chunk_id == "test_chunk_0"
        assert chunk.original_content == "Original text"
        assert chunk.contextualized_content == "Context: Original text"
        assert chunk.metadata == {"page": 1}
        assert chunk.token_count == 10

    def test_chunk_to_dict(self):
        """Test converting Chunk to dictionary."""
        chunk = Chunk(
            chunk_id="test_chunk_0",
            original_content="Original text",
            contextualized_content="Context: Original text",
            metadata={"page": 1},
            token_count=10
        )

        result = chunk.to_dict()

        assert result["chunk_id"] == "test_chunk_0"
        assert result["original_content"] == "Original text"
        assert result["contextualized_content"] == "Context: Original text"
        assert result["metadata"] == {"page": 1}
        assert result["token_count"] == 10


class TestContextualChunker:
    """Tests for ContextualChunker class."""

    @pytest.fixture
    def mock_chunker(self):
        """Create a chunker with mocked AWS clients."""
        with patch("boto3.client"), \
             patch("src.contextual_chunking.EmbeddingClient") as mock_embed:

            mock_embed_instance = MagicMock()
            mock_embed_instance.get_embedding.return_value = MagicMock()
            mock_embed.return_value = mock_embed_instance

            chunker = ContextualChunker(
                aws_region="us-east-1",
                claude_model="anthropic.claude-3-haiku",
                vision_model="anthropic.claude-3-sonnet",
                embedding_model="amazon.titan-embed-text-v2:0",
                embedding_dimensions=1024,
                similarity_threshold=0.70,
                min_chunk_size=100,
                max_chunk_size=500
            )

            yield chunker

    def test_count_tokens(self, mock_chunker):
        """Test token counting."""
        text = "Hello world, this is a test."
        count = mock_chunker.count_tokens(text)

        assert isinstance(count, int)
        assert count > 0

    def test_load_unstructured_elements(self, mock_chunker, sample_elements):
        """Test loading elements from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_elements, f)
            temp_path = f.name

        try:
            elements = mock_chunker.load_unstructured_elements(temp_path)
            assert len(elements) == 5
            assert elements[0]["type"] == "Title"
        finally:
            os.unlink(temp_path)

    def test_html_table_to_markdown(self, mock_chunker):
        """Test HTML table conversion to Markdown."""
        html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>John</td><td>30</td></tr></table>"
        markdown = mock_chunker.html_table_to_markdown(html)

        assert "Name" in markdown
        assert "Age" in markdown
        assert "John" in markdown
        assert "30" in markdown
        assert "|" in markdown

    def test_html_table_to_markdown_empty(self, mock_chunker):
        """Test HTML table conversion with empty table."""
        html = "<table></table>"
        markdown = mock_chunker.html_table_to_markdown(html)

        assert markdown.strip() == ""

    def test_combine_chunk_text(self, mock_chunker):
        """Test combining elements into chunk text."""
        chunk = {
            "elements": [
                {"type": "NarrativeText", "text": "First paragraph."},
                {"type": "NarrativeText", "text": "Second paragraph."}
            ]
        }

        result = mock_chunker.combine_chunk_text(chunk)

        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_combine_chunk_text_with_table(self, mock_chunker):
        """Test combining elements with HTML table."""
        chunk = {
            "elements": [
                {
                    "type": "Table",
                    "text": "Header | Value",
                    "metadata": {
                        "text_as_html": "<table><tr><th>Header</th></tr><tr><td>Value</td></tr></table>"
                    }
                }
            ]
        }

        result = mock_chunker.combine_chunk_text(chunk)

        assert "Header" in result
        assert "Value" in result

    def test_finalize_chunk_sets_defaults(self, mock_chunker):
        """Test that _finalize_chunk sets default metadata."""
        chunk = {
            "elements": [{"type": "Text", "text": "content"}],
            "tokens": 10,
            "metadata": {"page_number": 1}
        }

        result = mock_chunker._finalize_chunk(chunk)

        assert result["metadata"]["chunk_type"] == "text"
        assert result["metadata"]["has_image"] is False

    def test_finalize_chunk_preserves_existing(self, mock_chunker):
        """Test that _finalize_chunk preserves existing metadata."""
        chunk = {
            "elements": [{"type": "Image", "text": "content"}],
            "tokens": 10,
            "metadata": {
                "page_number": 1,
                "chunk_type": "image",
                "has_image": True
            }
        }

        result = mock_chunker._finalize_chunk(chunk)

        assert result["metadata"]["chunk_type"] == "image"
        assert result["metadata"]["has_image"] is True


class TestContextualChunkerIntegration:
    """Integration tests for ContextualChunker (with mocked API calls)."""

    @pytest.fixture
    def chunker_with_mocks(self):
        """Create chunker with all external calls mocked."""
        with patch("boto3.client") as mock_boto, \
             patch("src.contextual_chunking.EmbeddingClient") as mock_embed:

            mock_bedrock = MagicMock()
            response_body = MagicMock()
            response_body.read.return_value = json.dumps({
                "content": [{"text": "This chunk discusses the introduction."}]
            })
            mock_bedrock.invoke_model.return_value = {"body": response_body}
            mock_boto.return_value = mock_bedrock

            import numpy as np
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_embedding.return_value = np.random.rand(1024)
            mock_embed.return_value = mock_embed_instance

            chunker = ContextualChunker(
                min_chunk_size=10,
                max_chunk_size=100,
                max_workers=1
            )

            yield chunker

    def test_group_elements_by_size(self, chunker_with_mocks, sample_elements):
        """Test grouping elements by size."""
        text_elements = [e for e in sample_elements if e["type"] != "Image"]

        chunks = chunker_with_mocks.group_elements_by_size(text_elements)

        assert len(chunks) > 0
        table_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "table"]
        assert len(table_chunks) == 1

    def test_process_document_creates_output(self, chunker_with_mocks, sample_elements):
        """Test that process_document creates output file."""
        text_elements = [e for e in sample_elements if e["type"] != "Image"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            output_path = os.path.join(tmpdir, "output_chunks.json")

            with open(input_path, 'w') as f:
                json.dump(text_elements, f)

            chunks = chunker_with_mocks.process_document(
                json_path=input_path,
                output_path=output_path,
                use_llm_context=False,
                parallel=False
            )

            assert os.path.exists(output_path)
            assert len(chunks) > 0

            with open(output_path) as f:
                saved_chunks = json.load(f)
            assert len(saved_chunks) == len(chunks)
