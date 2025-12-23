"""
Integration tests that use real API calls.

Run with: uv run pytest tests/test_integration.py --run-integration -v

WARNING: These tests make real API calls and may incur costs!
Requires valid credentials in .env file.
"""
import os
import json
import tempfile
import pytest
from dotenv import load_dotenv

load_dotenv()


# Skip all tests in this file if --run-integration not passed
pytestmark = pytest.mark.integration


def has_aws_credentials():
    """Check if AWS credentials are configured."""
    return all([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL")
    ])


def has_pinecone_credentials():
    """Check if Pinecone credentials are configured."""
    return all([
        os.getenv("PINECONE_API_KEY"),
        os.getenv("PINECONE_INDEX_NAME")
    ])


class TestEmbeddingIntegration:
    """Integration tests for EmbeddingClient with real AWS Bedrock."""

    @pytest.mark.skipif(not has_aws_credentials(), reason="AWS credentials not configured")
    def test_real_embedding_generation(self):
        """Test generating a real embedding from AWS Bedrock."""
        from src.embedding import EmbeddingClient

        client = EmbeddingClient(
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            model_id=os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL"),
            dimensions=1024,
            normalize=True
        )

        embedding = client.get_embedding("Hello, this is a test sentence.")

        # Verify embedding properties
        assert len(embedding) == 1024
        assert embedding.dtype.name.startswith('float')
        assert not all(v == 0 for v in embedding)  # Not a zero vector

        # Test caching works
        embedding2 = client.get_embedding("Hello, this is a test sentence.")
        assert client.cache_info().hits == 1

        print(f"\n  Embedding shape: {embedding.shape}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Cache info: {client.cache_info()}")

    @pytest.mark.skipif(not has_aws_credentials(), reason="AWS credentials not configured")
    def test_real_cosine_similarity(self):
        """Test cosine similarity with real embeddings."""
        from src.embedding import EmbeddingClient, cosine_similarity

        client = EmbeddingClient(
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            model_id=os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL"),
            dimensions=1024
        )

        # Similar sentences should have high similarity
        emb1 = client.get_embedding("The cat sat on the mat.")
        emb2 = client.get_embedding("A cat was sitting on a mat.")
        emb3 = client.get_embedding("The stock market crashed yesterday.")

        sim_similar = cosine_similarity(emb1, emb2)
        sim_different = cosine_similarity(emb1, emb3)

        print(f"\n  Similar sentences similarity: {sim_similar:.4f}")
        print(f"  Different sentences similarity: {sim_different:.4f}")

        assert sim_similar > sim_different
        assert sim_similar > 0.7  # Similar sentences should be quite similar


class TestPineconeIntegration:
    """Integration tests for PineconeIndexer with real Pinecone."""

    @pytest.fixture
    def test_index_name(self):
        """Use a test-specific index name to avoid conflicts."""
        return f"test-integration-{os.getpid()}"

    @pytest.mark.skipif(
        not (has_aws_credentials() and has_pinecone_credentials()),
        reason="AWS and Pinecone credentials required"
    )
    def test_real_pinecone_query(self):
        """Test querying real Pinecone index."""
        from src.indexing import PineconeIndexer

        # Use existing index from .env
        indexer = PineconeIndexer(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding_dimensions=1024,
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )

        # Get stats
        stats = indexer.get_index_stats()
        print(f"\n  Index stats: {stats}")

        # Only run query if index has vectors
        if stats["total_vectors"] > 0:
            results = indexer.query_with_filters(
                query_text="test query",
                top_k=3
            )
            print(f"  Found {len(results)} results")
            for r in results:
                print(f"    - Score: {r.score:.4f}, Doc: {r.metadata.get('document', 'N/A')}")


class TestChunkingIntegration:
    """Integration tests for contextual chunking with real APIs."""

    @pytest.mark.skipif(not has_aws_credentials(), reason="AWS credentials not configured")
    def test_real_chunk_processing(self):
        """Test processing a document with real LLM context."""
        from src.contextual_chunking import ContextualChunker

        chunker = ContextualChunker(
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            claude_model=os.getenv("AWS_BEDROCK_CLAUDE_MODEL"),
            vision_model=os.getenv("AWS_BEDROCK_VISION_MODEL"),
            embedding_model=os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL"),
            embedding_dimensions=1024,
            min_chunk_size=50,
            max_chunk_size=200
        )

        # Create test elements
        test_elements = [
            {"type": "Title", "text": "Introduction to Testing", "metadata": {"page_number": 1}},
            {"type": "NarrativeText", "text": "This is a test document about software testing. Testing is important for quality.", "metadata": {"page_number": 1}},
            {"type": "NarrativeText", "text": "Unit tests verify individual components work correctly.", "metadata": {"page_number": 1}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "test_doc.json")
            output_path = os.path.join(tmpdir, "test_doc_chunks.json")

            with open(input_path, 'w') as f:
                json.dump(test_elements, f)

            # Process WITHOUT LLM context (faster, cheaper)
            chunks = chunker.process_document(
                json_path=input_path,
                output_path=output_path,
                use_llm_context=False,  # Set to True to test Claude context generation
                parallel=False,
                chunking_strategy="basic"
            )

            print(f"\n  Created {len(chunks)} chunks")
            for chunk in chunks:
                print(f"    - {chunk.chunk_id}: {chunk.token_count} tokens")

            assert len(chunks) > 0
            assert os.path.exists(output_path)

    @pytest.mark.skipif(not has_aws_credentials(), reason="AWS credentials not configured")
    def test_real_llm_context_generation(self):
        """Test actual LLM context generation (costs money!)."""
        from src.contextual_chunking import ContextualChunker

        chunker = ContextualChunker(
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            claude_model=os.getenv("AWS_BEDROCK_CLAUDE_MODEL"),
            embedding_model=os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL"),
        )

        # Test context generation
        chunk_text = "Unit tests verify individual components work correctly."
        full_doc = "Introduction to Testing\n\nThis is a test document about software testing. Testing is important for quality.\n\nUnit tests verify individual components work correctly."

        context = chunker.generate_contextual_summary(
            chunk_text=chunk_text,
            full_document_text=full_doc,
            document_name="test_doc",
            section_title="Testing",
            page_number=1
        )

        print(f"\n  Generated context: {context}")

        assert len(context) > 0
        assert context != chunk_text  # Should be different from original


class TestEndToEndIntegration:
    """Full pipeline integration test."""

    @pytest.mark.skipif(
        not (has_aws_credentials() and has_pinecone_credentials()),
        reason="All credentials required for E2E test"
    )
    def test_full_pipeline(self):
        """Test the complete pipeline: chunk -> index -> retrieve."""
        from src.contextual_chunking import ContextualChunker
        from src.indexing import PineconeIndexer
        import time

        # 1. Create chunks
        chunker = ContextualChunker(
            aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            claude_model=os.getenv("AWS_BEDROCK_CLAUDE_MODEL"),
            embedding_model=os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL"),
            min_chunk_size=20,
            max_chunk_size=100
        )

        test_elements = [
            {"type": "Title", "text": "Integration Test Document", "metadata": {"page_number": 1}},
            {"type": "NarrativeText", "text": "This document tests the full data ingestion pipeline from chunking to retrieval.", "metadata": {"page_number": 1}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "e2e_test.json")
            output_path = os.path.join(tmpdir, "e2e_test_chunks.json")

            with open(input_path, 'w') as f:
                json.dump(test_elements, f)

            chunks = chunker.process_document(
                json_path=input_path,
                output_path=output_path,
                use_llm_context=False,
                parallel=False
            )

            print(f"\n  Step 1: Created {len(chunks)} chunks")

            # 2. Index chunks
            indexer = PineconeIndexer(
                index_name=os.getenv("PINECONE_INDEX_NAME"),
                embedding_dimensions=1024,
                aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            )

            # Load and index
            with open(output_path) as f:
                chunk_data = json.load(f)

            stats = indexer.index_chunks(chunk_data)
            print(f"  Step 2: Indexed {stats['upserted']} vectors")

            # Wait for indexing to propagate
            time.sleep(2)

            # 3. Query
            results = indexer.query_with_filters(
                query_text="data ingestion pipeline",
                top_k=5
            )

            print(f"  Step 3: Retrieved {len(results)} results")
            for r in results:
                print(f"    - Score: {r.score:.4f}")

            assert len(results) > 0
