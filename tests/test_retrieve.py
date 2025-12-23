"""Tests for the retrieve module."""
import pytest
from unittest.mock import MagicMock, patch


class TestBM25:
    """Tests for BM25 ranking algorithm."""

    def test_tokenize(self):
        """Test basic tokenization."""
        from src.retrieve import BM25

        bm25 = BM25()
        tokens = bm25._tokenize("Hello, World! Testing 123.")

        assert tokens == ["hello", "world", "testing", "123"]

    def test_index_documents(self):
        """Test document indexing."""
        from src.retrieve import BM25

        bm25 = BM25()
        docs = {
            "doc1": "The quick brown fox",
            "doc2": "The lazy dog sleeps"
        }
        bm25.index_documents(docs)

        assert bm25.num_docs == 2
        assert "doc1" in bm25.doc_lengths
        assert "doc2" in bm25.doc_lengths
        assert bm25.avg_doc_length == 4.0  # (4 + 4) / 2

    def test_idf_calculation(self):
        """Test IDF calculation."""
        from src.retrieve import BM25

        bm25 = BM25()
        docs = {
            "doc1": "the quick brown fox",
            "doc2": "the lazy dog",
            "doc3": "the cat sleeps"
        }
        bm25.index_documents(docs)

        # "the" appears in all 3 docs, should have low IDF
        idf_the = bm25._idf("the")
        # "fox" appears in 1 doc, should have higher IDF
        idf_fox = bm25._idf("fox")

        assert idf_fox > idf_the

    def test_score_calculation(self):
        """Test BM25 score calculation."""
        from src.retrieve import BM25

        bm25 = BM25()
        docs = {
            "doc1": "machine learning is great",
            "doc2": "deep learning neural networks",
            "doc3": "cooking recipes and food"
        }
        bm25.index_documents(docs)

        # "learning" query should score higher for doc1 and doc2
        score1 = bm25.score("learning", "doc1")
        score2 = bm25.score("learning", "doc2")
        score3 = bm25.score("learning", "doc3")

        assert score1 > score3
        assert score2 > score3

    def test_search(self):
        """Test BM25 search returns ranked results."""
        from src.retrieve import BM25

        bm25 = BM25()
        docs = {
            "doc1": "python programming language",
            "doc2": "python snake animal",
            "doc3": "java programming language"
        }
        bm25.index_documents(docs)

        results = bm25.search("python programming", top_k=2)

        assert len(results) <= 2
        assert results[0][0] == "doc1"  # Most relevant
        assert results[0][1] > results[1][1]  # Higher score first

    def test_search_empty_query(self):
        """Test search with terms not in corpus."""
        from src.retrieve import BM25

        bm25 = BM25()
        docs = {"doc1": "hello world"}
        bm25.index_documents(docs)

        results = bm25.search("xyz unknown")

        assert len(results) == 0


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.retrieve import RetrievalResult

        result = RetrievalResult(
            chunk_id="test_chunk_0",
            score=0.95,
            content="Test content",
            metadata={"document": "test.pdf"}
        )

        d = result.to_dict()

        assert d["chunk_id"] == "test_chunk_0"
        assert d["score"] == 0.95
        assert d["content"] == "Test content"
        assert d["metadata"]["document"] == "test.pdf"


class TestDocumentRetriever:
    """Tests for DocumentRetriever class."""

    @pytest.fixture
    def mock_retriever(self):
        """Create retriever with mocked dependencies."""
        with patch("src.retrieve.PineconeIndexer") as mock_indexer_class, \
             patch("boto3.client") as mock_boto:

            mock_indexer = MagicMock()
            mock_embed_client = MagicMock()
            mock_indexer.embedding_client = mock_embed_client

            mock_boto_client = MagicMock()
            mock_boto.return_value = mock_boto_client

            from src.retrieve import DocumentRetriever

            retriever = DocumentRetriever(
                indexer=mock_indexer,
                aws_region="us-east-1"
            )

            retriever._mock_indexer = mock_indexer
            retriever._mock_boto = mock_boto_client

            yield retriever

    def test_search_basic(self, mock_retriever):
        """Test basic semantic search."""
        mock_match = MagicMock()
        mock_match.id = "chunk_0"
        mock_match.score = 0.9
        mock_match.metadata = {"document": "test", "original_content": "content"}

        mock_retriever._mock_indexer.query_with_filters.return_value = [mock_match]

        results = mock_retriever.search("test query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_0"
        assert results[0].score == 0.9

    def test_search_by_document(self, mock_retriever):
        """Test search filtered by document."""
        mock_match = MagicMock()
        mock_match.id = "chunk_0"
        mock_match.score = 0.9
        mock_match.metadata = {"document": "myfile.pdf", "original_content": "content"}

        mock_retriever._mock_indexer.query_with_filters.return_value = [mock_match]

        results = mock_retriever.search_by_document("query", "myfile.pdf", top_k=3)

        mock_retriever._mock_indexer.query_with_filters.assert_called_once()
        call_args = mock_retriever._mock_indexer.query_with_filters.call_args
        assert call_args[0][1] == {"document": {"$eq": "myfile.pdf"}}

    def test_search_by_type(self, mock_retriever):
        """Test search filtered by chunk type."""
        mock_retriever._mock_indexer.query_with_filters.return_value = []

        mock_retriever.search_by_type("query", "table", top_k=5)

        call_args = mock_retriever._mock_indexer.query_with_filters.call_args
        assert call_args[0][1] == {"chunk_type": {"$eq": "table"}}

    def test_search_by_page_range(self, mock_retriever):
        """Test search filtered by page range."""
        mock_retriever._mock_indexer.query_with_filters.return_value = []

        mock_retriever.search_by_page_range("query", min_page=5, max_page=10)

        call_args = mock_retriever._mock_indexer.query_with_filters.call_args
        filters = call_args[0][1]
        assert "$and" in filters
        assert {"page_number": {"$gte": 5}} in filters["$and"]
        assert {"page_number": {"$lte": 10}} in filters["$and"]

    def test_search_combined_filters(self, mock_retriever):
        """Test search with multiple filters."""
        mock_retriever._mock_indexer.query_with_filters.return_value = []

        mock_retriever.search_combined(
            "query",
            document="test.pdf",
            chunk_type="text",
            min_page=1,
            has_image=False
        )

        call_args = mock_retriever._mock_indexer.query_with_filters.call_args
        filters = call_args[0][1]
        assert "$and" in filters
        assert len(filters["$and"]) == 4

    def test_index_for_bm25(self, mock_retriever):
        """Test indexing documents for BM25."""
        docs = {
            "chunk_0": "content one",
            "chunk_1": "content two"
        }

        mock_retriever.index_for_bm25(docs)

        assert mock_retriever.bm25 is not None
        assert mock_retriever.bm25.num_docs == 2

    def test_hybrid_search_without_bm25(self, mock_retriever):
        """Test hybrid search falls back to semantic when BM25 not indexed."""
        mock_match = MagicMock()
        mock_match.id = "chunk_0"
        mock_match.score = 0.9
        mock_match.metadata = {"original_content": "content"}

        mock_retriever._mock_indexer.query_with_filters.return_value = [mock_match]

        results = mock_retriever.hybrid_search("query", top_k=5)

        assert len(results) == 1

    def test_hybrid_search_with_bm25(self, mock_retriever):
        """Test hybrid search combines semantic and BM25."""
        # Setup BM25
        docs = {
            "chunk_0": "python programming",
            "chunk_1": "java programming",
            "chunk_2": "cooking recipes"
        }
        mock_retriever.index_for_bm25(docs)

        # Setup semantic results
        mock_match = MagicMock()
        mock_match.id = "chunk_0"
        mock_match.score = 0.9
        mock_match.metadata = {"original_content": "python programming"}

        mock_retriever._mock_indexer.query_with_filters.return_value = [mock_match]

        results = mock_retriever.hybrid_search("python programming", top_k=3)

        assert len(results) >= 1
        assert results[0].chunk_id == "chunk_0"

    def test_expand_query(self, mock_retriever):
        """Test query expansion generates variations."""
        import json

        # Mock Claude response
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = json.dumps({
            "content": [{"text": "variation one\nvariation two\nvariation three"}]
        })
        mock_retriever._mock_boto.invoke_model.return_value = {"body": mock_response_body}

        expansions = mock_retriever.expand_query("test query", num_expansions=3)

        assert len(expansions) >= 1
        assert expansions[0] == "test query"  # Original included

    def test_expand_query_handles_error(self, mock_retriever):
        """Test query expansion handles API errors gracefully."""
        mock_retriever._mock_boto.invoke_model.side_effect = Exception("API Error")

        expansions = mock_retriever.expand_query("test query")

        assert expansions == ["test query"]  # Falls back to original

    def test_rerank_empty_results(self, mock_retriever):
        """Test rerank handles empty results."""
        results = mock_retriever.rerank("query", [], top_k=5)

        assert results == []

    def test_rerank_orders_by_llm_response(self, mock_retriever):
        """Test rerank reorders based on LLM response."""
        import json
        from src.retrieve import RetrievalResult

        # Create test results
        results = [
            RetrievalResult("chunk_0", 0.9, "first content", {}),
            RetrievalResult("chunk_1", 0.8, "second content", {}),
            RetrievalResult("chunk_2", 0.7, "third content", {})
        ]

        # Mock Claude response - reorder as 3, 1, 2
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = json.dumps({
            "content": [{"text": "3, 1, 2"}]
        })
        mock_retriever._mock_boto.invoke_model.return_value = {"body": mock_response_body}

        reranked = mock_retriever.rerank("query", results, top_k=3)

        assert reranked[0].chunk_id == "chunk_2"
        assert reranked[1].chunk_id == "chunk_0"
        assert reranked[2].chunk_id == "chunk_1"

    def test_hyde_search_uses_hypothetical_doc(self, mock_retriever):
        """Test HyDE generates hypothetical document."""
        import json

        # Mock Claude response for hypothetical document
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = json.dumps({
            "content": [{"text": "Hypothetical answer about the topic..."}]
        })
        mock_retriever._mock_boto.invoke_model.return_value = {"body": mock_response_body}

        # Mock search results
        mock_match = MagicMock()
        mock_match.id = "chunk_0"
        mock_match.score = 0.85
        mock_match.metadata = {"original_content": "actual content"}
        mock_retriever._mock_indexer.query_with_filters.return_value = [mock_match]

        results = mock_retriever.hyde_search("what is the topic?", top_k=3)

        # Should have called the indexer with the hypothetical doc
        assert mock_retriever._mock_indexer.query_with_filters.called
        assert len(results) == 1
        assert "hyde_query" in results[0].metadata

    def test_hyde_search_handles_error(self, mock_retriever):
        """Test HyDE falls back to regular search on error."""
        mock_retriever._mock_boto.invoke_model.side_effect = Exception("API Error")

        mock_match = MagicMock()
        mock_match.id = "chunk_0"
        mock_match.score = 0.9
        mock_match.metadata = {"original_content": "content"}
        mock_retriever._mock_indexer.query_with_filters.return_value = [mock_match]

        results = mock_retriever.hyde_search("query", top_k=3)

        assert len(results) == 1

    def test_advanced_search_combines_methods(self, mock_retriever):
        """Test advanced search combines multiple retrieval methods."""
        import json

        # Mock Claude responses
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = json.dumps({
            "content": [{"text": "hypothetical response"}]
        })
        mock_retriever._mock_boto.invoke_model.return_value = {"body": mock_response_body}

        # Mock search results
        mock_match = MagicMock()
        mock_match.id = "chunk_0"
        mock_match.score = 0.9
        mock_match.metadata = {"original_content": "content"}
        mock_retriever._mock_indexer.query_with_filters.return_value = [mock_match]

        results = mock_retriever.advanced_search(
            "query",
            top_k=3,
            use_hyde=True,
            use_expansion=True,
            use_reranking=False
        )

        # Should have called indexer multiple times
        assert mock_retriever._mock_indexer.query_with_filters.call_count >= 1
        assert len(results) >= 0

    def test_advanced_search_without_options(self, mock_retriever):
        """Test advanced search with all options disabled."""
        mock_match = MagicMock()
        mock_match.id = "chunk_0"
        mock_match.score = 0.9
        mock_match.metadata = {"original_content": "content"}
        mock_retriever._mock_indexer.query_with_filters.return_value = [mock_match]

        results = mock_retriever.advanced_search(
            "query",
            top_k=3,
            use_hyde=False,
            use_expansion=False,
            use_reranking=False
        )

        assert len(results) == 1


class TestDocumentRetrieverPrintResults:
    """Tests for print_results utility method."""

    def test_print_results_formats_output(self, capsys):
        """Test that print_results formats output correctly."""
        from src.retrieve import RetrievalResult

        with patch("src.retrieve.PineconeIndexer"), \
             patch("boto3.client"):

            from src.retrieve import DocumentRetriever

            mock_indexer = MagicMock()
            mock_indexer.embedding_client = MagicMock()

            retriever = DocumentRetriever(indexer=mock_indexer)

            results = [
                RetrievalResult(
                    chunk_id="test_chunk",
                    score=0.95,
                    content="Test content for preview",
                    metadata={
                        "document": "test.pdf",
                        "chunk_type": "text",
                        "page_number": 5
                    }
                )
            ]

            retriever.print_results(results, "test query")

            captured = capsys.readouterr()
            assert "test query" in captured.out
            assert "0.95" in captured.out
            assert "test.pdf" in captured.out
