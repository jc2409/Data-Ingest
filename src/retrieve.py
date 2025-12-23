"""
Advanced retrieval with metadata filtering and quality improvements.

Features:
- Basic filtered search
- Hybrid search (semantic + BM25)
- LLM reranking
- HyDE (Hypothetical Document Embeddings)
- Query expansion
"""
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import boto3
from dotenv import load_dotenv

from .indexing import PineconeIndexer

load_dotenv()


@dataclass
class RetrievalResult:
    """Represents a single retrieval result with score and metadata."""
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata
        }


class BM25:
    """
    BM25 ranking algorithm for keyword-based retrieval.

    BM25 is a bag-of-words retrieval function that ranks documents
    based on query term frequency and inverse document frequency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with tuning parameters.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Document length normalization (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.num_docs: int = 0
        self.documents: Dict[str, str] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def index_documents(self, documents: Dict[str, str]):
        """
        Index documents for BM25 retrieval.

        Args:
            documents: Dictionary mapping doc_id to content
        """
        self.documents = documents
        self.num_docs = len(documents)
        total_length = 0

        for doc_id, content in documents.items():
            tokens = self._tokenize(content)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Count term frequencies in this document
            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            self.term_freqs[doc_id] = dict(term_freq)

            # Count document frequencies
            for token in set(tokens):
                self.doc_freqs[token] += 1

        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_id: str) -> float:
        """
        Calculate BM25 score for a query-document pair.

        Args:
            query: Search query
            doc_id: Document identifier

        Returns:
            BM25 score (higher is better)
        """
        query_tokens = self._tokenize(query)
        doc_length = self.doc_lengths.get(doc_id, 0)
        term_freqs = self.term_freqs.get(doc_id, {})

        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search indexed documents with BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        scores = []
        for doc_id in self.documents:
            score = self.score(query, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DocumentRetriever:
    """
    Advanced document retrieval with multiple search strategies.

    Features:
    - Basic semantic search with metadata filters
    - Hybrid search (semantic + BM25)
    - LLM reranking
    - HyDE (Hypothetical Document Embeddings)
    - Query expansion
    """

    def __init__(
        self,
        indexer: PineconeIndexer,
        aws_region: str = None,
        claude_model: str = None,
        max_workers: int = 3
    ):
        """
        Initialize document retriever.

        Args:
            indexer: PineconeIndexer instance
            aws_region: AWS region for Bedrock
            claude_model: Claude model ID for reranking and HyDE
            max_workers: Max parallel workers for LLM calls
        """
        self.indexer = indexer
        self.embedding_client = indexer.embedding_client
        self.max_workers = max_workers

        # Initialize Bedrock client for Claude
        aws_region = aws_region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region
        )
        self.claude_model = claude_model or os.getenv("AWS_BEDROCK_CLAUDE_MODEL")

        # BM25 index (populated when needed)
        self.bm25: Optional[BM25] = None
        self._bm25_docs: Dict[str, str] = {}

    def _call_claude(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Claude via AWS Bedrock."""
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }

        response = self.bedrock_client.invoke_model(
            modelId=self.claude_model,
            body=json.dumps(native_request)
        )

        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"].strip()

    def _to_results(self, matches: List[Any]) -> List[RetrievalResult]:
        """Convert Pinecone matches to RetrievalResult objects."""
        results = []
        for match in matches:
            results.append(RetrievalResult(
                chunk_id=match.id,
                score=match.score,
                content=match.metadata.get("original_content", ""),
                metadata=dict(match.metadata)
            ))
        return results

    # =========================================================================
    # Basic Filtered Search
    # =========================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Basic semantic search with optional filters.

        Args:
            query: Search query
            top_k: Number of results
            filters: Pinecone filter dictionary

        Returns:
            List of retrieval results
        """
        matches = self.indexer.query_with_filters(query, filters, top_k)
        return self._to_results(matches)

    def search_by_document(
        self,
        query: str,
        document: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Search within a specific document."""
        return self.search(query, top_k, {"document": {"$eq": document}})

    def search_by_type(
        self,
        query: str,
        chunk_type: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Search for specific content types (text, table, image)."""
        return self.search(query, top_k, {"chunk_type": {"$eq": chunk_type}})

    def search_by_page_range(
        self,
        query: str,
        min_page: int,
        max_page: int,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Search within a page range."""
        filters = {
            "$and": [
                {"page_number": {"$gte": min_page}},
                {"page_number": {"$lte": max_page}}
            ]
        }
        return self.search(query, top_k, filters)

    def search_combined(
        self,
        query: str,
        document: Optional[str] = None,
        chunk_type: Optional[str] = None,
        min_page: Optional[int] = None,
        max_page: Optional[int] = None,
        has_image: Optional[bool] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Search with multiple combined filters."""
        conditions = []
        if document:
            conditions.append({"document": {"$eq": document}})
        if chunk_type:
            conditions.append({"chunk_type": {"$eq": chunk_type}})
        if min_page is not None:
            conditions.append({"page_number": {"$gte": min_page}})
        if max_page is not None:
            conditions.append({"page_number": {"$lte": max_page}})
        if has_image is not None:
            conditions.append({"has_image": {"$eq": has_image}})

        filters = {"$and": conditions} if conditions else None
        return self.search(query, top_k, filters)

    # =========================================================================
    # Hybrid Search (Semantic + BM25)
    # =========================================================================

    def index_for_bm25(self, documents: Dict[str, str]):
        """
        Index documents for BM25 search.

        Args:
            documents: Dictionary mapping chunk_id to content
        """
        self.bm25 = BM25()
        self._bm25_docs = documents
        self.bm25.index_documents(documents)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining semantic similarity and BM25 keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to combine rankings.

        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores (0-1)
            filters: Optional Pinecone filters

        Returns:
            Combined ranked results
        """
        semantic_results = self.search(query, top_k * 2, filters)

        if self.bm25 is None:
            return semantic_results[:top_k]

        bm25_scores = self.bm25.search(query, top_k * 2)

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = defaultdict(float)
        k = 60

        for rank, result in enumerate(semantic_results):
            rrf_scores[result.chunk_id] += semantic_weight * (1 / (k + rank + 1))

        bm25_weight = 1 - semantic_weight
        for rank, (doc_id, _) in enumerate(bm25_scores):
            rrf_scores[doc_id] += bm25_weight * (1 / (k + rank + 1))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        result_map = {r.chunk_id: r for r in semantic_results}

        for chunk_id in sorted_ids[:top_k]:
            if chunk_id in result_map:
                result = result_map[chunk_id]
                result.score = rrf_scores[chunk_id]
                results.append(result)
            elif chunk_id in self._bm25_docs:
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    score=rrf_scores[chunk_id],
                    content=self._bm25_docs[chunk_id],
                    metadata={"source": "bm25"}
                ))

        return results

    # =========================================================================
    # LLM Reranking
    # =========================================================================

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank results using Claude to score relevance.

        Args:
            query: Original search query
            results: Initial retrieval results
            top_k: Number of results to return after reranking

        Returns:
            Reranked results
        """
        if not results:
            return []

        docs_text = ""
        for i, result in enumerate(results):
            content = result.content[:500]
            docs_text += f"\n[Document {i+1}]\n{content}\n"

        prompt = f"""Given the query and documents below, rank the documents by relevance.
Return ONLY a comma-separated list of document numbers in order of relevance (most relevant first).

Query: {query}

Documents:
{docs_text}

Ranking (comma-separated numbers):"""

        try:
            response = self._call_claude(prompt, max_tokens=100)
            numbers = re.findall(r'\d+', response)
            ranking = [int(n) - 1 for n in numbers if 0 < int(n) <= len(results)]

            seen = set()
            reranked = []
            for idx in ranking:
                if idx not in seen:
                    seen.add(idx)
                    reranked.append(results[idx])

            for idx, result in enumerate(results):
                if idx not in seen and len(reranked) < top_k:
                    reranked.append(results[idx])

            return reranked[:top_k]

        except Exception as e:
            print(f"Reranking failed: {e}")
            return results[:top_k]

    # =========================================================================
    # HyDE (Hypothetical Document Embeddings)
    # =========================================================================

    def hyde_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        HyDE: Generate a hypothetical document, then search with its embedding.

        Args:
            query: Search query
            top_k: Number of results
            filters: Optional Pinecone filters

        Returns:
            Search results
        """
        prompt = f"""Write a short, detailed paragraph that would be a perfect answer to this question.
Write the content directly as if from a reference document. No preamble.

Question: {query}

Answer:"""

        try:
            hypothetical_doc = self._call_claude(prompt, max_tokens=200)
            print(f"\nHyDE Answer: '{hypothetical_doc}'")
            matches = self.indexer.query_with_filters(hypothetical_doc, filters, top_k)
            results = self._to_results(matches)

            for result in results:
                result.metadata["hyde_query"] = hypothetical_doc

            return results

        except Exception as e:
            print(f"HyDE failed: {e}")
            return self.search(query, top_k, filters)

    # =========================================================================
    # Query Expansion
    # =========================================================================

    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Generate query variations to improve recall.

        Args:
            query: Original query
            num_expansions: Number of variations to generate

        Returns:
            List of query variations including original
        """
        prompt = f"""Generate {num_expansions} different ways to phrase this search query.
Each variation should capture the same intent but use different words.
Return ONLY the variations, one per line, no numbering.

Original: {query}

Variations:"""

        try:
            response = self._call_claude(prompt, max_tokens=200)
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            print(f"\nQuery variations:\n")
            for v in variations:
                print(f"- {v}")
            return [query] + variations[:num_expansions]

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [query]

    def search_expanded(
        self,
        query: str,
        top_k: int = 5,
        num_expansions: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search with query expansion for improved recall.

        Args:
            query: Original query
            top_k: Number of final results
            num_expansions: Number of query variations
            filters: Optional Pinecone filters

        Returns:
            Combined and deduplicated results
        """
        queries = self.expand_query(query, num_expansions)

        all_results: Dict[str, RetrievalResult] = {}
        chunk_scores: Dict[str, List[float]] = defaultdict(list)

        def search_query(q: str) -> List[RetrievalResult]:
            return self.search(q, top_k, filters)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(search_query, q): q for q in queries}
            for future in as_completed(futures):
                for result in future.result():
                    chunk_scores[result.chunk_id].append(result.score)
                    if result.chunk_id not in all_results:
                        all_results[result.chunk_id] = result

        final_results = []
        for chunk_id, result in all_results.items():
            scores = chunk_scores[chunk_id]
            result.score = sum(scores) / len(scores)
            final_results.append(result)

        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]

    # =========================================================================
    # Combined Advanced Search
    # =========================================================================

    def advanced_search(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_expansion: bool = True,
        use_reranking: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Full advanced search pipeline combining multiple techniques.

        Pipeline:
        1. Standard semantic search
        2. HyDE search (if enabled)
        3. Query expansion (if enabled)
        4. Combine and deduplicate
        5. LLM reranking (if enabled)

        Args:
            query: Search query
            top_k: Number of results
            use_hyde: Whether to use HyDE
            use_expansion: Whether to use query expansion
            use_reranking: Whether to use LLM reranking
            filters: Optional Pinecone filters

        Returns:
            Final ranked results
        """
        all_results: Dict[str, RetrievalResult] = {}
        chunk_scores: Dict[str, List[float]] = defaultdict(list)

        # Standard search
        for result in self.search(query, top_k * 2, filters):
            chunk_scores[result.chunk_id].append(result.score)
            all_results[result.chunk_id] = result

        # HyDE search
        if use_hyde:
            for result in self.hyde_search(query, top_k * 2, filters):
                chunk_scores[result.chunk_id].append(result.score)
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result

        # Query expansion
        if use_expansion:
            for result in self.search_expanded(query, top_k * 2, 2, filters):
                chunk_scores[result.chunk_id].append(result.score)
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result

        # Combine scores
        combined = []
        for chunk_id, result in all_results.items():
            scores = chunk_scores[chunk_id]
            result.score = sum(scores) / len(scores)
            combined.append(result)

        combined.sort(key=lambda x: x.score, reverse=True)

        # Rerank
        if use_reranking:
            return self.rerank(query, combined[:top_k * 2], top_k)

        return combined[:top_k]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def print_results(self, results: List[RetrievalResult], query: str, show_score: bool = True):
        """Pretty print search results."""
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results")
        print("=" * 80)

        for rank, result in enumerate(results, 1):
            if show_score:
                print(f"\n{rank}. Score: {result.score:.4f}")
            else:
                print(f"\nRank {rank}")
            print(f"   Chunk: {result.chunk_id}")
            print(f"   Document: {result.metadata.get('document', 'N/A')}")
            print(f"   Type: {result.metadata.get('chunk_type', 'N/A')}")
            print(f"   Page: {result.metadata.get('page_number', 'N/A')}")

            if result.metadata.get('section_title'):
                print(f"   Section: {result.metadata['section_title']}")

            preview = result.content[:200].replace('\n', ' ')
            print(f"   Preview: {preview}...")


if __name__ == "__main__":
    indexer = PineconeIndexer(
        index_name=os.getenv("PINECONE_INDEX_NAME", "data-ingest"),
        embedding_dimensions=1024,
        aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )

    retriever = DocumentRetriever(indexer)

    print("=" * 80)
    print("RETRIEVAL EXAMPLES")
    print("=" * 80)

    query = input("Enter your search query: ")

    # Basic search
    print("\n1. BASIC SEMANTIC SEARCH")
    print("-" * 40)
    results = retriever.search(query, top_k=3)
    retriever.print_results(results, query)

    # HyDE search
    print("\n2. HyDE SEARCH")
    print("-" * 40)
    results = retriever.hyde_search(query, top_k=3)
    retriever.print_results(results, query)

    # Query expansion
    print("\n3. QUERY EXPANSION SEARCH")
    print("-" * 40)
    results = retriever.search_expanded(query, top_k=3)
    retriever.print_results(results, query)

    # Advanced search
    print("\n4. ADVANCED SEARCH (HyDE + Expansion + Reranking)")
    print("-" * 40)
    results = retriever.advanced_search(query, top_k=3)
    retriever.print_results(results, query, show_score=False)

    print("\n" + "=" * 80)
