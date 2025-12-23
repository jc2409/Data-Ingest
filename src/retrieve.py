"""
Retrieval examples with metadata filtering
Demonstrates various filtered search patterns
"""
import os
from typing import List, Dict, Any, Optional
from .indexing import PineconeIndexer
from dotenv import load_dotenv

load_dotenv()


class DocumentRetriever:
    """
    Handles retrieval from Pinecone with various filtering strategies
    """

    def __init__(self, indexer: PineconeIndexer):
        self.indexer = indexer

    def search_all(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Basic semantic search without filters

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of matches
        """
        return self.indexer.query_with_filters(
            query_text=query,
            filters=None,
            top_k=top_k
        )

    def search_by_document(
        self,
        query: str,
        document: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific document

        Args:
            query: Search query
            document: Document name to filter by
            top_k: Number of results

        Returns:
            List of matches from specified document
        """
        filters = {"document": {"$eq": document}}
        return self.indexer.query_with_filters(query, filters, top_k)

    def search_by_chunk_type(
        self,
        query: str,
        chunk_type: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for specific content types

        Args:
            query: Search query
            chunk_type: Type to filter by ("text", "table", "image")
            top_k: Number of results

        Returns:
            List of matches of specified type
        """
        filters = {"chunk_type": {"$eq": chunk_type}}
        return self.indexer.query_with_filters(query, filters, top_k)

    def search_by_page_range(
        self,
        query: str,
        min_page: int,
        max_page: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search within a page range

        Args:
            query: Search query
            min_page: Minimum page number (inclusive)
            max_page: Maximum page number (inclusive)
            top_k: Number of results

        Returns:
            List of matches within page range
        """
        filters = {
            "$and": [
                {"page_number": {"$gte": min_page}},
                {"page_number": {"$lte": max_page}}
            ]
        }
        return self.indexer.query_with_filters(query, filters, top_k)

    def search_images_only(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search only chunks with images

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of image chunk matches
        """
        filters = {"has_image": {"$eq": True}}
        return self.indexer.query_with_filters(query, filters, top_k)

    def search_tables_only(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search only table chunks

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of table chunk matches
        """
        return self.search_by_chunk_type(query, "table", top_k)

    def search_by_section(
        self,
        query: str,
        section_title: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific section

        Args:
            query: Search query
            section_title: Section title to filter by
            top_k: Number of results

        Returns:
            List of matches from specified section
        """
        filters = {"section_title": {"$eq": section_title}}
        return self.indexer.query_with_filters(query, filters, top_k)

    def search_with_combined_filters(
        self,
        query: str,
        document: Optional[str] = None,
        chunk_type: Optional[str] = None,
        min_page: Optional[int] = None,
        max_page: Optional[int] = None,
        has_image: Optional[bool] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search with multiple combined filters

        Args:
            query: Search query
            document: Filter by document name
            chunk_type: Filter by chunk type
            min_page: Minimum page number
            max_page: Maximum page number
            has_image: Filter by image presence
            top_k: Number of results

        Returns:
            List of matches meeting all criteria
        """
        filter_conditions = []

        if document:
            filter_conditions.append({"document": {"$eq": document}})

        if chunk_type:
            filter_conditions.append({"chunk_type": {"$eq": chunk_type}})

        if min_page is not None:
            filter_conditions.append({"page_number": {"$gte": min_page}})

        if max_page is not None:
            filter_conditions.append({"page_number": {"$lte": max_page}})

        if has_image is not None:
            filter_conditions.append({"has_image": {"$eq": has_image}})

        # Combine all filters with AND
        filters = {"$and": filter_conditions} if filter_conditions else None

        return self.indexer.query_with_filters(query, filters, top_k)

    def print_results(self, results: List[Any], query: str):
        """
        Pretty print search results

        Args:
            results: List of search results
            query: Original query for display
        """
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results")
        print("=" * 80)

        for i, match in enumerate(results, 1):
            metadata = match.metadata
            print(f"\n{i}. Score: {match.score:.4f}")
            print(f"   Document: {metadata.get('document', 'N/A')}")
            print(f"   Chunk ID: {metadata.get('chunk_id', 'N/A')}")
            print(f"   Type: {metadata.get('chunk_type', 'N/A')}")
            print(f"   Page: {metadata.get('page_number', 'N/A')}")

            if metadata.get('section_title'):
                print(f"   Section: {metadata.get('section_title')}")

            if metadata.get('has_image'):
                print(f"   Contains Image: Yes")
                if metadata.get('image_description'):
                    desc = metadata.get('image_description')[:100]
                    print(f"   Image Desc: {desc}...")

            # Show content preview
            content = metadata.get('original_content', '')
            preview = content[:200].replace('\n', ' ')
            print(f"   Preview: {preview}...")


if __name__ == "__main__":
    # Initialize indexer and retriever
    indexer = PineconeIndexer(
        index_name=os.getenv("PINECONE_INDEX_NAME", "data-ingest"),
        embedding_dimensions=1024,
        normalize_embeddings=True,
        aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        metric="cosine"
    )

    retriever = DocumentRetriever(indexer)

    print("=" * 80)
    print("RETRIEVAL EXAMPLES WITH METADATA FILTERING")
    print("=" * 80)

    # Example 1: Basic semantic search
    print("\n\n1. BASIC SEMANTIC SEARCH (no filters)")
    print("-" * 80)
    results = retriever.search_all("blood transfusion protocol", top_k=3)
    retriever.print_results(results, "blood transfusion protocol")

    # Example 2: Search specific document
    print("\n\n2. SEARCH WITHIN SPECIFIC DOCUMENT")
    print("-" * 80)
    results = retriever.search_by_document(
        query="emergency procedures",
        document="Module 1 - Introduction to Emergency Fresh Whole Blood Transfusion V1.6",
        top_k=3
    )
    retriever.print_results(results, "emergency procedures [document filter]")

    # Example 3: Search tables only
    print("\n\n3. SEARCH TABLES ONLY")
    print("-" * 80)
    results = retriever.search_tables_only("dosage guidelines", top_k=3)
    retriever.print_results(results, "dosage guidelines [tables only]")

    # Example 4: Search by page range
    print("\n\n4. SEARCH WITHIN PAGE RANGE")
    print("-" * 80)
    results = retriever.search_by_page_range(
        query="safety precautions",
        min_page=5,
        max_page=15,
        top_k=3
    )
    retriever.print_results(results, "safety precautions [pages 5-15]")

    # Example 5: Search images only
    print("\n\n5. SEARCH IMAGES ONLY")
    print("-" * 80)
    results = retriever.search_images_only("medical diagram", top_k=3)
    retriever.print_results(results, "medical diagram [images only]")

    # Example 6: Combined filters
    print("\n\n6. COMBINED FILTERS")
    print("-" * 80)
    results = retriever.search_with_combined_filters(
        query="treatment protocol",
        chunk_type="text",
        min_page=1,
        max_page=20,
        has_image=False,
        top_k=3
    )
    retriever.print_results(
        results,
        "treatment protocol [text chunks, pages 1-20, no images]"
    )

    # Example 7: Custom filter - high token count chunks
    print("\n\n7. CUSTOM FILTER - LARGE CHUNKS")
    print("-" * 80)
    custom_filters = {"token_count": {"$gte": 500}}
    results = indexer.query_with_filters(
        query_text="comprehensive overview",
        filters=custom_filters,
        top_k=3
    )
    retriever.print_results(results, "comprehensive overview [large chunks only]")

    print("\n\n" + "=" * 80)
    print("RETRIEVAL EXAMPLES COMPLETE")
    print("=" * 80)
