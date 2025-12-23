"""
Index contextualized chunks into Pinecone vector database
Preserves all metadata for filtered retrieval
"""
import glob
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from .embedding import EmbeddingClient

load_dotenv()


class PineconeIndexer:
    """
    Index chunks with embeddings into Pinecone

    Features:
    - Generates embeddings using Amazon Titan
    - Stores full metadata with each vector for filtered search
    - Handles large metadata (images stored as separate field)
    - Batch upsert for efficiency
    """

    def __init__(
        self,
        index_name: str = os.getenv("PINECONE_INDEX_NAME"),
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        embedding_dimensions: int = 1024,
        normalize_embeddings: bool = True,
        aws_region: str = "us-east-1",
        metric: str = "cosine",
        batch_size: int = 100
    ):
        """
        Initialize Pinecone indexer

        Args:
            index_name: Name of Pinecone index
            embedding_model: Amazon Titan model ID
            embedding_dimensions: Embedding dimensions (256, 512, or 1024)
            normalize_embeddings: Whether to normalize embeddings
            aws_region: AWS region for Bedrock
            metric: Distance metric (cosine, euclidean, or dotproduct)
            batch_size: Number of vectors to upsert at once
        """
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        self.embedding_dimensions = embedding_dimensions
        self.batch_size = batch_size

        # Shared embedding client with caching
        self.embedding_client = EmbeddingClient(
            aws_region=aws_region,
            model_id=embedding_model,
            dimensions=embedding_dimensions,
            normalize=normalize_embeddings
        )

        # Create or connect to index
        self._setup_index(metric)

    def _setup_index(self, metric: str):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimensions,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
            print(f"Index '{self.index_name}' created successfully")
        else:
            print(f"Using existing index: {self.index_name}")

        # Connect to index
        self.index = self.pc.Index(self.index_name)

    def prepare_vector(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a chunk for Pinecone upsert with filterable metadata

        Args:
            chunk: Chunk dictionary from contextual chunking

        Returns:
            Dictionary with id, values, and metadata for Pinecone
        """
        # Generate embedding from contextualized content
        embedding = self.embedding_client.get_embedding_list(chunk['contextualized_content'])

        # Prepare metadata for filtered search
        # All fields here are filterable in Pinecone queries
        metadata = {
            # Core identifiers
            "chunk_id": chunk['chunk_id'],
            "document": chunk['metadata'].get('document', ''),
            "chunk_index": int(chunk['metadata'].get('chunk_index', 0)),

            # Content metadata (for filtering)
            "chunk_type": chunk['metadata'].get('chunk_type', 'text'),  # text, table, image
            "section_title": chunk['metadata'].get('section_title', '') or '',  # Ensure string
            "page_number": int(chunk['metadata'].get('page_number') or 0),

            # Statistics
            "element_count": int(chunk['metadata'].get('element_count', 0)),
            "token_count": int(chunk.get('token_count', 0)),

            # Image-specific metadata
            "has_image": bool(chunk['metadata'].get('has_image', False)),
        }

        # Add image description if present (searchable in semantic search)
        if chunk['metadata'].get('image_description'):
            metadata['image_description'] = chunk['metadata']['image_description'][:1000]  # Truncate if needed

        # Add filetype for images
        if chunk['metadata'].get('filetype'):
            metadata['filetype'] = chunk['metadata']['filetype']

        # Store original content (truncated to fit Pinecone limits)
        # Pinecone metadata has size limits (~40KB per vector)
        metadata['original_content'] = chunk['original_content'][:2000]  # First 2000 chars

        # Note: image_base64 is NOT stored in Pinecone metadata due to size
        # Instead, retrieve original chunk from JSON using chunk_id if needed

        return {
            "id": chunk['chunk_id'],
            "values": embedding,
            "metadata": metadata
        }

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index a list of chunks into Pinecone

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary with indexing statistics
        """
        print(f"Indexing {len(chunks)} chunks...")

        vectors = []
        for chunk in tqdm(chunks, desc="Preparing vectors"):
            vector = self.prepare_vector(chunk)
            vectors.append(vector)

        # Upsert in batches
        total_upserted = 0
        print("Upserting vectors to Pinecone...")
        for i in tqdm(range(0, len(vectors), self.batch_size), desc="Upserting batches"):
            batch = vectors[i:i + self.batch_size]
            self.index.upsert(vectors=batch)
            total_upserted += len(batch)

        print(f"Successfully indexed {total_upserted} chunks")

        # Get index stats
        stats = self.index.describe_index_stats()
        return {
            "upserted": total_upserted,
            "total_vectors": stats.total_vector_count,
            "dimensions": stats.dimension
        }

    def index_from_file(self, chunk_file: str) -> Dict[str, int]:
        """
        Load chunks from JSON file and index them

        Args:
            chunk_file: Path to JSON file with chunks

        Returns:
            Indexing statistics
        """
        print(f"Loading chunks from: {chunk_file}")
        with open(chunk_file, 'r') as f:
            chunks = json.load(f)

        return self.index_chunks(chunks)

    def index_from_directory(self, chunks_dir: str) -> Dict[str, int]:
        """
        Index all chunk files in a directory

        Args:
            chunks_dir: Directory containing chunk JSON files

        Returns:
            Indexing statistics
        """
        chunk_files = glob.glob(os.path.join(chunks_dir, "*_chunks.json"))
        print(f"Found {len(chunk_files)} chunk files")

        all_chunks = []
        for chunk_file in chunk_files:
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)

        print(f"Total chunks to index: {len(all_chunks)}")
        return self.index_chunks(all_chunks)

    def query_with_filters(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the index with optional metadata filters

        Args:
            query_text: Text to search for
            filters: Pinecone filter dictionary (see examples below)
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results

        Returns:
            List of matching results with scores and metadata

        Example filters:
            # Filter by document
            {"document": {"$eq": "module1.pdf"}}

            # Filter by chunk type
            {"chunk_type": {"$eq": "table"}}

            # Filter by page range
            {"page_number": {"$gte": 5, "$lte": 10}}

            # Filter images only
            {"has_image": {"$eq": True}}

            # Combine multiple filters with $and
            {"$and": [
                {"document": {"$eq": "module1.pdf"}},
                {"chunk_type": {"$eq": "text"}},
                {"page_number": {"$gte": 1}}
            ]}
        """
        # Generate query embedding
        query_embedding = self.embedding_client.get_embedding_list(query_text)

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            filter=filters,
            top_k=top_k,
            include_metadata=include_metadata
        )

        return results.matches

    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics"""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimensions": stats.dimension,
            "index_fullness": stats.index_fullness
        }

    def delete_all_vectors(self):
        """Delete all vectors from index (keeps index structure)"""
        print(f"Deleting all vectors from index: {self.index_name}")
        self.index.delete(delete_all=True)
        print("All vectors deleted")

    def delete_index(self):
        """Delete the entire index (use with caution!)"""
        print(f"Deleting index: {self.index_name}")
        self.pc.delete_index(self.index_name)
        print("Index deleted")


if __name__ == "__main__":
    # Initialize indexer
    indexer = PineconeIndexer(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding_dimensions=1024,  # Match your chunking settings
        normalize_embeddings=True,
        aws_region=os.getenv("AWS_DEFAULT_REGION"),
        metric="cosine",  # Best for normalized embeddings
        batch_size=100
    )

    # Index all chunks from directory
    chunks_dir = os.path.join(os.getcwd(), "dataset/chunks")
    stats = indexer.index_from_directory(chunks_dir)

    print("\n" + "="*60)
    print("Indexing Complete!")
    print(f"  Total vectors in index: {stats['total_vectors']}")
    print(f"  Dimensions: {stats['dimensions']}")
    print("="*60)

    # Example: Query with filters
    print("\n" + "="*60)
    print("Example Filtered Search:")
    print("="*60)

    # Search for tables only
    results = indexer.query_with_filters(
        query_text="blood transfusion protocol",
        filters={"chunk_type": {"$eq": "table"}},
        top_k=3
    )

    print(f"\nFound {len(results)} table chunks:")
    for i, match in enumerate(results, 1):
        print(f"\n{i}. Score: {match.score:.4f}")
        print(f"   Document: {match.metadata.get('document')}")
        print(f"   Page: {match.metadata.get('page_number')}")
        print(f"   Type: {match.metadata.get('chunk_type')}")
