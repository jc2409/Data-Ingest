import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AzureOpenAI
import boto3
from botocore.exceptions import ClientError
import tiktoken
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Chunk:
    """Represents a chunk with context"""
    chunk_id: str
    original_content: str
    contextualized_content: str
    metadata: Dict[str, Any]
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "original_content": self.original_content,
            "contextualized_content": self.contextualized_content,
            "metadata": self.metadata,
            "token_count": self.token_count
        }


class ContextualChunker:
    """
    Implements semantic chunking strategy:
    - Embedding-based semantic grouping using text-embedding-3-large (Azure OpenAI)
    - Automatic topic boundary detection via cosine similarity
    - LLM-enhanced context generation using Claude on AWS Bedrock
    - Rich metadata preservation
    """

    def __init__(
        self,
        azure_endpoint: str,
        azure_api_key: str,
        api_version: str = "2025-01-01-preview",
        embedding_model: str = "text-embedding-3-large",
        aws_region: str = "us-east-1",
        claude_model: str = "arn:aws:bedrock:us-east-1:522946686627:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 400,
        max_chunk_size: int = 1200,
        max_workers: int = 5
    ):
        # Azure OpenAI client for embeddings
        self.azure_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version
        )
        self.embedding_model = embedding_model

        # AWS Bedrock client for Claude chat completions
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region
        )
        self.claude_model = claude_model

        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_workers = max_workers

        # Initialize tokenizer (using cl100k_base for consistency)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using Azure OpenAI text-embedding-3-large

        Args:
            text: Text to embed

        Returns:
            numpy array of embedding vector
        """
        try:
            response = self.azure_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(3072)  # text-embedding-3-large dimension

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def load_unstructured_elements(self, json_path: str) -> List[Dict[str, Any]]:
        """Load parsed elements from Unstructured JSON output"""
        with open(json_path, 'r') as f:
            return json.load(f)

    def group_elements_semantically(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group elements using semantic similarity with embeddings.

        Strategy:
        - Generate embeddings for each element using text-embedding-3-large
        - Calculate cosine similarity between consecutive elements
        - Create chunk boundaries when similarity drops below threshold
        - Keep tables as separate chunks (self-contained units)
        - Respect min/max chunk size constraints
        """
        # Process elements: generate descriptions for images, filter empty text elements
        valid_elements = []
        print("  Processing elements (generating image descriptions if present)...")

        for element in tqdm(elements, desc="  Processing elements", leave=False):
            element_type = element.get("type", "")

            # Handle Image elements - generate descriptions
            if element_type == "Image":
                # Get base64 image data
                image_base64 = element.get("metadata", {}).get("image_base64")
                if image_base64:
                    # Get media type from metadata and normalize to supported formats
                    filetype = element.get("metadata", {}).get("filetype", "png")

                    # Map file extensions to Claude-supported media types
                    media_type_map = {
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "png": "image/png",
                        "gif": "image/gif",
                        "webp": "image/webp",
                        "image/jpg": "image/jpeg",
                        "image/jpeg": "image/jpeg",
                        "image/png": "image/png",
                        "image/gif": "image/gif",
                        "image/webp": "image/webp"
                    }

                    # Normalize and map to supported format
                    filetype_lower = filetype.lower().strip()
                    media_type = media_type_map.get(filetype_lower, "image/png")  # Default to PNG

                    # Generate description using Claude Vision
                    description = self.generate_image_description(image_base64, media_type)

                    # Add description as text so it can be embedded and chunked
                    element["text"] = f"[IMAGE] {description}"
                    element["metadata"]["image_description"] = description
                    element["metadata"]["has_image"] = True
                    valid_elements.append(element)

            # Handle text elements - only keep if they have content
            elif element.get("text", "").strip():
                valid_elements.append(element)

        if not valid_elements:
            return []

        print("  Generating embeddings for semantic chunking...")

        # Generate embeddings for all elements (with progress bar)
        element_embeddings = []
        for element in tqdm(valid_elements, desc="  Creating embeddings", leave=False):
            embedding = self.get_embedding(element.get("text", ""))
            element_embeddings.append(embedding)

        chunks = []
        current_chunk = {
            "elements": [],
            "tokens": 0,
            "metadata": {},
            "embedding": None
        }
        current_section_title = None

        for i, element in enumerate(valid_elements):
            element_type = element.get("type", "")
            element_text = element.get("text", "")
            element_tokens = self.count_tokens(element_text)
            element_embedding = element_embeddings[i]

            # Extract metadata
            metadata = element.get("metadata", {})
            page_number = metadata.get("page_number")

            # Track section titles for context
            if element_type == "Title":
                current_section_title = element_text

            # Images should be their own chunks (with generated descriptions)
            if element_type == "Image":
                # Finish current chunk if it has content
                if current_chunk["elements"]:
                    chunks.append(self._finalize_chunk(current_chunk))
                    current_chunk = {"elements": [], "tokens": 0, "metadata": {}, "embedding": None}

                # Add image as standalone chunk with base64 data preserved
                image_metadata = {
                    "section_title": current_section_title,
                    "page_number": page_number,
                    "chunk_type": "image",
                    "has_image": True,
                    "image_base64": metadata.get("image_base64"),  # Preserve base64 for retrieval
                    "image_description": metadata.get("image_description"),
                    "filetype": metadata.get("filetype")
                }

                chunks.append({
                    "elements": [element],
                    "tokens": element_tokens,
                    "metadata": image_metadata
                })
                continue

            # Tables should be their own chunks (usually complete info)
            if element_type == "Table":
                # Finish current chunk if it has content
                if current_chunk["elements"]:
                    chunks.append(self._finalize_chunk(current_chunk))
                    current_chunk = {"elements": [], "tokens": 0, "metadata": {}, "embedding": None}

                # Add table as standalone chunk
                chunks.append({
                    "elements": [element],
                    "tokens": element_tokens,
                    "metadata": {
                        "section_title": current_section_title,
                        "page_number": page_number,
                        "chunk_type": "table"
                    }
                })
                continue

            # Determine if we should start a new chunk based on semantic similarity
            should_split = False

            if current_chunk["elements"]:
                # Calculate similarity with current chunk
                chunk_embedding = current_chunk["embedding"]
                similarity = self.cosine_similarity(chunk_embedding, element_embedding)

                # Split if similarity is below threshold (topic change detected)
                if similarity < self.similarity_threshold:
                    should_split = True

                # Force split if max size exceeded
                if current_chunk["tokens"] + element_tokens > self.max_chunk_size:
                    should_split = True

                # Don't split if below min size (unless similarity is very low)
                if current_chunk["tokens"] < self.min_chunk_size and similarity >= 0.5:
                    should_split = False

            if should_split:
                # Save current chunk
                chunks.append(self._finalize_chunk(current_chunk))

                # Start new chunk
                current_chunk = {
                    "elements": [element],
                    "tokens": element_tokens,
                    "metadata": {
                        "section_title": current_section_title,
                        "page_number": page_number
                    },
                    "embedding": element_embedding
                }
            else:
                # Add to current chunk
                current_chunk["elements"].append(element)
                current_chunk["tokens"] += element_tokens

                # Update chunk embedding (average of all element embeddings)
                if current_chunk["embedding"] is None:
                    current_chunk["embedding"] = element_embedding
                else:
                    # Running average of embeddings
                    n = len(current_chunk["elements"])
                    current_chunk["embedding"] = (
                        (current_chunk["embedding"] * (n - 1) + element_embedding) / n
                    )

                if not current_chunk["metadata"]:
                    current_chunk["metadata"] = {
                        "section_title": current_section_title,
                        "page_number": page_number
                    }

        # Add final chunk
        if current_chunk["elements"]:
            chunks.append(self._finalize_chunk(current_chunk))

        return chunks

    def _finalize_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Remove embedding from chunk before returning (not needed in output)"""
        return {
            "elements": chunk["elements"],
            "tokens": chunk["tokens"],
            "metadata": chunk["metadata"]
        }

    def combine_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Combine elements into a single text string, using HTML for tables when available"""
        texts = []
        for e in chunk["elements"]:
            # For tables, prefer HTML representation if available for better structure preservation
            if e.get("type") == "Table":
                html_text = e.get("metadata", {}).get("text_as_html")
                if html_text:
                    texts.append(html_text)
                else:
                    # Fall back to plain text if HTML not available
                    texts.append(e.get("text", ""))
            else:
                texts.append(e.get("text", ""))

        return "\n\n".join(text for text in texts if text.strip())

    def _create_full_document_text(self, elements: List[Dict[str, Any]]) -> str:
        """
        Create full document text from all elements for Anthropic's contextual retrieval.

        Combines all element texts into a single document string that will be used
        as context when generating chunk summaries. Uses HTML for tables when available.
        """
        texts = []
        for element in elements:
            # For tables, prefer HTML representation for better structure
            if element.get("type") == "Table":
                html_text = element.get("metadata", {}).get("text_as_html")
                if html_text and html_text.strip():
                    texts.append(html_text)
                else:
                    text = element.get("text", "").strip()
                    if text:
                        texts.append(text)
            else:
                text = element.get("text", "").strip()
                if text:
                    texts.append(text)
        return "\n\n".join(texts)

    def generate_contextual_summary(
        self,
        chunk_text: str,
        full_document_text: str,
        document_name: str,
        section_title: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> str:
        """
        Use Claude on AWS Bedrock to generate contextual summary using Anthropic's method.

        Following Anthropic's Contextual Retrieval approach:
        - Provides ENTIRE document as context
        - Uses exact prompt template from Anthropic
        - Generates 50-100 token context

        Reference: https://www.anthropic.com/engineering/contextual-retrieval
        """
        # Anthropic's exact prompt template
        prompt = f"""<document>
{full_document_text}
</document>

<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        try:
            # Format request for Bedrock Claude
            native_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,  # 50-100 tokens as per Anthropic's method
                "temperature": 0.0,  # Deterministic for consistency
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            }

            # Invoke Claude model on Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.claude_model,
                body=json.dumps(native_request)
            )

            # Parse response
            model_response = json.loads(response["body"].read())
            context = model_response["content"][0]["text"].strip()
            return context

        except (ClientError, Exception) as e:
            print(f"Error generating context: {e}")
            # Fallback to simple metadata context
            return f"This chunk is from {document_name}" + (
                f" in the {section_title} section" if section_title else ""
            ) + (f" on page {page_number}" if page_number else "") + "."

    def generate_image_description(
        self,
        image_base64: str,
        media_type: str = "image/png"
    ) -> str:
        """
        Use Claude Vision on AWS Bedrock to generate a description of an image.

        Args:
            image_base64: Base64-encoded image data
            media_type: Image media type (image/png, image/jpeg, etc.)

        Returns:
            Text description of the image
        """
        prompt = "Describe this image in detail. Include what type of content it is (chart, diagram, screenshot, photo, etc.) and all relevant information visible in the image. Be specific and thorough."

        try:
            # Format request for Bedrock Claude Vision
            native_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            # Invoke Claude Vision model on Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.claude_model,
                body=json.dumps(native_request)
            )

            # Parse response
            model_response = json.loads(response["body"].read())

            # Check if content exists and is not empty
            if "content" in model_response and len(model_response["content"]) > 0:
                description = model_response["content"][0]["text"].strip()
                return description
            else:
                print(f"Error: Claude returned empty content. Response: {model_response}")
                return "[Image: description unavailable - empty response]"

        except ClientError as e:
            print(f"Error generating image description (ClientError): {e}")
            return "[Image: description unavailable]"
        except Exception as e:
            print(f"Error generating image description: {type(e).__name__}: {e}")
            return "[Image: description unavailable]"

    def process_single_chunk(
        self,
        chunk_data: Dict[str, Any],
        chunk_index: int,
        document_name: str,
        full_document_text: str,
        use_llm_context: bool
    ) -> Optional[Chunk]:
        """Process a single chunk (used for parallel processing)"""
        chunk_text = self.combine_chunk_text(chunk_data)

        if not chunk_text.strip():
            return None

        # Generate contextual summary using Anthropic's method
        if use_llm_context:
            context_summary = self.generate_contextual_summary(
                chunk_text,
                full_document_text,  # Pass entire document
                document_name,
                chunk_data["metadata"].get("section_title"),
                chunk_data["metadata"].get("page_number")
            )
            contextualized_content = f"{context_summary}\n\n{chunk_text}"
        else:
            contextualized_content = chunk_text

        # Create chunk object
        chunk = Chunk(
            chunk_id=f"{document_name}_chunk_{chunk_index}",
            original_content=chunk_text,
            contextualized_content=contextualized_content,
            metadata={
                **chunk_data["metadata"],
                "document": document_name,
                "chunk_index": chunk_index,
                "element_count": len(chunk_data["elements"])
            },
            token_count=self.count_tokens(contextualized_content)
        )
        return chunk

    def process_document(
        self,
        json_path: str,
        output_path: str,
        use_llm_context: bool = True,
        parallel: bool = True
    ) -> List[Chunk]:
        """
        Process a single document through the full chunking pipeline.

        Args:
            json_path: Path to Unstructured JSON output
            output_path: Where to save chunked output
            use_llm_context: Whether to use LLM for context generation
            parallel: Whether to process chunks in parallel (faster for LLM context)

        Returns:
            List of Chunk objects
        """
        print(f"Processing: {os.path.basename(json_path)}")

        # Load elements
        elements = self.load_unstructured_elements(json_path)

        # Create full document text for Anthropic's contextual retrieval method
        full_document_text = self._create_full_document_text(elements)

        # Group semantically
        grouped_chunks = self.group_elements_semantically(elements)

        # Get document name
        document_name = os.path.basename(json_path).replace(".json", "")

        # Process chunks (with optional parallelization for LLM calls)
        chunks = []

        if parallel and use_llm_context:
            # Process in parallel for speed
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        self.process_single_chunk,
                        chunk_data,
                        i,
                        document_name,
                        full_document_text,  # Pass entire document
                        use_llm_context
                    ): i
                    for i, chunk_data in enumerate(grouped_chunks)
                }

                with tqdm(total=len(grouped_chunks), desc="Creating chunks") as pbar:
                    for future in as_completed(future_to_index):
                        chunk = future.result()
                        if chunk:
                            chunks.append(chunk)
                        pbar.update(1)

            # Sort by chunk_index to maintain order
            chunks.sort(key=lambda x: x.metadata["chunk_index"])
        else:
            # Process sequentially
            for i, chunk_data in enumerate(tqdm(grouped_chunks, desc="Creating chunks")):
                chunk = self.process_single_chunk(
                    chunk_data, i, document_name, full_document_text, use_llm_context
                )
                if chunk:
                    chunks.append(chunk)

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2)

        print(f"Created {len(chunks)} chunks, saved to {output_path}")
        return chunks

    def process_all_documents(
        self,
        input_dir: str,
        output_dir: str,
        use_llm_context: bool = True,
        parallel: bool = True
    ):
        """
        Process all JSON documents in a directory.

        Args:
            input_dir: Directory containing Unstructured JSON outputs
            output_dir: Directory to save chunked outputs
            use_llm_context: Whether to use LLM for context generation
            parallel: Whether to use parallel processing
        """
        os.makedirs(output_dir, exist_ok=True)

        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        print(f"Processing {len(json_files)} documents...")

        for json_file in json_files:
            input_path = os.path.join(input_dir, json_file)
            output_path = os.path.join(output_dir, json_file.replace('.json', '_chunks.json'))

            try:
                self.process_document(input_path, output_path, use_llm_context, parallel)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        print(f"\nAll documents processed! Output in {output_dir}")


if __name__ == "__main__":

    # Initialize chunker with semantic chunking using Azure embeddings and AWS Bedrock Claude
    chunker = ContextualChunker(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-01-01-preview",
        embedding_model="text-embedding-3-large",  # Azure OpenAI embeddings for semantic chunking
        aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),  # AWS region for Bedrock
        claude_model=os.getenv("AWS_BEDROCK_CLAUDE_MODEL"),  # Claude model ARN from .env
        similarity_threshold=0.75,  # Cosine similarity threshold for topic boundaries
        min_chunk_size=400,  # Minimum chunk size in tokens
        max_chunk_size=1200,  # Maximum chunk size in tokens
        max_workers=2  # Sequential processing to avoid Bedrock rate limits
    )

    # Process all documents
    chunker.process_all_documents(
        input_dir=os.getcwd() + "/dataset/res",
        output_dir=os.getcwd() + "/dataset/chunks",
        use_llm_context=True,  # Set to False for faster processing without LLM context
        parallel=True  # Use parallel processing for speed
    )
