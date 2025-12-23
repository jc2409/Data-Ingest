import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3
import tiktoken
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

from .embedding import EmbeddingClient, cosine_similarity

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
    - Embedding-based semantic grouping using Amazon Titan Embed Text v2
    - Automatic topic boundary detection via cosine similarity
    - LLM-enhanced context generation using Claude on AWS Bedrock
    - Rich metadata preservation
    """

    def __init__(
        self,
        aws_region: str = "us-east-1",
        claude_model: str = os.getenv("AWS_BEDROCK_CLAUDE_MODEL"),
        vision_model: str = os.getenv("AWS_BEDROCK_VISION_MODEL"),
        embedding_model: str = os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL"),
        embedding_dimensions: int = 1024,
        normalize_embeddings: bool = True,
        similarity_threshold: float = 0.70,
        min_chunk_size: int = 300,
        max_chunk_size: int = 1000,
        chunk_overlap: float = 0.0,
        max_workers: int = 5
    ):
        # AWS Bedrock client for Claude (context and vision)
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region
        )
        self.claude_model = claude_model  # Used for context generation (can be Haiku)
        # Vision model for image descriptions (must support vision, e.g., Sonnet)
        self.vision_model = vision_model

        # Shared embedding client with caching
        self.embedding_client = EmbeddingClient(
            aws_region=aws_region,
            model_id=embedding_model,
            dimensions=embedding_dimensions,
            normalize=normalize_embeddings
        )
        self.embedding_dimensions = embedding_dimensions

        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap  # 0.0 to 0.5 (e.g., 0.2 = 20% overlap)
        self.max_workers = max_workers

        # Initialize tokenizer (using cl100k_base for consistency)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def load_unstructured_elements(self, json_path: str) -> List[Dict[str, Any]]:
        """Load parsed elements from Unstructured JSON output"""
        with open(json_path, 'r') as f:
            return json.load(f)

    def _process_image_element(self, element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single image element by generating its description.
        Returns the modified element or None if invalid.
        """
        image_base64 = element.get("metadata", {}).get("image_base64")
        if not image_base64:
            return None

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

        filetype_lower = filetype.lower().strip()
        media_type = media_type_map.get(filetype_lower, "image/png")

        description = self.generate_image_description(image_base64, media_type)

        element["text"] = f"[IMAGE] {description}"
        element["metadata"]["image_description"] = description
        element["metadata"]["has_image"] = True
        return element

    def _process_elements_concurrently(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process elements concurrently: images in parallel, text immediately.

        Strategy:
        - Submit all images to thread pool for concurrent processing
        - Process text elements immediately (no API calls needed)
        - Wait for image descriptions to complete
        - Return all valid elements in original order
        """
        print("  Processing elements (generating image descriptions concurrently)...")

        # Separate images and text, preserving order
        image_indices = []
        image_elements = []
        text_elements = []
        element_order = []  # Track original positions

        for idx, element in enumerate(elements):
            element_type = element.get("type", "")

            if element_type == "Image":
                image_base64 = element.get("metadata", {}).get("image_base64")
                if image_base64:
                    image_indices.append(idx)
                    image_elements.append(element)
                    element_order.append(('image', len(image_elements) - 1))
            elif element.get("text", "").strip():
                text_elements.append(element)
                element_order.append(('text', len(text_elements) - 1))

        # Process images concurrently
        processed_images = [None] * len(image_elements)
        if image_elements:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._process_image_element, img): i
                    for i, img in enumerate(image_elements)
                }

                with tqdm(total=len(image_elements), desc="  Processing images", leave=False) as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        processed_images[idx] = future.result()
                        pbar.update(1)

        # Reconstruct elements in original order
        valid_elements = []
        for element_type, idx in element_order:
            if element_type == 'image':
                if processed_images[idx] is not None:
                    valid_elements.append(processed_images[idx])
            else:  # text
                valid_elements.append(text_elements[idx])

        return valid_elements

    def group_elements_by_size(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group elements using size-based chunking with natural boundaries.

        Strategy:
        - Use Titles as section boundaries
        - Keep Tables and Images as separate chunks
        - Group text elements until reaching target size
        - Respect natural document structure
        """
        # Process elements: generate descriptions for images (concurrently), filter empty text elements
        valid_elements = self._process_elements_concurrently(elements)

        if not valid_elements:
            return []

        print("  Grouping elements by size...")

        chunks = []
        current_chunk = {
            "elements": [],
            "tokens": 0,
            "metadata": {}
        }
        current_section_title = None

        for element in tqdm(valid_elements, desc="  Creating chunks", leave=False):
            element_type = element.get("type", "")
            element_text = element.get("text", "")
            element_tokens = self.count_tokens(element_text)

            metadata = element.get("metadata", {})
            page_number = metadata.get("page_number")

            # Track section titles for context
            if element_type == "Title":
                current_section_title = element_text

                # If we have a substantial chunk before this title, save it
                if current_chunk["elements"] and current_chunk["tokens"] >= self.min_chunk_size:
                    chunks.append(self._finalize_chunk(current_chunk))
                    current_chunk = {"elements": [], "tokens": 0, "metadata": {}}

            # Images should be their own chunks
            if element_type == "Image":
                # Finish current chunk if it exists
                if current_chunk["elements"]:
                    chunks.append(self._finalize_chunk(current_chunk))
                    current_chunk = {"elements": [], "tokens": 0, "metadata": {}}

                # Add image as standalone chunk
                image_metadata = {
                    "section_title": current_section_title,
                    "page_number": page_number,
                    "chunk_type": "image",
                    "has_image": True,
                    "image_base64": metadata.get("image_base64"),
                    "image_description": metadata.get("image_description"),
                    "filetype": metadata.get("filetype")
                }

                chunks.append({
                    "elements": [element],
                    "tokens": element_tokens,
                    "metadata": image_metadata
                })
                continue

            # Tables should be their own chunks
            if element_type == "Table":
                # Finish current chunk if it exists
                if current_chunk["elements"]:
                    chunks.append(self._finalize_chunk(current_chunk))
                    current_chunk = {"elements": [], "tokens": 0, "metadata": {}}

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

            # Determine if we should start a new chunk
            should_split = False

            # Split if adding this element would exceed max size
            if current_chunk["elements"] and current_chunk["tokens"] + element_tokens > self.max_chunk_size:
                # Only split if current chunk meets minimum size
                if current_chunk["tokens"] >= self.min_chunk_size:
                    should_split = True

            if should_split:
                finalized = self._finalize_chunk(current_chunk)
                chunks.append(finalized)

                # Get overlap from the chunk we just finalized
                overlap_text = self._get_overlap_text(finalized)

                # Start new chunk, optionally with overlap
                new_elements = []
                new_tokens = 0
                if overlap_text:
                    overlap_element = {"type": "Overlap", "text": f"[...] {overlap_text}", "metadata": {}}
                    new_elements.append(overlap_element)
                    new_tokens = self.count_tokens(overlap_text)

                new_elements.append(element)
                new_tokens += element_tokens

                current_chunk = {
                    "elements": new_elements,
                    "tokens": new_tokens,
                    "metadata": {
                        "section_title": current_section_title,
                        "page_number": page_number
                    }
                }
            else:
                # Add to current chunk
                current_chunk["elements"].append(element)
                current_chunk["tokens"] += element_tokens

                if not current_chunk["metadata"]:
                    current_chunk["metadata"] = {
                        "section_title": current_section_title,
                        "page_number": page_number
                    }

        # Add final chunk if it has content
        if current_chunk["elements"]:
            chunks.append(self._finalize_chunk(current_chunk))

        return chunks

    def group_elements_semantically(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group elements using semantic similarity with embeddings.

        Strategy:
        - Generate embeddings for each element using Amazon Titan Embed Text v2
        - Calculate cosine similarity between consecutive elements
        - Create chunk boundaries when similarity drops below threshold
        - Keep tables as separate chunks (self-contained units)
        - Respect min/max chunk size constraints
        """
        # Process elements: generate descriptions for images (concurrently), filter empty text elements
        valid_elements = self._process_elements_concurrently(elements)

        if not valid_elements:
            return []

        print("  Generating embeddings for semantic chunking...")

        # Generate embeddings for all elements (with progress bar)
        element_embeddings = []
        for element in tqdm(valid_elements, desc="  Creating embeddings", leave=False):
            embedding = self.embedding_client.get_embedding(element.get("text", ""))
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
                similarity = cosine_similarity(chunk_embedding, element_embedding)

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
                finalized = self._finalize_chunk(current_chunk)
                chunks.append(finalized)

                # Get overlap from the chunk we just finalized
                overlap_text = self._get_overlap_text(finalized)

                # Start new chunk, optionally with overlap
                new_elements = []
                new_tokens = 0
                if overlap_text:
                    overlap_element = {"type": "Overlap", "text": f"[...] {overlap_text}", "metadata": {}}
                    new_elements.append(overlap_element)
                    new_tokens = self.count_tokens(overlap_text)

                new_elements.append(element)
                new_tokens += element_tokens

                current_chunk = {
                    "elements": new_elements,
                    "tokens": new_tokens,
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
        # Ensure metadata has chunk_type and has_image set
        if "chunk_type" not in chunk["metadata"]:
            chunk["metadata"]["chunk_type"] = "text"
        if "has_image" not in chunk["metadata"]:
            chunk["metadata"]["has_image"] = False

        return {
            "elements": chunk["elements"],
            "tokens": chunk["tokens"],
            "metadata": chunk["metadata"]
        }

    def html_table_to_markdown(self, html: str) -> str:
        """
        Convert HTML table to Markdown format for better embedding quality

        Args:
            html: HTML table string

        Returns:
            Markdown formatted table
        """
        try:
            # Remove HTML tags but keep structure
            # Simple approach: extract cell contents and format as markdown

            # Extract all table cells
            rows = []

            # Find all <tr> elements
            tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
            tr_matches = tr_pattern.findall(html)

            for tr_content in tr_matches:
                # Find all <td> or <th> elements in this row
                cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
                cells = cell_pattern.findall(tr_content)

                # Clean cell content (remove nested tags, normalize whitespace)
                cleaned_cells = []
                for cell in cells:
                    # Remove HTML tags
                    clean = re.sub(r'<[^>]+>', '', cell)
                    # Normalize whitespace
                    clean = ' '.join(clean.split())
                    # Unescape HTML entities
                    clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&')
                    cleaned_cells.append(clean)

                if cleaned_cells:
                    rows.append(cleaned_cells)

            if not rows:
                # Fallback: just remove all HTML tags
                clean = re.sub(r'<[^>]+>', ' ', html)
                return ' '.join(clean.split())

            # Build markdown table
            markdown_lines = []

            # Determine column count
            max_cols = max(len(row) for row in rows) if rows else 0

            if max_cols == 0:
                return ' '.join(' '.join(row) for row in rows)

            # First row as header
            if rows:
                header = rows[0]
                # Pad header if needed
                while len(header) < max_cols:
                    header.append('')
                markdown_lines.append('| ' + ' | '.join(header) + ' |')
                markdown_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')

                # Remaining rows
                for row in rows[1:]:
                    # Pad row if needed
                    while len(row) < max_cols:
                        row.append('')
                    markdown_lines.append('| ' + ' | '.join(row) + ' |')

            return '\n'.join(markdown_lines)

        except Exception as e:
            print(f"Error converting HTML to markdown: {e}")
            # Fallback: strip all HTML tags
            clean = re.sub(r'<[^>]+>', ' ', html)
            return ' '.join(clean.split())

    def combine_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Combine elements into a single text string, converting HTML tables to Markdown"""
        texts = []
        for e in chunk["elements"]:
            # For tables, convert HTML to Markdown for better embedding quality
            if e.get("type") == "Table":
                html_text = e.get("metadata", {}).get("text_as_html")
                if html_text:
                    # Convert HTML table to Markdown
                    markdown_table = self.html_table_to_markdown(html_text)
                    texts.append(markdown_table)
                else:
                    # Fall back to plain text if HTML not available
                    texts.append(e.get("text", ""))
            else:
                texts.append(e.get("text", ""))

        return "\n\n".join(text for text in texts if text.strip())

    def _get_overlap_text(self, chunk: Dict[str, Any]) -> Optional[str]:
        """
        Get overlap text from the end of a chunk.

        Returns the last N% of tokens from the chunk as text,
        where N is determined by self.chunk_overlap.
        """
        if self.chunk_overlap <= 0:
            return None

        chunk_text = self.combine_chunk_text(chunk)
        if not chunk_text:
            return None

        tokens = self.tokenizer.encode(chunk_text)
        overlap_count = int(len(tokens) * self.chunk_overlap)

        if overlap_count <= 0:
            return None

        overlap_tokens = tokens[-overlap_count:]
        return self.tokenizer.decode(overlap_tokens).strip()

    def _create_full_document_text(self, elements: List[Dict[str, Any]]) -> str:
        """
        Create full document text from all elements for Anthropic's contextual retrieval.

        Combines all element texts into a single document string that will be used
        as context when generating chunk summaries. Uses Markdown for tables for better LLM understanding.
        """
        texts = []
        for element in elements:
            # For tables, convert HTML to Markdown for better structure and readability
            if element.get("type") == "Table":
                html_text = element.get("metadata", {}).get("text_as_html")
                if html_text and html_text.strip():
                    # Convert HTML table to Markdown
                    markdown_table = self.html_table_to_markdown(html_text)
                    texts.append(markdown_table)
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

            # Invoke Claude Vision model on Bedrock (use vision_model which supports images)
            response = self.bedrock_client.invoke_model(
                modelId=self.vision_model,
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
        parallel: bool = True,
        chunking_strategy: str = "basic"
    ) -> List[Chunk]:
        """
        Process a single document through the full chunking pipeline.

        Args:
            json_path: Path to Unstructured JSON output
            output_path: Where to save chunked output
            use_llm_context: Whether to use LLM for context generation
            parallel: Whether to process chunks in parallel (faster for LLM context)
            chunking_strategy: Chunking method to use ("basic" or "semantic")

        Returns:
            List of Chunk objects
        """
        print(f"Processing: {os.path.basename(json_path)}")

        # Load elements
        elements = self.load_unstructured_elements(json_path)

        # Create full document text for Anthropic's contextual retrieval method
        full_document_text = self._create_full_document_text(elements)

        # Group elements based on selected strategy
        if chunking_strategy == "semantic":
            grouped_chunks = self.group_elements_semantically(elements)
        elif chunking_strategy == "basic":
            grouped_chunks = self.group_elements_by_size(elements)
        else:
            raise ValueError(f"Unknown chunking_strategy: {chunking_strategy}. Use 'basic' or 'semantic'.")

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
        parallel: bool = True,
        chunking_strategy: str = "basic"
    ):
        """
        Process all JSON documents in a directory.

        Args:
            input_dir: Directory containing Unstructured JSON outputs
            output_dir: Directory to save chunked outputs
            use_llm_context: Whether to use LLM for context generation
            parallel: Whether to use parallel processing
            chunking_strategy: Chunking method to use ("basic" or "semantic")
        """
        os.makedirs(output_dir, exist_ok=True)

        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        print(f"Processing {len(json_files)} documents...")

        for json_file in json_files:
            input_path = os.path.join(input_dir, json_file)
            output_path = os.path.join(output_dir, json_file.replace('.json', '_chunks.json'))

            try:
                self.process_document(input_path, output_path, use_llm_context, parallel, chunking_strategy)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        print(f"\nAll documents processed! Output in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Contextual chunking for document processing"
    )
    parser.add_argument(
        "-s", "--single",
        type=str,
        metavar="FILE",
        help="Process a single document"
    )
    parser.add_argument(
        "-m", "--multiple",
        action="store_true",
        help="Process all documents in dataset/res/"
    )
    parser.add_argument(
        "-c", "--chunking",
        type=str,
        choices=["basic", "semantic"],
        default="basic",
        help="Chunking strategy: 'basic' or 'semantic' (default: basic)"
    )

    args = parser.parse_args()

    # Initialize chunker using Amazon Titan embeddings and AWS Bedrock Claude
    chunker = ContextualChunker(
        aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        claude_model=os.getenv("AWS_BEDROCK_CLAUDE_MODEL"),
        vision_model=os.getenv("AWS_BEDROCK_VISION_MODEL"),
        embedding_model=os.getenv("AWS_BEDROCK_TITAN_EMBEDDING_MODEL"),
        embedding_dimensions=1024,
        normalize_embeddings=True,
        similarity_threshold=0.70,
        min_chunk_size=400,
        max_chunk_size=800,
        chunk_overlap=0.15,  # 15% overlap between chunks
        max_workers=3
    )

    if args.single:
        print(f"Single Document Processing mode (chunking: {args.chunking})")
        json_path = os.path.join(os.getcwd(), args.single)
        output_path = json_path.replace("/res/", "/chunks/").replace(".json", "_chunks.json")
        chunker.process_document(
            json_path=json_path,
            output_path=output_path,
            use_llm_context=True,
            parallel=True,
            chunking_strategy=args.chunking
        )
    elif args.multiple:
        print(f"Multiple Document Processing mode (chunking: {args.chunking})")
        chunker.process_all_documents(
            input_dir=os.path.join(os.getcwd(), "dataset/res"),
            output_dir=os.path.join(os.getcwd(), "dataset/chunks"),
            use_llm_context=True,
            parallel=True,
            chunking_strategy=args.chunking
        )
    else:
        parser.print_help()