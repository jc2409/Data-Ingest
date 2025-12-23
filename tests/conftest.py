"""Shared pytest fixtures for data-ingest tests."""
import json
import os
import pytest
from unittest.mock import MagicMock, patch


def pytest_addoption(parser):
    """Add command line options for integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires API credentials)"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires --run-integration)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(reason="Need --run-integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def mock_bedrock_client():
    """Mock AWS Bedrock client for embedding tests."""
    with patch("boto3.client") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance
        yield client_instance


@pytest.fixture
def sample_embedding():
    """Sample 1024-dimension embedding vector."""
    return [0.1] * 1024


@pytest.fixture
def mock_embedding_response(sample_embedding):
    """Mock response from Titan embedding API."""
    response_body = MagicMock()
    response_body.read.return_value = json.dumps({"embedding": sample_embedding})
    return {"body": response_body}


@pytest.fixture
def sample_elements():
    """Sample document elements from Unstructured."""
    return [
        {
            "type": "Title",
            "text": "Introduction",
            "metadata": {"page_number": 1}
        },
        {
            "type": "NarrativeText",
            "text": "This is the first paragraph of the document. It contains important information.",
            "metadata": {"page_number": 1}
        },
        {
            "type": "NarrativeText",
            "text": "This is the second paragraph with more details about the topic.",
            "metadata": {"page_number": 1}
        },
        {
            "type": "Table",
            "text": "Header1 | Header2\nValue1 | Value2",
            "metadata": {
                "page_number": 2,
                "text_as_html": "<table><tr><th>Header1</th><th>Header2</th></tr><tr><td>Value1</td><td>Value2</td></tr></table>"
            }
        },
        {
            "type": "Image",
            "text": "",
            "metadata": {
                "page_number": 3,
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "filetype": "png"
            }
        }
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for indexing tests."""
    return [
        {
            "chunk_id": "test_doc_chunk_0",
            "original_content": "This is the first chunk content.",
            "contextualized_content": "From test document introduction: This is the first chunk content.",
            "metadata": {
                "document": "test_doc",
                "chunk_index": 0,
                "chunk_type": "text",
                "section_title": "Introduction",
                "page_number": 1,
                "element_count": 2,
                "has_image": False
            },
            "token_count": 15
        },
        {
            "chunk_id": "test_doc_chunk_1",
            "original_content": "| Header1 | Header2 |\n| --- | --- |\n| Value1 | Value2 |",
            "contextualized_content": "Table from test document: | Header1 | Header2 |\n| --- | --- |\n| Value1 | Value2 |",
            "metadata": {
                "document": "test_doc",
                "chunk_index": 1,
                "chunk_type": "table",
                "section_title": "Data",
                "page_number": 2,
                "element_count": 1,
                "has_image": False
            },
            "token_count": 25
        }
    ]
