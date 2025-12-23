"""Tests for the process_documents module."""
import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from src.process_documents import DocumentProcessing


class TestDocumentProcessing:
    """Tests for DocumentProcessing class."""

    @pytest.fixture
    def mock_processor(self):
        """Create processor with mocked Unstructured client."""
        with patch("src.process_documents.unstructured_client.UnstructuredClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            processor = DocumentProcessing(max_workers=2)
            processor._mock_client = mock_instance

            yield processor

    def test_init(self, mock_processor):
        """Test processor initialization."""
        assert mock_processor.max_workers == 2

    def test_get_optimal_parameters_pdf(self, mock_processor):
        """Test optimal parameters for PDF files."""
        params = mock_processor._get_optimal_parameters("document.pdf")

        assert "strategy" in params
        assert params.get("pdf_infer_table_structure") is True
        assert "extract_image_block_types" in params

    def test_get_optimal_parameters_pptx(self, mock_processor):
        """Test optimal parameters for PPTX files."""
        params = mock_processor._get_optimal_parameters("presentation.pptx")

        assert "strategy" in params
        assert params.get("infer_table_structure") is True

    def test_get_optimal_parameters_docx(self, mock_processor):
        """Test optimal parameters for DOCX files."""
        params = mock_processor._get_optimal_parameters("document.docx")

        assert "strategy" in params
        assert params.get("infer_table_structure") is True

    def test_get_optimal_parameters_txt(self, mock_processor):
        """Test optimal parameters for TXT files."""
        params = mock_processor._get_optimal_parameters("readme.txt")

        assert "strategy" in params
        assert "languages" in params

    def test_get_optimal_parameters_html(self, mock_processor):
        """Test optimal parameters for HTML files."""
        params = mock_processor._get_optimal_parameters("page.html")

        assert "strategy" in params

    def test_get_optimal_parameters_unknown(self, mock_processor):
        """Test optimal parameters for unknown file types."""
        params = mock_processor._get_optimal_parameters("file.xyz")

        assert "strategy" in params
        assert "languages" in params

    def test_process_single_file_success(self, mock_processor):
        """Test successful single file processing."""
        mock_response = MagicMock()
        mock_response.elements = [
            {"type": "Title", "text": "Test"},
            {"type": "NarrativeText", "text": "Content"}
        ]
        mock_processor._mock_client.general.partition.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "dataset", "src")
            res_dir = os.path.join(tmpdir, "dataset", "res")
            os.makedirs(src_dir)
            os.makedirs(res_dir)

            test_file = os.path.join(src_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("Test content")

            result = mock_processor.process_single_file(test_file)

            assert result["status"] == "success"
            assert result["elements_count"] == 2
            assert "test.txt" in result["filename"]

    def test_process_single_file_failure(self, mock_processor):
        """Test failed single file processing."""
        mock_processor._mock_client.general.partition.side_effect = Exception("API Error")

        result = mock_processor.process_single_file("/nonexistent/file.pdf")

        assert result["status"] == "failed"
        assert "error" in result

    def test_print_summary(self, mock_processor, capsys):
        """Test summary printing."""
        results = [
            {"status": "success", "filename": "doc1.pdf", "strategy": "HI_RES", "elements_count": 10},
            {"status": "success", "filename": "doc2.pdf", "strategy": "FAST", "elements_count": 5},
            {"status": "failed", "filename": "doc3.pdf", "error": "API Error"}
        ]

        mock_processor._print_summary(results)

        captured = capsys.readouterr()
        assert "Total files: 3" in captured.out
        assert "Successful: 2" in captured.out
        assert "Failed: 1" in captured.out
        assert "doc1.pdf" in captured.out
        assert "doc3.pdf" in captured.out
        assert "API Error" in captured.out
