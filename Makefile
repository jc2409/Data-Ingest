.PHONY: install dev clean test test-unit test-integration process chunk chunk-all index pipeline help

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# Directories
SRC_DIR := dataset/src
RES_DIR := dataset/res
CHUNKS_DIR := dataset/chunks

# Default target
help:
	@echo "Data Ingest Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make dev              Install with dev dependencies"
	@echo ""
	@echo "Pipeline:"
	@echo "  make process          Step 1: Extract documents from dataset/src/"
	@echo "  make chunk            Step 2: Generate contextual chunks (all files)"
	@echo "  make chunk FILE=x     Step 2: Generate chunks for single file"
	@echo "  make index            Step 3: Index chunks to Pinecone"
	@echo "  make pipeline         Run full pipeline (process → chunk → index)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run unit tests"
	@echo "  make test-integration Run integration tests (real APIs)"
	@echo "  make coverage         Run tests with coverage report"
	@echo ""
	@echo "Utilities:"
	@echo "  make query            Run retrieval examples"
	@echo "  make stats            Show Pinecone index statistics"
	@echo "  make clean            Remove all generated files"

# Setup
install:
	uv sync

dev:
	uv sync --extra dev

# Pipeline steps
process:
	uv run python -m src.process_documents

chunk:
ifdef FILE
	@echo "Processing single file: $(FILE)"
	uv run python -m src.contextual_chunking --single $(FILE)
else
	@echo "=========================================="
	@echo "Chunking All Documents"
	@echo "=========================================="
	@mkdir -p $(CHUNKS_DIR)
	@total=0; success=0; failed=0; \
	for file in $(RES_DIR)/*.json; do \
		if [ -f "$$file" ]; then \
			total=$$((total + 1)); \
			filename=$$(basename "$$file"); \
			echo ""; \
			echo "$(YELLOW)[$$total] Processing: $$filename$(NC)"; \
			if uv run python -m src.contextual_chunking --single "$$file" 2>&1; then \
				success=$$((success + 1)); \
				echo "$(GREEN)✓ Success: $$filename$(NC)"; \
			else \
				failed=$$((failed + 1)); \
				echo "$(RED)✗ Failed: $$filename$(NC)"; \
			fi; \
		fi; \
	done; \
	echo ""; \
	echo "=========================================="; \
	echo "Complete: $$success/$$total succeeded"; \
	if [ $$failed -gt 0 ]; then \
		echo "$(RED)Failed: $$failed$(NC)"; \
		exit 1; \
	fi
endif

chunk-semantic:
ifdef FILE
	uv run python -m src.contextual_chunking --single $(FILE) --chunking semantic
else
	@echo "Processing all files with semantic chunking..."
	@for file in $(RES_DIR)/*.json; do \
		if [ -f "$$file" ]; then \
			echo "Processing: $$file"; \
			uv run python -m src.contextual_chunking --single "$$file" --chunking semantic; \
		fi; \
	done
endif

index:
	uv run python -m src.indexing

pipeline: process chunk index
	@echo ""
	@echo "$(GREEN)Pipeline complete!$(NC)"

# Testing
test: test-unit

test-unit:
	uv run pytest tests/ -v --ignore=tests/test_integration.py

test-integration:
	uv run pytest tests/test_integration.py --run-integration -v

test-all:
	uv run pytest tests/ --run-integration -v

coverage:
	uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "HTML report: htmlcov/index.html"

# Utilities
query:
	uv run python -m src.retrieve

stats:
	@uv run python -c "\
from src import PineconeIndexer; \
import os; \
i = PineconeIndexer(index_name=os.getenv('PINECONE_INDEX_NAME')); \
s = i.get_index_stats(); \
print(f'Vectors: {s[\"total_vectors\"]:,}'); \
print(f'Dimensions: {s[\"dimensions\"]}'); \
print(f'Fullness: {s[\"index_fullness\"]:.2%}')"

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

# Cleaning
clean:
	rm -rf $(CHUNKS_DIR)/* $(RES_DIR)/*
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .coverage htmlcov/
	@echo "$(GREEN)Cleaned all generated files$(NC)"

clean-chunks:
	rm -rf $(CHUNKS_DIR)/*
	@echo "Cleaned $(CHUNKS_DIR)/"

clean-cache:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .coverage htmlcov/
	@echo "Cleaned cache files"
