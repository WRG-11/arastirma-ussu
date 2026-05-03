"""Document loading from the local filesystem."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]


def load_documents(doc_dir: str | Path = "data/documents") -> list:
    """Load documents from *doc_dir* using LlamaIndex SimpleDirectoryReader.

    Returns an empty list when the directory is missing, empty, or contains
    no files with a supported extension.
    """
    from llama_index.core import SimpleDirectoryReader

    doc_path = Path(doc_dir)
    if not doc_path.exists() or not doc_path.is_dir():
        logger.warning("Belge dizini bulunamadi: %s", doc_path)
        return []

    # Check if there are any supported files before creating the reader
    has_files = any(
        f
        for f in doc_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not has_files:
        logger.info("Desteklenen dosya bulunamadi: %s", doc_path)
        return []

    reader = SimpleDirectoryReader(
        input_dir=str(doc_path),
        required_exts=SUPPORTED_EXTENSIONS,
        recursive=True,
        filename_as_id=True,
    )
    documents = reader.load_data()
    logger.info("%d belge yuklendi (%s)", len(documents), doc_path)
    return documents
