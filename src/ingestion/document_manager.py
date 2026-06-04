import os
from typing import List, Dict, Any, Optional

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.docx_loader import DOCXLoader
from src.ingestion.text_loader import TextLoader
from src.ingestion.xml_loader import XMLLoader
from src.ingestion.json_loader import JSONLoader
from schemas.entity_schema import ArtifactType


class DocumentManager:
    """
    Single entry point for loading any supported document type.
    Dispatches to the correct loader and injects source_id and
    artifact_type into every section dict.
    """

    SUPPORTED = {
        ".pdf":  PDFLoader,
        ".docx": DOCXLoader,
        ".txt":  TextLoader,
        ".xml":  XMLLoader,
        ".uml":  XMLLoader,
        ".xmi":  XMLLoader,
        ".json": JSONLoader,
        ".yaml": TextLoader,
        ".yml":  TextLoader,
        ".puml": TextLoader,
    }

    def __init__(self):
        self._loaders = {ext: cls() for ext, cls in self.SUPPORTED.items()}

    def load_document(
        self,
        file_path: str,
        artifact_type: str,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load a document and return unified sections.

        Each section dict contains:
            section_id    : int
            section_ref   : str
            source_id     : str
            artifact_type : str
            text          : str
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("Document not found: {}".format(file_path))

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self._loaders:
            raise ValueError(
                "Unsupported format '{}'. Supported: {}".format(
                    ext, list(self.SUPPORTED.keys())
                )
            )

        valid_types = [e.value for e in ArtifactType]
        if artifact_type not in valid_types:
            raise ValueError(
                "Invalid artifact_type '{}'. Must be one of: {}".format(
                    artifact_type, valid_types
                )
            )

        if source_id is None:
            source_id = os.path.splitext(os.path.basename(file_path))[0]

        raw_sections = self._loaders[ext].load(file_path)

        for section in raw_sections:
            section["source_id"]    = source_id
            section["artifact_type"] = artifact_type

        return raw_sections

    def load_directory(
        self,
        dir_path: str,
        artifact_type: str,
    ) -> List[Dict[str, Any]]:
        """Load all supported documents in a directory."""
        all_sections = []
        for fname in sorted(os.listdir(dir_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.SUPPORTED:
                fpath = os.path.join(dir_path, fname)
                try:
                    sections = self.load_document(fpath, artifact_type)
                    all_sections.extend(sections)
                    print("  Loaded {}: {} sections".format(fname, len(sections)))
                except Exception as e:
                    print("  WARNING: Failed to load {}: {}".format(fname, e))
        return all_sections