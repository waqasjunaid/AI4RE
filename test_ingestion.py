from src.ingestion.document_manager import DocumentManager


manager = DocumentManager()

document = manager.load_document(
    "data/raw/src/sample.txt"
)

print(document[:2])