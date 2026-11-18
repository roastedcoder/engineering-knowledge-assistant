from pathlib import Path

from app.ingestion.markdown_loader import MarkdownLoader
from app.ingestion.pdf_loader import PDFLoader
from app.models import Document


class DocumentIngestionService:

    def __init__(self):
        self.markdown_loader = MarkdownLoader()
        self.pdf_loader = PDFLoader()

    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)

        if path.suffix.lower() == ".md":
            return [self.markdown_loader.load(file_path)]

        if path.suffix.lower() == ".pdf":
            return self.pdf_loader.load(file_path)

        raise ValueError(f"Unsupported file type: {path.suffix}")

    def load_directory(self, directory_path: str) -> list[Document]:
      directory = Path(directory_path)

      if not directory.exists():
          raise FileNotFoundError(
              f"Directory not found: {directory_path}"
          )

      if not directory.is_dir():
          raise NotADirectoryError(
              f"Not a directory: {directory_path}"
          )

      documents = []

      supported_extensions = {".pdf", ".md"}

      for file_path in sorted(directory.iterdir()):
          if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
              documents.extend(self.load(str(file_path)))

      return documents