from pathlib import Path

import pymupdf

from app.models import Document


class PDFLoader:

    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported")

        documents = []

        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                content = page.get_text().strip()

                if not content:
                    continue

                documents.append(
                    Document(
                        content=content,
                        metadata={
                            "source": path.name,
                            "file_type": "pdf",
                            "page": page_number
                        }
                    )
                )

        return documents