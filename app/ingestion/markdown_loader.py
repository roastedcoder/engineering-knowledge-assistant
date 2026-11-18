from pathlib import Path

from app.models import Document


class MarkdownLoader:

    def load(self, file_path: str) -> Document:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".md":
            raise ValueError("Only Markdown files are supported")

        content = path.read_text(encoding="utf-8")

        return Document(
            content=content,
            metadata={
                "source": path.name,
                "file_type": "markdown"
            }
        )