from app.models import Chunk, Document


class TextChunker:

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        content = document.content
        chunks = []

        step = self.chunk_size - self.overlap

        for start in range(0, len(content), step):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end].strip()

            if chunk_content:
                metadata = {
                    **document.metadata,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end,
                }

                chunks.append(
                    Chunk(
                        content=chunk_content,
                        metadata=metadata,
                    )
                )

            if end == len(content):
                break

        return chunks