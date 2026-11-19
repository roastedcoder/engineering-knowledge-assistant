
import hashlib
import json

from pathlib import Path


class DocumentTracker:

    SUPPORTED_EXTENSIONS = {".md", ".pdf"}

    def __init__(self, manifest_path: str = "storage/manifest.json"):
        self.manifest_path = Path(manifest_path)

    def calculate_hash(self, file_path: Path) -> str:

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:
            for block in iter(lambda: file.read(4096), b""):
                sha256.update(block)

        return sha256.hexdigest()

    def load_manifest(self) -> dict[str, str]:

        if not self.manifest_path.exists():
            return {}

        with open(self.manifest_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def scan(self, directory_path: str) -> dict:

        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory_path}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {directory_path}"
            )

        previous_manifest = self.load_manifest()
        current_manifest = {}

        for file_path in sorted(directory.iterdir()):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            file_name = file_path.name
            current_manifest[file_name] = self.calculate_hash(file_path)

        added = []
        modified = []
        unchanged = []

        for file_name, current_hash in current_manifest.items():

            previous_hash = previous_manifest.get(file_name)

            if previous_hash is None:
                added.append(file_name)

            elif previous_hash != current_hash:
                modified.append(file_name)

            else:
                unchanged.append(file_name)

        deleted = [
            file_name
            for file_name in previous_manifest
            if file_name not in current_manifest
        ]

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "unchanged": unchanged,
            "manifest": current_manifest,
        }

    def save_manifest(self, manifest: dict[str, str]) -> None:

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(self.manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
