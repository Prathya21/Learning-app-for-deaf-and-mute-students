import json
from pathlib import Path
from typing import Optional


class VideoLookupService:
    def __init__(self, dictionary_path: Optional[str] = None):
        if dictionary_path is None:
            base_dir = Path(__file__).parent.parent
            dictionary_path = base_dir / "data" / "isl_dictionary.json"
        self.dictionary_path = Path(dictionary_path)
        self._dictionary: dict = {}
        self._load_dictionary()

    def _load_dictionary(self) -> None:
        try:
            with open(self.dictionary_path, "r", encoding="utf-8") as f:
                self._dictionary = json.load(f)
        except FileNotFoundError:
            self._dictionary = {}
        except json.JSONDecodeError:
            self._dictionary = {}

    def _get_video_file(self, entry) -> Optional[str]:
        if isinstance(entry, str):
            return entry.split("/")[-1]
        elif isinstance(entry, dict):
            return entry.get("video_file")
        return None

    def lookup(self, word: str) -> dict:
        upper_word = word.upper()
        entry = self._dictionary.get(upper_word)
        video_file = self._get_video_file(entry)
        if video_file:
            video_path = Path(__file__).parent.parent / "data" / "videos" / video_file
            file_exists = video_path.exists()
        else:
            file_exists = False
        return {
            "word": upper_word,
            "found": file_exists,
            "video_url": f"/api/video/{video_file}" if file_exists else None,
            "video_file": video_file if file_exists else None
        }

    def get_video_path(self, word: str) -> Optional[Path]:
        upper_word = word.upper()
        entry = self._dictionary.get(upper_word)
        video_file = self._get_video_file(entry)
        if video_file:
            return Path(__file__).parent.parent / "data" / "videos" / video_file
        return None

    def reload(self) -> None:
        self._load_dictionary()


video_lookup_service = VideoLookupService()