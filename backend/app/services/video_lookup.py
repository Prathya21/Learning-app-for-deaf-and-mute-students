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

    def lookup(self, word: str) -> dict:
        upper_word = word.upper()
        video_path = self._dictionary.get(upper_word)
        return {
            "word": upper_word,
            "found": video_path is not None,
            "video_path": video_path
        }

    def reload(self) -> None:
        self._load_dictionary()


video_lookup_service = VideoLookupService()