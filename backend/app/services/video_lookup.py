import json
from pathlib import Path
from typing import Optional
import os


class VideoLookupService:
    def __init__(self, dictionary_path: Optional[str] = None):
        if dictionary_path is None:
            # __file__ = .../backend/app/services/video_lookup.py
            # parent = .../backend/app/services
            # parent.parent = .../backend/app
            # parent.parent.parent = .../backend
            base_dir = Path(__file__).parent.parent.parent
            dictionary_path = base_dir / "app" / "data" / "isl_dictionary.json"
        else:
            base_dir = Path(dictionary_path).parent.parent
        self.dictionary_path = Path(dictionary_path)
        self._video_dir = Path(__file__).parent.parent.parent / "data" / "real_videos_canonical"
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

    def _find_video_file(self, video_file: str) -> Optional[Path]:
        """Find video file with case-insensitive matching."""
        if not video_file:
            return None
        
        # First try exact match
        exact_path = self._video_dir / video_file
        if exact_path.exists():
            return exact_path
        
        # Try uppercase
        upper_path = self._video_dir / video_file.upper()
        if upper_path.exists():
            return upper_path
        
        # Try lowercase
        lower_path = self._video_dir / video_file.lower()
        if lower_path.exists():
            return lower_path
        
        # Try to find by stem (without extension)
        stem = Path(video_file).stem
        for ext in ['.mp4', '.MP4', '.mov', '.MOV', '.webm', '.WEBM']:
            for variant in [video_file, video_file.upper(), video_file.lower()]:
                test_path = self._video_dir / (Path(variant).stem + ext)
                if test_path.exists():
                    return test_path
        
        return None

    def lookup(self, word: str) -> dict:
        upper_word = word.upper()
        entry = self._dictionary.get(upper_word)
        video_file = self._get_video_file(entry)
        if video_file:
            video_path = self._find_video_file(video_file)
            file_exists = video_path is not None
        else:
            file_exists = False
            video_path = None
        video_url = f"/static/videos/{video_file}" if file_exists else None
        return {
            "word": upper_word,
            "found": file_exists,
            "video_url": video_url,
            "video_file": video_file if file_exists else None,
            "video_path": str(video_path) if video_path else None
        }

    def get_video_path(self, word: str) -> Optional[Path]:
        upper_word = word.upper()
        entry = self._dictionary.get(upper_word)
        video_file = self._get_video_file(entry)
        if video_file:
            return self._find_video_file(video_file)
        return None

    def reload(self) -> None:
        self._load_dictionary()


video_lookup_service = VideoLookupService()