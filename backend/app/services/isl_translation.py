import re
from typing import List
from app.services.video_lookup import video_lookup_service


class TranslationService:
    def __init__(self):
        self.video_lookup = video_lookup_service

    def normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def to_gloss(self, tokens: List[str]) -> List[str]:
        gloss_map = {
            "hello": "HELLO",
            "hi": "HELLO",
            "thanks": "THANK_YOU",
            "thank you": "THANK_YOU",
            "please": "PLEASE",
            "yes": "YES",
            "no": "NO",
            "student": "STUDENT",
            "teacher": "TEACHER",
            "learn": "LEARN",
            "book": "BOOK",
            "water": "WATER",
            "good": "GOOD",
            "bad": "BAD",
            "help": "HELP",
            "understand": "UNDERSTAND",
            "question": "QUESTION",
        }
        gloss_sequence = []
        for token in tokens:
            gloss = gloss_map.get(token, token.upper())
            gloss_sequence.append(gloss)
        return gloss_sequence

    def translate(self, text: str) -> dict:
        normalized = self.normalize_text(text)
        tokens = self.tokenize(normalized)
        gloss_sequence = self.to_gloss(tokens)

        videos = []
        for gloss in gloss_sequence:
            video_info = self.video_lookup.lookup(gloss)
            videos.append(video_info)

        return {
            "original_text": text,
            "gloss_sequence": gloss_sequence,
            "videos": videos
        }


translation_service = TranslationService()