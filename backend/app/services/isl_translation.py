import re
from typing import List, Dict, Optional, Tuple
from app.services.video_lookup import video_lookup_service
from app.services.nlp_processor import nlp_processor, SentenceAnalysis


PHRASE_GLOSS_MAP = {
    "thank you": "THANK_YOU",
    "thanks": "THANK_YOU",
    "good morning": "GOOD_MORNING",
    "good afternoon": "GOOD_AFTERNOON",
    "good evening": "GOOD_EVENING",
    "how are you": "HOW_ARE_YOU",
    "what is your name": "WHAT_IS_YOUR_NAME",
    "my name is": "MY_NAME_IS",
    "i am": "I_AM",
    "i do not": "I_NOT",
    "i don't": "I_NOT",
    "i don't know": "I_NOT_KNOW",
    "you are": "YOU_ARE",
    "he is": "HE_IS",
    "she is": "SHE_IS",
    "it is": "IT_IS",
    "we are": "WE_ARE",
    "they are": "THEY_ARE",
}

WORD_GLOSS_MAP = {
    "hello": "HELLO",
    "hi": "HELLO",
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
    "read": "READ",
    "write": "WRITE",
    "speak": "SPEAK",
    "listen": "LISTEN",
    "watch": "WATCH",
    "see": "SEE",
    "know": "KNOW",
    "think": "THINK",
    "want": "WANT",
    "need": "NEED",
    "have": "HAVE",
    "has": "HAVE",
    "had": "HAVE",
    "do": "DO",
    "does": "DO",
    "did": "DO",
    "can": "CAN",
    "could": "CAN",
    "will": "WILL",
    "would": "WILL",
    "should": "SHOULD",
    "must": "MUST",
    "am": "BE",
    "is": "BE",
    "are": "BE",
    "was": "BE",
    "were": "BE",
    "be": "BE",
    "been": "BE",
    "being": "BE",
    "i": "I",
    "you": "YOU",
    "he": "HE",
    "she": "SHE",
    "it": "IT",
    "we": "WE",
    "they": "THEY",
    "me": "ME",
    "him": "HIM",
    "her": "HER",
    "us": "US",
    "them": "THEM",
    "my": "MY",
    "your": "YOUR",
    "his": "HIS",
    "her": "HER",
    "its": "ITS",
    "our": "OUR",
    "their": "THEIR",
    "this": "THIS",
    "that": "THAT",
    "these": "THESE",
    "those": "THOSE",
    "what": "WHAT",
    "who": "WHO",
    "where": "WHERE",
    "when": "WHEN",
    "why": "WHY",
    "how": "HOW",
    "which": "WHICH",
    "not": "NOT",
    "never": "NOT",
    "no": "NO",
    "nothing": "NOTHING",
    "nowhere": "NOWHERE",
    "neither": "NEITHER",
    "nor": "NOR",
    "and": "AND",
    "or": "OR",
    "but": "BUT",
    "if": "IF",
    "then": "THEN",
    "because": "BECAUSE",
    "so": "SO",
    "a": "A",
    "an": "AN",
    "the": "THE",
    "in": "IN",
    "on": "ON",
    "at": "AT",
    "to": "TO",
    "for": "FOR",
    "with": "WITH",
    "from": "FROM",
    "by": "BY",
    "about": "ABOUT",
    "as": "AS",
    "of": "OF",
    "all": "ALL",
    "some": "SOME",
    "any": "ANY",
    "many": "MANY",
    "much": "MUCH",
    "few": "FEW",
    "little": "LITTLE",
    "more": "MORE",
    "most": "MOST",
    "other": "OTHER",
    "such": "SUCH",
    "only": "ONLY",
    "own": "OWN",
    "same": "SAME",
    "very": "VERY",
    "too": "TOO",
    "also": "ALSO",
    "just": "JUST",
    "even": "EVEN",
    "still": "STILL",
    "back": "BACK",
    "here": "HERE",
    "there": "THERE",
    "where": "WHERE",
    "when": "WHEN",
    "why": "WHY",
    "how": "HOW",
    "now": "NOW",
    "then": "THEN",
    "today": "TODAY",
    "tomorrow": "TOMORROW",
    "yesterday": "YESTERDAY",
}


class TranslationService:
    def __init__(self):
        self.video_lookup = video_lookup_service
        self.nlp = nlp_processor
        self._nlp_available = True
        self._check_nlp()

    def _check_nlp(self):
        try:
            _ = self.nlp.model
        except RuntimeError:
            self._nlp_available = False

    def normalize_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def extract_phrases(self, text: str) -> Tuple[str, List[str]]:
        normalized = text.lower()
        found_phrases = []

        sorted_phrases = sorted(PHRASE_GLOSS_MAP.keys(), key=len, reverse=True)

        for phrase in sorted_phrases:
            if phrase in normalized:
                gloss = PHRASE_GLOSS_MAP[phrase]
                normalized = normalized.replace(phrase, f" {gloss} ")
                found_phrases.append(gloss)

        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized, found_phrases

    def apply_prototype_reordering(self, analysis: SentenceAnalysis) -> List[str]:
        gloss_order = []

        seen_lemmas = set()

        def add_gloss(lemma: str, force: bool = False):
            if lemma in seen_lemmas and not force:
                return
            gloss = WORD_GLOSS_MAP.get(lemma, lemma.upper())
            gloss_order.append(gloss)
            seen_lemmas.add(lemma)

        for subj in analysis.subjects:
            add_gloss(subj.lemma)

        for obj in analysis.indirect_objects:
            add_gloss(obj.lemma)

        for obj in analysis.direct_objects:
            add_gloss(obj.lemma)

        for neg in analysis.negations:
            add_gloss(neg.lemma)

        if analysis.root_verb:
            add_gloss(analysis.root_verb.lemma)

        for qword in analysis.question_words:
            if qword.lemma not in seen_lemmas:
                add_gloss(qword.lemma)

        for token in analysis.tokens:
            if token.is_punct:
                continue
            if token.lemma in seen_lemmas:
                continue
            if token.pos in ('DET', 'AUX', 'CCONJ', 'SCONJ'):
                continue
            if token.pos == 'PRON' and token.lemma in ('i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'):
                continue
            if token.is_stop and token.lemma not in ('not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nor'):
                continue
            add_gloss(token.lemma)

        return gloss_order

    def to_gloss(self, text: str) -> List[str]:
        normalized, phrase_glosses = self.extract_phrases(text)

        if not self._nlp_available:
            return self._fallback_to_gloss(normalized, phrase_glosses)

        try:
            analysis = self.nlp.analyze(normalized)
        except Exception:
            return self._fallback_to_gloss(normalized, phrase_glosses)

        if not analysis.tokens:
            return phrase_glosses if phrase_glosses else []

        has_clear_structure = (
            analysis.root_verb is not None and
            (analysis.subjects or analysis.direct_objects)
        )

        if has_clear_structure and not analysis.is_question:
            gloss_order = self.apply_prototype_reordering(analysis)
        else:
            gloss_order = self._fallback_to_gloss(normalized, phrase_glosses)

        if phrase_glosses:
            for pg in phrase_glosses:
                if pg not in gloss_order:
                    gloss_order.append(pg)

        return gloss_order

    def _fallback_to_gloss(self, normalized: str, phrase_glosses: List[str]) -> List[str]:
        tokens = normalized.split()
        gloss_sequence = []

        for token in tokens:
            if token in WORD_GLOSS_MAP.values() or token in PHRASE_GLOSS_MAP.values():
                gloss_sequence.append(token)
            else:
                gloss = WORD_GLOSS_MAP.get(token, token.upper())
                gloss_sequence.append(gloss)

        for pg in phrase_glosses:
            if pg not in gloss_sequence:
                gloss_sequence.append(pg)

        return gloss_sequence

    def translate(self, text: str) -> dict:
        normalized = self.normalize_text(text)
        gloss_sequence = self.to_gloss(normalized)

        videos = []
        for gloss in gloss_sequence:
            video_info = self.video_lookup.lookup(gloss)
            videos.append(video_info)

        return {
            "original_text": text,
            "gloss_sequence": gloss_sequence,
            "videos": videos,
            "translation_metadata": {
                "processing_mode": "nlp_assisted_prototype" if self._nlp_available else "rule_based_fallback",
                "is_question": text.strip().endswith('?'),
            }
        }


translation_service = TranslationService()