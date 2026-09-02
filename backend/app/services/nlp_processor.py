import spacy
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TokenInfo:
    text: str
    lemma: str
    pos: str
    dep: str
    is_stop: bool
    is_punct: bool
    idx: int


@dataclass
class SentenceAnalysis:
    tokens: List[TokenInfo]
    is_question: bool
    root_verb: Optional[TokenInfo]
    subjects: List[TokenInfo]
    direct_objects: List[TokenInfo]
    indirect_objects: List[TokenInfo]
    negations: List[TokenInfo]
    question_words: List[TokenInfo]


class NLPProcessor:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        try:
            self._model = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install it with: python -m spacy download en_core_web_sm"
            )

    @property
    def model(self):
        return self._model

    def analyze(self, text: str) -> SentenceAnalysis:
        doc = self._model(text)

        tokens = []
        for token in doc:
            tokens.append(TokenInfo(
                text=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                dep=token.dep_,
                is_stop=token.is_stop,
                is_punct=token.is_punct,
                idx=token.i
            ))

        is_question = text.strip().endswith('?') or any(
            t.pos in ('INTJ', 'PART') and t.lemma in ('what', 'who', 'where', 'when', 'why', 'how', 'which')
            for t in tokens
        )

        root_verb = None
        subjects = []
        direct_objects = []
        indirect_objects = []
        negations = []
        question_words = []

        for token in tokens:
            if token.dep == 'ROOT' and token.pos == 'VERB':
                root_verb = token
            elif token.dep in ('nsubj', 'nsubjpass'):
                subjects.append(token)
            elif token.dep == 'dobj':
                direct_objects.append(token)
            elif token.dep == 'iobj':
                indirect_objects.append(token)
            elif token.dep == 'neg':
                negations.append(token)
            elif token.pos == 'PRON' and token.lemma in ('what', 'who', 'where', 'when', 'why', 'how', 'which'):
                question_words.append(token)
            elif token.pos == 'ADV' and token.lemma in ('what', 'who', 'where', 'when', 'why', 'how', 'which'):
                question_words.append(token)

        return SentenceAnalysis(
            tokens=tokens,
            is_question=is_question,
            root_verb=root_verb,
            subjects=subjects,
            direct_objects=direct_objects,
            indirect_objects=indirect_objects,
            negations=negations,
            question_words=question_words
        )

    def lemmatize(self, text: str) -> List[str]:
        doc = self._model(text.lower())
        return [token.lemma_ for token in doc if not token.is_punct]

    def filter_function_words(self, tokens: List[TokenInfo], preserve: Optional[List[str]] = None) -> List[TokenInfo]:
        if preserve is None:
            preserve = ['not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nor']

        filtered = []
        for token in tokens:
            if token.is_punct:
                continue
            if token.pos in ('DET', 'AUX', 'CCONJ', 'SCONJ', 'PART') and token.lemma not in preserve:
                continue
            if token.pos == 'PRON' and token.lemma in ('i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'):
                continue
            if token.is_stop and token.lemma not in preserve:
                continue
            filtered.append(token)
        return filtered


nlp_processor = NLPProcessor()