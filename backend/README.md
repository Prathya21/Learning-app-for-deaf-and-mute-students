# EduSign Backend

FastAPI backend for Indian Sign Language translation.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/videos/{word}` - Look up video for ISL gloss word
- `GET /api/video/{filename}` - Serve ISL video file
- `POST /api/translate/text-to-isl` - Convert text to ISL gloss sequence with video paths

## Translation Pipeline (Phase 3A)

The text-to-ISL translation uses **NLP-assisted prototype gloss generation**:

```
Input text
    ↓
Text normalization (lowercase, punctuation, whitespace)
    ↓
spaCy linguistic analysis (POS, lemmatization, dependencies)
    ↓
Multi-word phrase matching (e.g., "thank you" → THANK_YOU)
    ↓
Function word filtering (preserves negation, pronouns, question words)
    ↓
Prototype clause analysis (SOV reordering for simple sentences)
    ↓
Gloss normalization (uppercase, lemma-based)
    ↓
Dictionary lookup (ISL video mapping)
```

**Important**: This is a rule-assisted prototype system for educational demonstration. It does not claim linguistically complete or accurate ISL translation.

## Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── videos.py       # Video lookup endpoints
│   │   │   └── translation.py  # Translation endpoints
│   │   └── __init__.py
│   ├── services/
│   │   ├── video_lookup.py     # Dictionary lookup service
│   │   ├── isl_translation.py  # Text to ISL translation service
│   │   └── nlp_processor.py    # spaCy NLP processing
│   ├── models/
│   └── data/
│       └── isl_dictionary.json # ISL gloss to video mapping
├── requirements.txt
└── README.md
```

## Dependencies

- fastapi==0.109.0
- uvicorn==0.27.0
- pydantic==2.8.2
- python-multipart==0.0.6
- spacy==3.8.16 (with en_core_web_sm model)

## Fallback Behavior

If the spaCy model is not installed or NLP analysis fails:
- Returns clear error on startup if model missing
- Falls back to rule-based processing for individual requests if NLP fails
- Response includes `translation_metadata.processing_mode` indicating mode