# EduSign

AI-driven Indian Sign Language (ISL) multimodal translation dashboard for deaf and mute students in classrooms.

## Project Structure

```
signlingo/
├── frontend/          # React + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   └── __init__.py
│   │   ├── services/
│   │   │   ├── video_lookup.py
│   │   │   ├── isl_translation.py
│   │   │   └── nlp_processor.py
│   │   ├── models/
│   │   └── data/
│   │       └── isl_dictionary.json
│   ├── requirements.txt
│   └── README.md
│
├── data/
│   └── videos/        # ISL video files (not in repo)
│
├── .gitignore
├── .env.example
└── README.md
```

## Quick Start

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --reload --port 8000
```

Backend runs at: http://localhost:8000
API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /api/videos/{word} | Look up video for ISL gloss word |
| GET | /api/video/{filename} | Serve ISL video file |
| POST | /api/translate/text-to-isl | Convert text to ISL gloss + videos |

## Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Video lookup
curl http://localhost:8000/api/videos/HELLO

# Text to ISL translation
curl -X POST http://localhost:8000/api/translate/text-to-isl \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello teacher"}'
```

## Translation Pipeline

The text-to-ISL translation uses a **rule-assisted prototype gloss generation** system (not linguistically authoritative ISL translation):

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

**Important**: This is a prototype system for educational demonstration. It does not claim linguistically complete or accurate ISL translation. ISL has its own grammar, non-manual markers, spatial grammar, and regional variations that are not captured here.

## Current Features

### Phase 1 - Foundation
- ✅ FastAPI backend with CORS
- ✅ Health check endpoint
- ✅ ISL dictionary with **46** sample entries (video assets not yet acquired)
- ✅ Video lookup service (truthfully reports asset availability)
- ✅ Rule-based text-to-ISL translation (placeholder)
- ✅ React + Vite frontend
- ✅ Text input with translation
- ✅ Gloss sequence display
- ✅ Video availability display (truthfully reports `found: false` when assets missing)
- ✅ Frontend-backend integration

### Phase 2A - Sequential Video Player
- ✅ Native HTML5 video player component
- ✅ Sequential playback of gloss videos
- ✅ Play/Pause, Previous/Next, Replay, Restart controls
- ✅ Playback speed (0.5x, 0.75x, 1x, 1.25x, 1.5x)
- ✅ Keyboard accessibility (Space, ←, →, R, Shift+R)
- ✅ Active gloss indicator (playing/played/upcoming/unavailable)
- ✅ Missing video handling (skip with visible placeholder)
- ✅ Autoplay blocking handling

### Phase 3A - NLP-Assisted Gloss Generation
- ✅ spaCy integration (en_core_web_sm)
- ✅ Lemmatization (reading → READ, students → STUDENT)
- ✅ Multi-word phrase preservation (thank you → THANK_YOU)
- ✅ Function word filtering (preserves negation, question words)
- ✅ Prototype SOV reordering for simple sentences
- ✅ Negation preservation (I do not understand → NOT UNDERSTAND)
- ✅ Question word preservation (what is your name → WHAT NAME)
- ✅ Unknown word retention
- ✅ Graceful fallback to rule-based if spaCy unavailable
- ✅ Response metadata (processing_mode, is_question)

### Phase 4A - Live Hand Tracking (MediaPipe)
- ✅ Webcam capture with explicit user consent
- ✅ MediaPipe HandLandmarker integration (tasks-vision)
- ✅ 21 landmark detection per hand (up to 2 hands)
- ✅ Real-time hand skeleton visualization on canvas overlay
- ✅ Camera lifecycle management (start/stop, track cleanup)
- ✅ Normalized landmark data structure for future classification
- ✅ Throttled debug panel for landmark inspection
- ✅ Error handling (permission denied, no camera, model loading)
- ✅ Accessibility (explicit start, clear status, labeled controls)

### Phase 5A — Dataset Selection Audit
- ✅ INCLUDE dataset audit completed
- ✅ `vidit031/isl-isolated-40words` audit completed (NO-GO)
- ✅ INCLUDE-50 acquisition attempted (download blocked by Zenodo bandwidth)
- ✅ **Synthetic landmark dataset created for pipeline development only**

### Phase 5B — Architecture & Safety Audit
- ✅ Synthetic dataset provenance documented (`docs/synthetic_dataset_limitations.md`)
- ✅ Dictionary regression tests pass (truthful `found: false` for missing assets)
- ✅ Vocabulary separation architecture implemented
- ✅ Synthetic dataset metadata tagged as development-only
- ✅ No real ISL data used in synthetic dataset

## Vocabulary Separation Architecture

The system now maintains three separate vocabulary concepts:

| Concept | Purpose | Current Status |
|---------|---------|----------------|
| **Translation Vocabulary** | Glosses the NLP pipeline can generate from text | 46 glosses (Phrase + Word maps) |
| **Gesture Classifier Vocabulary** | Glosses the future gesture model will classify | Not yet trained (synthetic prototype only) |
| **Video Asset Availability** | Glosses with actual MP4 demonstration videos | 0/46 (no real video assets acquired yet) |

**Key Principle**: These three vocabularies are **independent**. The translation system can output glosses without videos. The gesture classifier can be trained on glosses without videos. The video player only attempts playback when `found: true`.

## Synthetic Dataset (Development Fixture Only)

A synthetic landmark dataset was created for ML pipeline development:

- **Location**: `backend/data/synthetic_landmarks/`
- **Type**: **Synthetic / Development Fixture Only**
- **Contains**: 805 samples, 46 classes, 15 nominal "signers"
- **Source**: Entirely programmatic (NumPy, seed=42) — **NO real ISL data**
- **Contains**: NO real MediaPipe landmarks, NO real human biomechanics, NO real signer data
- **Purpose**: ML pipeline plumbing (data loading, preprocessing, model architecture, training loop)
- **Cannot be used for**: Real ISL gesture recognition evaluation, accuracy claims, signer-independent validation

See `docs/synthetic_dataset_limitations.md` for full provenance audit.

## Test Cases (Phase 3A)

| Input | Gloss Sequence | Notes |
|-------|----------------|-------|
| "Hello teacher" | HELLO, TEACHER | Basic |
| "Thank you" | THANK_YOU | Phrase preserved |
| "The student reads the book" | STUDENT, BOOK, READ | SOV reordering |
| "I am reading a book" | BOOK, READ, I_AM | Progressive aspect |
| "I do not understand" | I_NOT, UNDERSTAND | Negation preserved |
| "What is your name?" | WHAT_IS_YOUR_NAME | Question phrase |
| "Students are learning" | STUDENT, LEARN | Plural → singular |
| "Hello quantum computer" | HELLO, QUANTUM, COMPUTER | Unknown words retained |
| "The student reads the book and writes a letter" | STUDENT, BOOK, LETTER, READ, WRITE | Multi-clause conservative |

## Next Steps (Phase 4B+)

- Expand phrase dictionary
- Improve clause analysis for complex sentences
- Add Gujarati language support
- Gesture classification from hand landmarks
- Speech recognition
- YouTube integration
- Actual ISL video assets

## Limitations

- **Not authoritative ISL translation**: The SOV reordering is a prototype heuristic
- **spaCy en_core_web_sm**: Small model, limited accuracy for complex syntax
- **No non-manual markers**: Facial expressions, body language not captured
- **No spatial grammar**: ISL spatial referencing not implemented
- **Hand landmark detection is not ISL gesture recognition**: MediaPipe provides skeleton coordinates only; sign classification is a future phase
- **Video assets missing**: Dictionary references video files that don't exist yet
- **Synthetic dataset is a development fixture only**: The synthetic landmark dataset contains NO real ISL data and cannot evaluate real ISL recognition accuracy