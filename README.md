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
│   │   │   └── isl_translation.py
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
└── README.md
```

## Quick Start

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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

## Current Features (Phase 1)

- ✅ FastAPI backend with CORS
- ✅ Health check endpoint
- ✅ ISL dictionary with 15 sample entries
- ✅ Video lookup service
- ✅ Rule-based text-to-ISL translation (placeholder)
- ✅ React + Vite frontend
- ✅ Text input with translation
- ✅ Gloss sequence display
- ✅ Video availability display
- ✅ Frontend-backend integration

## Next Steps (Phase 2+)

- Real NLP/SOV grammar processing
- MediaPipe hand tracking
- Gesture recognition model
- Video stitching/playback
- Gujarati language support
- Speech recognition
- Text-to-speech
- YouTube integration