# EduSign Backend

FastAPI backend for Indian Sign Language translation.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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
- `POST /api/translate/text-to-isl` - Convert text to ISL gloss sequence with video paths

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
│   │   └── isl_translation.py  # Text to ISL translation service
│   ├── models/
│   └── data/
│       └── isl_dictionary.json # ISL gloss to video mapping
├── requirements.txt
└── README.md
```