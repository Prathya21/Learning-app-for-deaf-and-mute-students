from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import videos, translation, inference

app = FastAPI(
    title="EduSign Backend",
    description="Indian Sign Language Translation API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router)
app.include_router(translation.router)
app.include_router(inference.router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "EduSign Backend"
    }


@app.get("/")
async def root():
    return {
        "message": "EduSign API",
        "version": "0.1.0",
        "docs": "/docs"
    }