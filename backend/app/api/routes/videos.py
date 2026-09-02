from fastapi import APIRouter
from app.services.video_lookup import video_lookup_service

router = APIRouter(prefix="/api", tags=["videos"])


@router.get("/videos/{word}")
async def get_video(word: str):
    result = video_lookup_service.lookup(word)
    return result