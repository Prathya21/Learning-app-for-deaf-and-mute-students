from fastapi import APIRouter, Response
from fastapi.responses import FileResponse
from app.services.video_lookup import video_lookup_service

router = APIRouter(prefix="/api", tags=["videos"])


@router.get("/videos/{word}")
async def get_video(word: str):
    result = video_lookup_service.lookup(word)
    return result


@router.get("/video/{filename}")
async def serve_video(filename: str):
    video_path = video_lookup_service.get_video_path(filename)
    if video_path and video_path.exists():
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"}
        )
    return Response(content="Video not found", status_code=404)