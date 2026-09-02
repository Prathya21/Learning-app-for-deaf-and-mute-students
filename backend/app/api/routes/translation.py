from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.isl_translation import translation_service

router = APIRouter(prefix="/api", tags=["translation"])


class VideoInfo(BaseModel):
    word: str
    found: bool
    video_url: Optional[str] = None
    video_file: Optional[str] = None


class TextToISLRequest(BaseModel):
    text: str


class TextToISLResponse(BaseModel):
    original_text: str
    gloss_sequence: List[str]
    videos: List[VideoInfo]


@router.post("/translate/text-to-isl", response_model=TextToISLResponse)
async def translate_text_to_isl(request: TextToISLRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = translation_service.translate(request.text)
    return result