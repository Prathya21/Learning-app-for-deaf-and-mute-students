from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.isl_translation import translation_service

router = APIRouter(prefix="/api", tags=["translation"])


class TextToISLRequest(BaseModel):
    text: str


class TextToISLResponse(BaseModel):
    original_text: str
    gloss_sequence: list[str]
    videos: list[dict]


@router.post("/translate/text-to-isl", response_model=TextToISLResponse)
async def translate_text_to_isl(request: TextToISLRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = translation_service.translate(request.text)
    return result