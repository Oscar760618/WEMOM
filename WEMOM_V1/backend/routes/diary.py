from fastapi import APIRouter, HTTPException

from backend.schemas import AppendRequest, AppendResponse, MergeRequest, MergeResponse, StartResponse, UserConfig
from backend.services.diary_engine import DiarySessionManager

router = APIRouter(prefix="/diary", tags=["diary"])
manager = DiarySessionManager()


@router.post("/start", response_model=StartResponse)
def start_diary():
    session_id, session = manager.start()
    return StartResponse(session_id=session_id, diary_id=session.diary_id)


@router.post("/append", response_model=AppendResponse)
def append_sentence(req: AppendRequest):
    try:
        user_config = (req.user_config or UserConfig()).model_dump()
        result = manager.generate_clip(req.session_id, req.text, user_config)
        if not result["wav_url"]:
            raise HTTPException(status_code=500, detail="WAV file not found")
        return AppendResponse(**result)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/merge", response_model=MergeResponse)
def merge_diary(req: MergeRequest):
    try:
        result = manager.merge(req.session_id)
        if not result["wav_url"]:
            raise HTTPException(status_code=400, detail="No clips to merge")
        return MergeResponse(**result)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
