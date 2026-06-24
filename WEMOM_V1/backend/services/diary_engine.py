import threading
import uuid
from pathlib import Path
from typing import Dict

from Interface.diary_workflow_test import DiaryEngine

from backend.settings import DIARY_LOG, GENERATED_DIR


class DiarySession:
    def __init__(self):
        self.engine = DiaryEngine()
        self.engine.start_new_diary(str(DIARY_LOG))

    @property
    def diary_id(self) -> int:
        return self.engine.current_diary_id


def _to_media_url(path: str) -> str:
    p = Path(path).resolve()
    try:
        rel = p.relative_to(GENERATED_DIR.resolve())
    except ValueError:
        return ""
    return "/media/" + rel.as_posix()


class DiarySessionManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, DiarySession] = {}

    def start(self) -> (str, DiarySession):
        with self._lock:
            session_id = str(uuid.uuid4())
            session = DiarySession()
            self._sessions[session_id] = session
            return session_id, session

    def get(self, session_id: str) -> DiarySession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("session not found")
            return self._sessions[session_id]

    def generate_clip(self, session_id: str, text: str, user_config: dict) -> dict:
        session = self.get(session_id)
        wav_path = session.engine.generate_clip(text, user_config, str(DIARY_LOG))
        eval_res = session.engine.last_eval or {}
        return {
            "sentence_id": session.engine.sentence_count,
            "wav_url": _to_media_url(wav_path),
            "evaluation": eval_res,
        }

    def merge(self, session_id: str) -> dict:
        session = self.get(session_id)
        wav_path = session.engine.merge_diary(str(DIARY_LOG))
        macro_stats = session.engine.last_macro or {}
        return {
            "wav_url": _to_media_url(wav_path) if wav_path else "",
            "evaluation": macro_stats,
        }
