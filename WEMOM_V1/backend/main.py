from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.diary import router as diary_router
from backend.settings import GENERATED_DIR

app = FastAPI(title="WEMOM Diary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diary_router)

app.mount("/media", StaticFiles(directory=str(GENERATED_DIR), html=False), name="media")
