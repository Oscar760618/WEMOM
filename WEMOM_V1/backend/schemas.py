from pydantic import BaseModel, Field
from typing import Optional


class UserConfig(BaseModel):
    density: float = 0.0
    pitch: float = 0.0
    velocity: float = 0.0
    scale: str = "Auto"
    grid: str = "Auto"
    rule_vel: str = "Auto"


class StartResponse(BaseModel):
    session_id: str
    diary_id: int


class AppendRequest(BaseModel):
    session_id: str
    text: str = Field(..., min_length=1, max_length=500)
    user_config: Optional[UserConfig] = None


class Evaluation(BaseModel):
    notes: int | None = None
    density: float | None = None
    duration: float | None = None
    velocity: float | None = None
    c_maj_ratio: float | None = None
    c_min_ratio: float | None = None


class AppendResponse(BaseModel):
    sentence_id: int
    wav_url: str
    evaluation: Evaluation | None = None


class MergeRequest(BaseModel):
    session_id: str


class MacroEvaluation(BaseModel):
    traj_density_var: float | None = None
    traj_pitch_var: float | None = None
    smoothness_avg_leap: float | None = None
    tonal_cohesion_var: float | None = None
    merged_eval: Evaluation | None = None


class MergeResponse(BaseModel):
    wav_url: str
    evaluation: MacroEvaluation | None = None
