"""The exact shape of what goes in and out. FastAPI checks these automatically."""
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    unit: int = Field(..., ge=1, description="Machine (engine) number")
    cycle: int | None = Field(None, ge=1, description="Replay up to this cycle; omit for the latest")


class Driver(BaseModel):
    feature: str
    label: str
    value: float
    shap: float
    effect: str

class Prediction(BaseModel):
    unit: int
    cycle: int
    risk: float
    threshold: float
    alarm: bool
    rul: float
    rul_text: str
    iforest_score: float
    iforest_limit: float
    iforest_flag: bool
    ae_score: float
    ae_limit: float
    ae_flag: bool
    n_drift: int
    level: int
    level_name: str
    reasons: list[str]
    drivers: list[Driver]

class AnalysisResponse(Prediction):
    explanation: str
    actions: str
    agents_ran: list[str]
    unknown_citations: list[str]
