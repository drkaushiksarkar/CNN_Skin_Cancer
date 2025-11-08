"""FastAPI inference service for the CNN classifier."""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import tensorflow as tf
import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .cases import (
    CaseMetadata,
    CaseRecord,
    CaseRepository,
    CaseStatus,
    ClinicianNote,
    Prediction,
)
from .config import TrainingConfig, load_config

cli = typer.Typer(help="Serve the model behind a FastAPI endpoint.")

HIGH_RISK_CLASSES = {
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "actinic_keratosis",
}


class PredictionResponse(BaseModel):
    top: list[Prediction]


class CasePublic(BaseModel):
    id: str
    metadata: CaseMetadata
    status: CaseStatus
    risk_level: str
    risk_score: float
    predictions: list[Prediction]
    probability_map: dict[str, float]
    created_at: datetime
    updated_at: datetime
    image_url: str | None = None
    clinician_notes: list[ClinicianNote] = Field(default_factory=list)


class CaseIntakeResponse(BaseModel):
    case: CasePublic


class CaseListResponse(BaseModel):
    cases: list[CasePublic]


class StatusUpdateRequest(BaseModel):
    status: CaseStatus


class NoteRequest(BaseModel):
    author: str = "clinician"
    message: str


class ClassDistribution(BaseModel):
    label: str
    count: int


class DashboardResponse(BaseModel):
    total_cases: int
    high_risk: int
    avg_risk: float
    last_updated: datetime | None
    status_breakdown: dict[str, int]
    class_distribution: list[ClassDistribution]
    recent_cases: list[CasePublic]


def _prepare(image_bytes: bytes, cfg: TrainingConfig) -> np.ndarray:
    image = tf.io.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, (cfg.img_height, cfg.img_width))
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()


def _predict_from_logits(
    probs: np.ndarray, cfg: TrainingConfig, top_k: int
) -> list[Prediction]:
    order = np.argsort(probs)[::-1][:top_k]
    return [
        Prediction(label=cfg.classes[int(idx)], probability=float(probs[idx]))
        for idx in order
    ]


def _probability_map(probs: np.ndarray, cfg: TrainingConfig) -> dict[str, float]:
    return {cfg.classes[int(i)]: float(prob) for i, prob in enumerate(probs)}


def _score_case(predictions: list[Prediction], priority: str) -> tuple[float, str]:
    if not predictions:
        return 0.0, "low"
    top = predictions[0]
    score = top.probability
    priority_norm = priority.lower()
    if priority_norm == "stat":
        score = min(1.0, score + 0.2)
    elif priority_norm == "urgent":
        score = min(1.0, score + 0.1)
    malignant = top.label in HIGH_RISK_CLASSES
    if score >= 0.65 or (malignant and score >= 0.5):
        level = "high"
    elif score >= 0.35:
        level = "medium"
    else:
        level = "low"
    return score, level


def _case_to_public(case: CaseRecord) -> CasePublic:
    image_url = f"/cases/{case.id}/image" if case.image_path else None
    return CasePublic(
        id=case.id,
        metadata=case.metadata,
        status=case.status,
        risk_level=case.risk_level,
        risk_score=case.risk_score,
        predictions=case.predictions,
        probability_map=case.probability_map,
        created_at=case.created_at,
        updated_at=case.updated_at,
        image_url=image_url,
        clinician_notes=case.clinician_notes,
    )


def _dashboard_payload(cases: list[CaseRecord]) -> DashboardResponse:
    status_counts = Counter(case.status.value for case in cases)
    class_counts = Counter(
        case.predictions[0].label for case in cases if case.predictions
    )
    avg_risk = (
        round(sum(case.risk_score for case in cases) / len(cases), 3) if cases else 0.0
    )
    high_risk = sum(1 for case in cases if case.risk_level == "high")
    last_updated = max((case.updated_at for case in cases), default=None)
    status_breakdown = {
        status.value: status_counts.get(status.value, 0) for status in CaseStatus
    }
    class_distribution = [
        ClassDistribution(label=label, count=count)
        for label, count in class_counts.most_common()
    ]
    recent_cases = [_case_to_public(case) for case in cases[:6]]
    return DashboardResponse(
        total_cases=len(cases),
        high_risk=high_risk,
        avg_risk=avg_risk,
        last_updated=last_updated,
        status_breakdown=status_breakdown,
        class_distribution=class_distribution,
        recent_cases=recent_cases,
    )


def create_app(model_path: str, cfg: TrainingConfig, top_k: int) -> FastAPI:
    model = tf.keras.models.load_model(model_path)
    repo = CaseRepository(cfg.paths.out_dir / "app_state")
    app = FastAPI(title="Skin Cancer Screening API", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/classes")
    async def list_classes() -> dict[str, list[str]]:
        return {"classes": cfg.classes}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: Annotated[UploadFile, File(..., description="Image to classify")]
    ) -> PredictionResponse:
        data = await file.read()
        arr = _prepare(data, cfg)
        probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        predictions = _predict_from_logits(probs, cfg, top_k)
        return PredictionResponse(top=predictions)

    @app.post("/cases/intake", response_model=CaseIntakeResponse)
    async def intake_case(
        patient_id: Annotated[str, Form(...)],
        patient_age: Annotated[int, Form(...)],
        sex: Annotated[str, Form(...)],
        lesion_site: Annotated[str, Form(...)],
        priority: Annotated[str, Form("routine")] = "routine",
        notes: Annotated[str | None, Form(None)] = None,
        image: Annotated[UploadFile, File(..., description="Dermatoscopic image")],
    ) -> CaseIntakeResponse:
        payload = await image.read()
        arr = _prepare(payload, cfg)
        probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        predictions = _predict_from_logits(probs, cfg, top_k)
        prob_map = _probability_map(probs, cfg)
        case_id = uuid.uuid4().hex
        image_path = repo.save_image(
            case_id, image.filename or f"{case_id}.jpg", payload
        )

        metadata = CaseMetadata(
            patient_id=patient_id,
            patient_age=patient_age,
            sex=sex,
            lesion_site=lesion_site,
            priority=priority,
            notes=notes,
        )
        risk_score, risk_level = _score_case(predictions, priority)
        status = CaseStatus.review if risk_level == "high" else CaseStatus.intake

        intake_notes = []
        if notes:
            intake_notes.append(ClinicianNote(author="intake", message=notes))

        case = CaseRecord(
            id=case_id,
            metadata=metadata,
            predictions=predictions,
            probability_map=prob_map,
            risk_score=risk_score,
            risk_level=risk_level,
            status=status,
            image_path=str(image_path),
            clinician_notes=intake_notes,
        )
        repo.upsert(case)
        return CaseIntakeResponse(case=_case_to_public(case))

    @app.get("/cases", response_model=CaseListResponse)
    async def list_cases() -> CaseListResponse:
        cases = [_case_to_public(case) for case in repo.list_cases()]
        return CaseListResponse(cases=cases)

    @app.get("/cases/dashboard", response_model=DashboardResponse)
    async def dashboard() -> DashboardResponse:
        cases = repo.list_cases()
        return _dashboard_payload(cases)

    @app.get("/cases/{case_id}", response_model=CasePublic)
    async def case_detail(case_id: str) -> CasePublic:
        case = repo.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return _case_to_public(case)

    @app.post("/cases/{case_id}/status", response_model=CasePublic)
    async def update_status(case_id: str, payload: StatusUpdateRequest) -> CasePublic:
        updated = repo.update_status(case_id, payload.status)
        if not updated:
            raise HTTPException(status_code=404, detail="Case not found")
        return _case_to_public(updated)

    @app.post("/cases/{case_id}/notes", response_model=CasePublic)
    async def add_note(case_id: str, payload: NoteRequest) -> CasePublic:
        note = ClinicianNote(author=payload.author, message=payload.message)
        updated = repo.add_note(case_id, note)
        if not updated:
            raise HTTPException(status_code=404, detail="Case not found")
        return _case_to_public(updated)

    @app.get("/cases/{case_id}/image")
    async def case_image(case_id: str):
        case = repo.get_case(case_id)
        if not case or not case.image_path:
            raise HTTPException(status_code=404, detail="Case not found")
        image_path = Path(case.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image missing")
        return FileResponse(image_path, filename=image_path.name)

    return app


@cli.command()
def serve(
    model_path: Annotated[str, typer.Argument(help="Path to .keras or SavedModel")],
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to YAML config")
    ] = "config/default.yaml",
    host: Annotated[str, typer.Option(help="Host/IP to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="TCP port")] = 8000,
    top_k: Annotated[int, typer.Option(help="How many classes to return")] = 3,
):
    cfg = load_config(config)
    app = create_app(model_path, cfg, top_k)
    uvicorn.run(app, host=host, port=port)
