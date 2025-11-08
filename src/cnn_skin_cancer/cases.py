"""
Case management utilities for the DermAssist API layer.

These helpers provide lightweight persistence so the FastAPI service can expose
case history, clinician notes, and dashboard telemetry without requiring an
external database.
"""
from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Serializable prediction payload."""

    label: str
    probability: float


class CaseStatus(str, Enum):
    """Simple lifecycle states for clinician workflows."""

    intake = "intake"
    review = "review"
    escalated = "escalated"
    resolved = "resolved"


class CaseMetadata(BaseModel):
    patient_id: str = Field(..., min_length=1)
    patient_age: int = Field(..., ge=0, le=120)
    sex: str = Field(..., min_length=1)
    lesion_site: str = Field(..., min_length=1)
    priority: str = Field(default="routine")
    notes: Optional[str] = None


class ClinicianNote(BaseModel):
    author: str = Field(default="system")
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CaseRecord(BaseModel):
    id: str
    metadata: CaseMetadata
    predictions: List[Prediction] = Field(default_factory=list)
    probability_map: Dict[str, float] = Field(default_factory=dict)
    risk_score: float = 0.0
    risk_level: str = "low"
    status: CaseStatus = CaseStatus.intake
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    image_path: Optional[str] = None
    clinician_notes: List[ClinicianNote] = Field(default_factory=list)


class CaseRepository:
    """Store case artifacts + metadata on disk to keep the API stateless."""

    def __init__(self, root: Path, retention: int = 50):
        self.root = Path(root)
        self.retention = retention
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.root / "cases"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "cases.json"
        self._lock = Lock()
        self._cases: List[CaseRecord] = self._load_cases()

    # --------------------------------------------------------------------- CRUD
    def _load_cases(self) -> List[CaseRecord]:
        if not self.db_path.exists():
            return []
        raw = self.db_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        payload = json.loads(raw)
        return [CaseRecord.model_validate(item) for item in payload]

    def _persist(self) -> None:
        data = [case.model_dump(mode="json") for case in self._cases]
        self.db_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_cases(self) -> List[CaseRecord]:
        with self._lock:
            return list(self._cases)

    def get_case(self, case_id: str) -> Optional[CaseRecord]:
        with self._lock:
            for case in self._cases:
                if case.id == case_id:
                    return case
        return None

    def upsert(self, case: CaseRecord) -> CaseRecord:
        with self._lock:
            self._cases = [c for c in self._cases if c.id != case.id]
            self._cases.insert(0, case)
            self._cases = self._cases[: self.retention]
            self._persist()
            return case

    # ----------------------------------------------------------------- Mutators
    def save_image(self, case_id: str, filename: str, payload: bytes) -> Path:
        suffix = Path(filename).suffix or ".jpg"
        path = self.image_dir / f"{case_id}{suffix}"
        path.write_bytes(payload)
        return path

    def update_status(self, case_id: str, status: CaseStatus) -> Optional[CaseRecord]:
        case = self.get_case(case_id)
        if not case:
            return None
        updated = case.model_copy(
            update={
                "status": status,
                "updated_at": datetime.utcnow(),
            }
        )
        return self.upsert(updated)

    def add_note(self, case_id: str, note: ClinicianNote) -> Optional[CaseRecord]:
        case = self.get_case(case_id)
        if not case:
            return None
        notes = list(case.clinician_notes) + [note]
        updated = case.model_copy(
            update={"clinician_notes": notes, "updated_at": datetime.utcnow()}
        )
        return self.upsert(updated)
