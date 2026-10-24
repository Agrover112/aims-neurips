"""FastAPI server for the AI Evaluation Forecaster demo."""

from __future__ import annotations

import hashlib
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import ForecastEngine, Reveal

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


@dataclass
class Session:
    session_id: str
    task_id: str
    policy: str = "uncertainty"
    revealed: list[Reveal] = field(default_factory=list)
    scored: bool = False
    score: dict[str, Any] | None = None


SESSIONS: dict[str, Session] = {}
ENGINE: ForecastEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    ENGINE = ForecastEngine()
    yield


app = FastAPI(title="AI Evaluation Forecaster", lifespan=lifespan)


class SessionRequest(BaseModel):
    task_id: str
    policy: str = "uncertainty"


class AdvanceRequest(BaseModel):
    budget: int = Field(ge=0, le=31)


class PolicyRequest(BaseModel):
    policy: str


class CustomRequest(BaseModel):
    subject: str = ""
    item: str


def _engine() -> ForecastEngine:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="model is still loading")
    return ENGINE


def _session(session_id: str) -> Session:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


def _rng_for(session: Session) -> np.random.Generator:
    key = f"{session.session_id}:{len(session.revealed)}".encode("utf-8")
    return np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:16], 16))


def _preview_text(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    match = re.search(r"###Question:\s*", cleaned)
    if match:
        cleaned = cleaned[match.end() :]
    return cleaned.strip()


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _task_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "benchmark": task["benchmark"],
        "subject": {
            "display_name": task["subject"]["display_name"],
            "provider": task["subject"].get("provider", ""),
            "normalized_name": task["subject"].get("normalized_name", ""),
            "harness": task["subject"].get("harness", ""),
            "reasoning_effort": task["subject"].get("reasoning_effort", ""),
        },
        "n_acquisition": len(task["acquisition"]),
        "n_evaluation": len(task["evaluation"]),
    }


def _state_payload(session: Session) -> dict[str, Any]:
    engine = _engine()
    task = engine.tasks[session.task_id]
    prediction = engine.forecast(task, session.revealed)
    next_item = engine.next_item(task, session.revealed, session.policy, _rng_for(session))
    history = []
    for position, entry in enumerate(session.revealed, start=1):
        item = task["acquisition"][entry.index]
        history.append(
            {
                "rank": position,
                "index": entry.index,
                "item_id": item["item_id"],
                "content": _clip(_preview_text(item["content"]), 220),
                "content_chars": item["content_chars"],
                "outcome": entry.outcome,
                "cnp": entry.cnp,
                "airt": entry.airt,
            }
        )
    next_payload = None
    if next_item is not None:
        item = next_item["item"]
        next_payload = {
            "index": next_item["index"],
            "item_id": item["item_id"],
            "content": _clip(_preview_text(item["content"]), 900),
            "content_chars": item["content_chars"],
            "cnp": next_item["cnp"],
            "airt": next_item["airt"],
        }
    return {
        "session_id": session.session_id,
        "task": _task_payload(task),
        "state": {
            "policy": session.policy,
            "revealed_count": len(session.revealed),
            "remaining_count": len(task["acquisition"]) - len(session.revealed),
            "budget_max": len(task["acquisition"]),
            "scored": session.scored,
        },
        "forecast": {
            "beta": prediction.beta,
            "cnp": float(np.mean(prediction.cnp)),
            "airt": float(np.mean(prediction.airt)),
            "evaluation_count": prediction.evaluation_count,
            "curve_x": list(range(len(prediction.cnp_curve))),
            "beta_curve": prediction.beta_curve,
            "cnp_curve": prediction.cnp_curve,
            "airt_curve": prediction.airt_curve,
        },
        "history": history,
        "next_item": next_payload,
        "score": session.score if session.scored else None,
    }


def _reveal_one(session: Session) -> bool:
    engine = _engine()
    task = engine.tasks[session.task_id]
    candidate = engine.next_item(task, session.revealed, session.policy, _rng_for(session))
    if candidate is None:
        return False
    item = candidate["item"]
    session.revealed.append(
        Reveal(
            index=int(candidate["index"]),
            outcome=int(item["label"]),
            cnp=float(candidate["cnp"]),
            airt=float(candidate["airt"]),
        )
    )
    session.scored = False
    session.score = None
    return True


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "loaded" if ENGINE is not None else "loading"}


@app.get("/api/tasks")
def tasks() -> dict[str, Any]:
    engine = _engine()
    return {
        "tasks": [engine.task_summary(task) for task in engine.tasks.values()],
        "checkpoints": engine.checkpoint_summary(),
        "validation": engine.validation_summary(),
        "protocol": {
            "validation_tasks": 104,
            "orders": 10,
            "budgets": [0, 1, 3, 7, 15, 31],
            "calibration": "5-fold out-of-fold, training benchmarks only",
            "metric": "Brier ALC (lower is better)",
        },
    }


@app.post("/api/sessions")
def create_session(request: SessionRequest) -> dict[str, Any]:
    engine = _engine()
    if request.task_id not in engine.tasks:
        raise HTTPException(status_code=404, detail="unknown task")
    policy = request.policy if request.policy in {"uncertainty", "hardest", "random"} else "uncertainty"
    session = Session(
        session_id=uuid.uuid4().hex,
        task_id=request.task_id,
        policy=policy,
    )
    SESSIONS[session.session_id] = session
    return _state_payload(session)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    return _state_payload(_session(session_id))


@app.post("/api/sessions/{session_id}/reveal")
def reveal(session_id: str) -> dict[str, Any]:
    session = _session(session_id)
    if not _reveal_one(session):
        raise HTTPException(status_code=400, detail="all acquisition items have been revealed")
    return _state_payload(session)


@app.post("/api/sessions/{session_id}/advance")
def advance(session_id: str, request: AdvanceRequest) -> dict[str, Any]:
    session = _session(session_id)
    while len(session.revealed) < request.budget:
        if not _reveal_one(session):
            break
    return _state_payload(session)


@app.post("/api/sessions/{session_id}/policy")
def set_policy(session_id: str, request: PolicyRequest) -> dict[str, Any]:
    session = _session(session_id)
    if request.policy not in {"uncertainty", "hardest", "random"}:
        raise HTTPException(status_code=400, detail="unknown policy")
    session.policy = request.policy
    return _state_payload(session)


@app.post("/api/sessions/{session_id}/reset")
def reset(session_id: str) -> dict[str, Any]:
    session = _session(session_id)
    session.revealed = []
    session.scored = False
    session.score = None
    return _state_payload(session)


@app.post("/api/sessions/{session_id}/score")
def score(session_id: str) -> dict[str, Any]:
    session = _session(session_id)
    engine = _engine()
    task = engine.tasks[session.task_id]
    session.score = engine.score(task, session.revealed)
    session.scored = True
    return _state_payload(session)


@app.post("/api/custom")
def custom(request: CustomRequest) -> dict[str, Any]:
    if not request.item.strip():
        raise HTTPException(status_code=400, detail="item text is required")
    engine = _engine()
    result = engine.custom_predict(request.subject, request.item)
    result["warning"] = (
        "Out-of-distribution probe. The released checkpoint was trained on 41 "
        "benchmarks; custom text is not validated. Subject features are masked "
        "in this checkpoint, so the score is driven by item content and the "
        "label-free prior."
    )
    return result


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
