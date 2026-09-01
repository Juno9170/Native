"""FastAPI service implementing the Native backend <-> inference contract."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from . import scoring, transcribe


@asynccontextmanager
async def lifespan(app: FastAPI):
    transcribe.load_model()
    yield


app = FastAPI(title="Native inference", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": transcribe.device.type,
        "model": transcribe.MODEL_NAME,
    }


@app.post("/transcribe")
async def transcribe_ep(request: Request) -> dict:
    body = await request.body()
    if len(body) % 2 != 0:
        raise HTTPException(
            status_code=400, detail="pcm16 payload must have an even byte count"
        )
    return {"ipa": transcribe.transcribe_pcm16(body)}


class ScoreRequest(BaseModel):
    text: str
    actualIpa: str


class WordScore(BaseModel):
    word: str
    expected: str
    ipa: str
    score: float


class ScoreResponse(BaseModel):
    expectedIpa: str
    score: float
    words: list[WordScore]


@app.post("/score", response_model=ScoreResponse)
def score_ep(req: ScoreRequest) -> dict:
    try:
        return scoring.score(req.text, req.actualIpa)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class AlignRequest(BaseModel):
    text: str
    actualIpa: str


@app.post("/align")
def align_ep(req: AlignRequest) -> dict:
    try:
        return {"wordIndex": scoring.align_progress(req.text, req.actualIpa)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
