"""FastAPI service implementing the Native backend <-> inference contract."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
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
async def transcribe_ep(request: Request, denoise: bool | None = None) -> dict:
    body = await request.body()
    if len(body) % 2 != 0:
        raise HTTPException(
            status_code=400, detail="pcm16 payload must have an even byte count"
        )
    # Off the event loop: denoise + transcription are blocking CPU/GPU work,
    # and a scored chunk must never starve the reader's peek requests.
    ipa = await run_in_threadpool(transcribe.transcribe_pcm16, body, denoise)
    return {"ipa": ipa}


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
    # "prefix" (default): actual is the cumulative transcript; fitting
    # alignment, hard-capped at maxWord (chunks arrive every ~2.5 s, so the
    # position can physically advance only so far between them).
    # "window": actual is a short trailing window; local alignment against
    # the band of words starting at fromWord, never answering beyond maxWord
    # (readers realistically skip ≤2 words between updates; a further match
    # is a duplicate-word coincidence).
    mode: str = "prefix"
    fromWord: int = 0
    maxWord: int | None = None


@app.post("/align")
def align_ep(req: AlignRequest) -> dict:
    try:
        if req.mode == "window":
            return {
                "wordIndex": scoring.locate_window(
                    req.text, req.actualIpa, req.fromWord, max_word=req.maxWord
                )
            }
        return {
            "wordIndex": scoring.align_progress(
                req.text, req.actualIpa, max_word=req.maxWord
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
