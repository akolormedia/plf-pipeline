"""
Pipeline HTTP server — exposes the generation pipeline over HTTP.

This is the service that plf-tauri's pipeline.rs expects:
  GET  /status  → service and component availability
  POST /generate → run the pipeline on a video, return .raw.plf text

Run with:
  python -m pipeline.server [--host 127.0.0.1] [--port 8765]
"""

import argparse
import traceback

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .schema import UserContext
from .pipeline import run_pipeline

app = FastAPI(title="PLF Pipeline", version="0.1.0")


class GenerateRequest(BaseModel):
    video_path: str
    context: dict = Field(default_factory=dict)


def _check_component(import_name: str) -> dict:
    try:
        __import__(import_name)
        return {"available": True}
    except ImportError:
        return {"available": False}


@app.get("/status")
def status():
    return {
        "status": "ok",
        "version": "0.1.0",
        "components": {
            "scenedetect": _check_component("scenedetect"),
            "whisper": _check_component("whisper"),
            "ollama": _check_component("ollama"),
            "pyannote": _check_component("pyannote.audio"),
            "panns": _check_component("panns_inference"),
        },
    }


@app.post("/generate", response_class=PlainTextResponse)
def generate(req: GenerateRequest):
    """Run the pipeline and return the .raw.plf text.

    Unknown context keys are ignored so the Tauri client can send a
    loose JSON context without breaking UserContext construction.
    """
    valid_keys = UserContext.__dataclass_fields__.keys()
    ctx_kwargs = {k: v for k, v in req.context.items() if k in valid_keys}
    context = UserContext(**ctx_kwargs)

    try:
        result = run_pipeline(req.video_path, context)
    except Exception:
        return PlainTextResponse(
            f"PIPELINE ERROR:\n{traceback.format_exc()}",
            status_code=500,
        )

    if not result.raw_plf_text:
        errors = "\n".join(result.errors) or "unknown error"
        return PlainTextResponse(
            f"PIPELINE ERROR (no output):\n{errors}",
            status_code=500,
        )

    return PlainTextResponse(result.raw_plf_text)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="PLF pipeline server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
