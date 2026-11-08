from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .pose import MODEL_PATH, extract_image, extract_video
from .dataset import router as dataset_router
from .feedback import analyze_feedback

app = FastAPI(title="VISION3D AI", version="0.2.0")
app.include_router(dataset_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"service": "vision3d-ai", "status": "ok", "model": "ready" if MODEL_PATH.is_file() else "missing"}


async def process(upload: UploadFile, kind: str, start_seconds=0.0, duration_seconds=60.0,
                  compare_to_example=True, tolerance_degrees=15.0):
    try:
        if not MODEL_PATH.is_file():
            raise HTTPException(503, "Falta el modelo. Ejecuta scripts/download_pose_model.py desde backend.")
        limit = (20 if kind == "image" else 100) * 1024 * 1024
        with TemporaryDirectory(prefix="vision3d-") as directory:
            # Decode content instead of trusting the original extension or MIME type.
            path = Path(directory) / "input"
            size = 0
            with path.open("wb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"El archivo supera el límite de {limit // 1024 // 1024} MB.")
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(422, "El archivo está vacío.")
            try:
                result = await run_in_threadpool(extract_image, path) if kind == "image" else await run_in_threadpool(
                    extract_video, path, start_seconds, duration_seconds)
                result = await run_in_threadpool(analyze_feedback, result, compare_to_example, tolerance_degrees)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            result["filename"] = Path(upload.filename or "archivo").name
            return result
    finally:
        await upload.close()


@app.post("/api/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    return await process(image, "image")


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...),
                  start_seconds: Annotated[float, Form(ge=0, allow_inf_nan=False)] = 0.0,
                  duration_seconds: Annotated[float, Form(gt=0, le=60, allow_inf_nan=False)] = 60.0,
                  compare_to_example: Annotated[bool, Form()] = True,
                  tolerance_degrees: Annotated[float, Form(ge=5, le=45, allow_inf_nan=False)] = 15.0):
    return await process(video, "video", start_seconds, duration_seconds, compare_to_example, tolerance_degrees)
