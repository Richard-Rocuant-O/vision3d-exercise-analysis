from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .pose import MODEL_PATH, extract_image, extract_video

app = FastAPI(title="VISION3D AI", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"service": "vision3d-ai", "status": "ok", "model": "ready" if MODEL_PATH.is_file() else "missing"}


async def process(upload: UploadFile, kind: str):
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
                result = await run_in_threadpool(extract_image if kind == "image" else extract_video, path)
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
async def analyze(video: UploadFile = File(...)):
    return await process(video, "video")
