"""Local example library. Exercise labels are not technique annotations."""
import ast
import base64
import binascii
import csv
import hashlib
import math
import os
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "dataset" / "dataset_preparado.csv"
MAX_CSV_BYTES = 100 * 1024 * 1024
csv.field_size_limit(MAX_CSV_BYTES)
router = APIRouter(prefix="/api/dataset", tags=["local dataset"])


def decode_base64(value: str) -> bytes:
    value = value.strip()
    # The supplied CSV stores Python bytes literals such as b'/9j/...'. Never eval.
    if value.startswith(("b'", 'b"')):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Imagen codificada inválida.") from exc
        if not isinstance(value, bytes):
            raise ValueError("Se esperaba una imagen codificada como bytes.")
        value = value.decode("ascii")
    if value.startswith("data:image/"):
        value = value.partition(",")[2]
    try:
        result = base64.b64decode("".join(value.split()), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("La imagen no contiene base64 válido.") from exc
    if not result or len(result) > 20 * 1024 * 1024:
        raise ValueError("La imagen está vacía o supera los 20 MB.")
    return result


@dataclass
class Snapshot:
    version: str
    rows: list[dict]
    counts: Counter
    signature: tuple[int, int] | None = None

    def summary(self):
        return {"loaded": True, "filename": "dataset_preparado.csv", "version": self.version,
                "total": len(self.rows), "labels": [{"label": label, "count": count}
                for label, count in sorted(self.counts.items())]}


def parse_csv(path: Path) -> Snapshot:
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("El CSV supera los 100 MB.")
    with path.open("rb") as stream:
        version = hashlib.file_digest(stream, "sha256").hexdigest()
    rows, counts = [], Counter()
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if not {"imageName", "base64encoded", "label"}.issubset(reader.fieldnames or []):
                raise ValueError("Faltan columnas: el CSV debe incluir imageName, base64encoded y label.")
            for number, row in enumerate(reader, 2):
                if len(rows) >= 10_000:
                    raise ValueError("Esta biblioteca admite hasta 10.000 imágenes.")
                if None in row or not row.get("label") or not row.get("base64encoded"):
                    raise ValueError(f"Fila {number}: faltan etiquetas o imágenes, o el CSV está mal formado.")
                label = row["label"].strip()
                if not label or len(label) > 120:
                    raise ValueError(f"Fila {number}: etiqueta vacía o demasiado larga.")
                try:
                    decode_base64(row["base64encoded"])
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Fila {number}: {exc}") from exc
                # Retain only what the browser needs. Old poseLandmarks are not treated as 3D world data.
                rows.append({"id": len(rows), "name": (row.get("imageName") or "imagen")[:240],
                             "label": label, "source_row": row.get("source_row") or str(number - 2),
                             "validation_fold": row.get("validation_fold") or None,
                             "image_hash": row.get("image_hash") or None,
                             "encoded": row["base64encoded"]})
                counts[label] += 1
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("No se pudo leer el CSV. Usa UTF-8 y columnas separadas por comas.") from exc
    if not rows:
        raise ValueError("El CSV no contiene imágenes.")
    return Snapshot(version, rows, counts)


class DatasetStore:
    def __init__(self, path: Path):
        self.path, self.cached = path, None
        self.lock = threading.RLock()

    def snapshot(self) -> Snapshot | None:
        with self.lock:
            if not self.path.is_file():
                self.cached = None
                return None
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if self.cached is None or self.cached.signature != signature:
                snapshot = parse_csv(self.path)
                snapshot.signature = signature
                self.cached = snapshot
            return self.cached

    def install(self, path: Path):
        # Validate before replacing; failed imports leave the previous library intact.
        snapshot = parse_csv(path)
        with self.lock:
            os.replace(path, self.path)
            stat = self.path.stat()
            snapshot.signature = (stat.st_mtime_ns, stat.st_size)
            self.cached = snapshot
        return snapshot.summary()


store = DatasetStore(DATASET_PATH)


def current() -> Snapshot:
    try:
        snapshot = store.snapshot()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if snapshot is None:
        raise HTTPException(404, "Carga un CSV para abrir la biblioteca.")
    return snapshot


@router.get("")
def summary():
    if not DATASET_PATH.is_file():
        return {"loaded": False, "total": 0, "labels": []}
    return current().summary()


@router.get("/examples")
def examples(label: str | None = None, page: int = Query(1, ge=1), page_size: int = Query(8, ge=1, le=24)):
    snapshot = current()
    rows = [row for row in snapshot.rows if label is None or row["label"] == label]
    total = len(rows)
    items = []
    for row in rows[(page - 1) * page_size:page * page_size]:
        item = {key: value for key, value in row.items() if key != "encoded"}
        item["version"] = snapshot.version
        item["image_url"] = f"/api/dataset/examples/{row['id']}/image?version={snapshot.version}"
        items.append(item)
    return {"items": items, "total": total, "page": page, "pages": max(1, math.ceil(total / page_size)), "version": snapshot.version}


@router.get("/examples/{example_id}/image")
def image(example_id: int, version: str, thumbnail: bool = False):
    snapshot = current()
    if version != snapshot.version:
        raise HTTPException(409, "El dataset cambió. Actualiza la biblioteca antes de elegir otra imagen.")
    if example_id < 0 or example_id >= len(snapshot.rows):
        raise HTTPException(404, "La imagen no existe en este dataset.")
    try:
        raw = decode_base64(snapshot.rows[example_id]["encoded"])
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] * frame.shape[1] > 30_000_000:
            raise ValueError("No se pudo abrir la imagen, o supera los 30 megapíxeles.")
        if thumbnail:
            scale = min(1, 240 / max(frame.shape[:2]))
            frame = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))))
        extension, mime = (".jpg", "image/jpeg") if thumbnail else (".png", "image/png")
        ok, encoded = cv2.imencode(extension, frame)
        if not ok:
            raise ValueError("No se pudo preparar la imagen.")
    except (ValueError, UnicodeError, cv2.error) as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(encoded.tobytes(), media_type=mime,
                    headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@router.post("/import")
async def import_csv(dataset: UploadFile = File(...)):
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    path = None
    try:
        with tempfile.NamedTemporaryFile(dir=DATASET_PATH.parent, suffix=".csv", delete=False) as temp:
            path = Path(temp.name)
            total = 0
            while chunk := await dataset.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_CSV_BYTES:
                    raise HTTPException(413, "El CSV supera los 100 MB.")
                temp.write(chunk)
        try:
            return await run_in_threadpool(store.install, path)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    finally:
        await dataset.close()
        if path is not None:
            path.unlink(missing_ok=True)
