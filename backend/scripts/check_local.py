"""Smoke check with a user-supplied clip; a derived image is not a new dataset."""
import asyncio
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException
from starlette.datastructures import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import analyze, analyze_image


async def check():
    source = Path(sys.argv[1]).resolve()
    output = Path(__file__).resolve().parents[2] / "data" / "local-check"
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    try:
        ok, frame = capture.read()
        assert ok, "No se pudo leer el video de prueba"
    finally:
        capture.release()
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    photo = output / "frame-from-video.jpg"
    photo.write_bytes(encoded.tobytes())
    image = await analyze_image(UploadFile(file=io.BytesIO(encoded.tobytes()), filename=photo.name))
    assert image["input_type"] == "image" and len(image["frames"]) == 1
    assert len(image["frames"][0]["landmarks"]) == 33
    assert len(image["frames"][0]["world_landmarks"]) == 33
    (output / "image-pose.json").write_text(json.dumps(image, allow_nan=False), encoding="utf-8")
    with source.open("rb") as stream:
        motion = await analyze(UploadFile(file=stream, filename=source.name))
    assert motion["detected_frames"] > 0
    times = [frame["timestamp_ms"] for frame in motion["frames"]]
    assert all(a < b for a, b in zip(times, times[1:]))
    (output / "video-pose.json").write_text(json.dumps(motion, allow_nan=False), encoding="utf-8")
    _, blank = cv2.imencode(".png", np.zeros((480, 640, 3), np.uint8))
    empty_pose = await analyze_image(UploadFile(file=io.BytesIO(blank.tobytes()), filename="synthetic-blank.png"))
    assert empty_pose["detected_frames"] == 0
    assert not empty_pose["frames"][0]["world_landmarks"]
    try:
        await analyze_image(UploadFile(file=io.BytesIO(b"not an image"), filename="invalid.jpg"))
    except HTTPException as error:
        assert error.status_code == 422
    else:
        raise AssertionError("Se aceptó un archivo inválido")
    print(json.dumps({"image_points": 33, "video_samples": len(times),
                      "video_detections": motion["detected_frames"], "timestamp_source": motion["timestamp_source"],
                      "blank_image": "no detection", "invalid_file": "422", "output": str(output)}))


if __name__ == "__main__":
    asyncio.run(check())
