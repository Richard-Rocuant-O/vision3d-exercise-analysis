"""Real end-to-end API smoke check. No synthetic claims of pose accuracy."""
import argparse
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


def upload(base, path, estimator, start=0, duration=60):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"estimator\"\r\n\r\n{estimator}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"start_seconds\"\r\n\r\n{start}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"duration_seconds\"\r\n\r\n{duration}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"clip.mp4\"\r\n"
            "Content-Type: video/mp4\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = Request(f"{base}/api/analyze", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=60)
    args = parser.parse_args()
    started = time.perf_counter()
    motion = upload(args.url, args.video, "motionbert", args.start, args.duration)
    elapsed = time.perf_counter() - started
    assert motion["schema_version"] == "2.1" and motion["selected_estimator"] == "motionbert"
    assert motion["sampling"] == "every_decoded_frame"
    assert len(motion["motionbert"]["landmark_names"]) == 17
    assert motion["motionbert"]["detected_frames"] > 0
    frames = motion["frames"]
    first_index = frames[0]["frame_index"]
    assert [f["frame_index"] for f in frames] == list(range(first_index, first_index + len(frames)))
    assert all(a["timestamp_ms"] < b["timestamp_ms"] for a, b in zip(frames, frames[1:]))
    assert frames[0]["timestamp_ms"] >= round(args.start * 1000)
    assert frames[-1]["timestamp_ms"] < round((args.start + args.duration) * 1000)
    assert motion["analysis_window"]["start_ms"] == frames[0]["timestamp_ms"]
    assert motion["analysis_window"]["end_ms"] <= round((args.start + args.duration) * 1000)
    for frame in frames:
        assert not frame["landmarks"] or len(frame["landmarks"]) == 33
        assert not frame["world_landmarks"] or len(frame["world_landmarks"]) == 33
        lifted = frame["motionbert_landmarks"]
        assert not lifted or len(lifted) == 17
        if not frame["landmarks"]:
            assert not lifted, "MotionBERT filled an interval without a person"
        for p in lifted:
            if p:
                assert all(math.isfinite(p[k]) for k in ("x", "y", "z", "visibility", "presence"))
        if lifted:
            assert all(abs(lifted[0][k]) < 1e-6 for k in ("x", "y", "z"))
    try:
        upload(args.url, args.video, "unknown-estimator")
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError("Unknown estimator was accepted")
    output = Path(__file__).resolve().parents[2] / "data" / "motionbert-check"
    output.mkdir(parents=True, exist_ok=True)
    (output / "video-pose.json").write_text(json.dumps(motion, allow_nan=False), encoding="utf-8")
    summary = {"frames": len(frames), "mediapipe_detections": motion["detected_frames"],
               "motionbert_detections": motion["motionbert"]["detected_frames"],
               "source_fps": motion["source_fps"], "device": motion["motionbert"]["device"],
               "analysis_window": motion["analysis_window"],
               "elapsed_seconds_this_machine": round(elapsed, 2), "invalid_estimator": "422",
               "output": str(output / "video-pose.json")}
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
