"""Check the live local library through the same proxy used by the browser."""
import json
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5173"


def get(path):
    with urlopen(BASE + path, timeout=60) as response:
        return response.read(), response.headers.get("Content-Type", "")


def json_get(path):
    return json.loads(get(path)[0])


def post(path, field, name, data, mime):
    boundary = "vision3d-dataset-check"
    prefix = f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\nContent-Type: {mime}\r\n\r\n'
    body = prefix.encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    request = Request(BASE + path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


summary = json_get("/api/dataset")
assert summary["loaded"] and summary["total"] > 0
assert sum(item["count"] for item in summary["labels"]) == summary["total"]
label = "Squat" if any(item["label"] == "Squat" for item in summary["labels"]) else summary["labels"][0]["label"]
page = json_get("/api/dataset/examples?" + urlencode({"label": label, "page_size": 2}))
assert 0 < len(page["items"]) <= 2
assert all(row["label"] == label and "encoded" not in row for row in page["items"])
expected = next(row["count"] for row in summary["labels"] if row["label"] == label)
assert page["total"] == expected
example = page["items"][0]
image, mime = get(example["image_url"])
assert mime == "image/png" and image.startswith(b"\x89PNG")
thumb, mime = get(example["image_url"] + "&thumbnail=true")
assert mime == "image/jpeg" and thumb.startswith(b"\xff\xd8")
try:
    get(f'/api/dataset/examples/{example["id"]}/image?version=stale')
except HTTPError as error:
    assert error.code == 409
else:
    raise AssertionError("Se aceptó una versión de dataset desactualizada")
try:
    post("/api/dataset/import", "dataset", "invalid.csv", b"wrong,columns\nx,y\n", "text/csv")
except HTTPError as error:
    assert error.code == 422
else:
    raise AssertionError("Se aceptó un CSV sin las columnas requeridas")
assert json_get("/api/dataset")["version"] == summary["version"]
pose = post("/api/analyze-image", "image", "dataset-example.png", image, "image/png")
assert pose["input_type"] == "image"
assert len(pose["frames"][0]["landmarks"]) in {0, 33}
print(json.dumps({"total": summary["total"], "labels": len(summary["labels"]), "filter": label,
                  "filtered_images": page["total"], "example_id": example["id"],
                  "detected_points": len(pose["frames"][0]["landmarks"]),
                  "invalid_csv": "422; previous dataset preserved", "stale_version": "409"}))
