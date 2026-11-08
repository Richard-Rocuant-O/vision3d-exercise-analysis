"""Download the official pose checkpoint at a pinned revision; verify before use."""
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "370a9196aa3c89198b134c82476143b01c0fb32c"
SHA256 = "9811155371db4ca5d20f31a36a232d41012e12e1333882888a564d741861148f"
URL = (f"https://huggingface.co/walterzhu/MotionBERT/resolve/{REVISION}/"
       "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin")
TARGET = Path(__file__).resolve().parents[1] / "models" / "motionbert-lite.bin"


def main():
    if TARGET.is_file() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == SHA256:
        print(f"MotionBERT ya verificado: {TARGET}")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".download")
    try:
        digest = hashlib.sha256()
        with urlopen(Request(URL, headers={"User-Agent": "VISION3D-local-prototype"}), timeout=120) as source, temporary.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                destination.write(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("La descarga no coincide con la huella del modelo oficial.")
        temporary.replace(TARGET)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"MotionBERT descargado y verificado: {TARGET}")


if __name__ == "__main__":
    main()
