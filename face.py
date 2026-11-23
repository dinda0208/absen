"""Tahap preprocessing: decode gambar dari browser, deteksi wajah, crop 224x224."""
import base64
import cv2
import numpy as np
import config

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def decode_data_url(data_url):
    try:
        raw = data_url.split(",", 1)[1] if "," in data_url else data_url
        arr = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def detect_face(img_bgr):
    """Return crop wajah (BGR, IMG_SIZE x IMG_SIZE) atau None bila tidak ada wajah."""
    if img_bgr is None:
        return None
    gray = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # ambil wajah terbesar
    m = int(0.15 * w)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(img_bgr.shape[1], x + w + m), min(img_bgr.shape[0], y + h + m)
    return cv2.resize(img_bgr[y0:y1, x0:x1], (config.IMG_SIZE, config.IMG_SIZE))
