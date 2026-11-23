"""Prediksi & verifikasi identitas dengan ResNet-50 hasil fine-tune.

Verifikasi 3 lapis (semua harus lolos):
  1. Prediksi ResNet-50 == akun yang sedang login
  2. Confidence softmax >= config.THRESHOLD
  3. Kemiripan fitur ResNet-50 (2048-d) dengan template wajah akun tsb >= ambang hasil kalibrasi training
"""
import os
import threading
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import cv2
import config

_lock = threading.Lock()
_model = None
_body = None            # ResNet-50 tanpa lapisan fc (penghasil fitur 2048-d)
_classes = []
_centroids = None       # template wajah per karyawan (n_kelas x 2048)
_mean = None
_sim_thr = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

_tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(n_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, n_classes))
    return m


def load(force=False):
    global _model, _body, _classes, _centroids, _mean, _sim_thr
    with _lock:
        if _model is not None and not force:
            return True
        if not os.path.exists(config.MODEL_PATH):
            _model, _body, _classes, _centroids = None, None, [], None
            return False
        ckpt = torch.load(config.MODEL_PATH, map_location=_device)
        m = build_model(len(ckpt["classes"]))
        m.load_state_dict(ckpt["model"])
        _model = m.to(_device).eval()
        _body = nn.Sequential(*list(_model.children())[:-1]).eval()
        _classes = ckpt["classes"]
        _centroids = ckpt.get("centroids")
        _mean = ckpt.get("mean")
        _sim_thr = ckpt.get("sim_threshold")
        if _centroids is not None:
            _centroids = _centroids.cpu()
            _mean = _mean.cpu()
        return True


def is_ready():
    return load()


def _forward(face_bgr):
    img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    x = _tf(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _body(x).flatten(1)
        probs = torch.softmax(_model.fc(feat), dim=1)[0]
    return feat[0].cpu(), probs.cpu()


def predict(face_bgr):
    """Return (kode_karyawan, confidence)."""
    load()
    _, probs = _forward(face_bgr)
    conf, idx = probs.max(0)
    return _classes[idx.item()], float(conf.item())


def verify(face_bgr, kode):
    """Verifikasi wajah terhadap akun `kode`. Return dict: ok, msg, pred, conf, sim_own, sim_other."""
    load()
    feat, probs = _forward(face_bgr)
    conf, idx = probs.max(0)
    res = dict(ok=False, msg="", pred=_classes[idx.item()], conf=float(conf.item()), sim_own=None, sim_other=None)

    if _centroids is None or _sim_thr is None:
        res["msg"] = "Model dibuat dengan versi lama. Minta admin melatih ulang model (menu Registrasi Wajah)."
        return res
    if kode not in _classes:
        res["msg"] = "Wajah Anda belum masuk ke model. Minta admin melatih ulang model."
        return res

    f = feat - _mean
    f = f / f.norm().clamp_min(1e-8)
    sims = _centroids @ f
    i = _classes.index(kode)
    res["sim_own"] = float(sims[i])
    others = torch.cat([sims[:i], sims[i + 1:]])
    res["sim_other"] = float(others.max()) if len(others) else None

    if res["pred"] != kode or res["conf"] < config.THRESHOLD or res["sim_own"] < _sim_thr:
        res["msg"] = f"Wajah tidak cocok dengan akun ini (kecocokan {res['sim_own'] * 100:.0f}%). Coba lagi."
        return res
    res["ok"] = True
    return res