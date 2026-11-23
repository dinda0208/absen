"""Fine-tuning ResNet-50 (transfer learning dari ImageNet) untuk wajah karyawan.

Bisa dipanggil dari web (tombol 'Latih model') atau manual:  python -m ml.train
"""
import json
import os
import random
import time
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF_TRAIN = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
TF_EVAL = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class FaceDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def collect_dataset():
    """Baca dataset/<kode>/*.jpg. Karyawan dengan foto < MIN_FOTO_TRAIN dilewati."""
    classes, files = [], {}
    if not os.path.isdir(config.DATASET_DIR):
        return classes, files
    for kode in sorted(os.listdir(config.DATASET_DIR)):
        folder = os.path.join(config.DATASET_DIR, kode)
        if not os.path.isdir(folder):
            continue
        imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if len(imgs) >= config.MIN_FOTO_TRAIN:
            classes.append(kode)
            files[kode] = [os.path.join(folder, f) for f in imgs]
    return classes, files


def split_blocks(paths, seed=42, block=5):
    """Split 70/15/15 per BLOK 5 foto berurutan (bukan per foto), supaya foto yang
    nyaris kembar tidak tersebar ke train dan test (mencegah data leakage)."""
    blocks = [paths[i:i + block] for i in range(0, len(paths), block)]
    random.Random(seed).shuffle(blocks)
    n = len(blocks)
    n_val = max(1, round(0.15 * n))
    n_test = max(1, round(0.15 * n))
    val = blocks[:n_val]
    test = blocks[n_val:n_val + n_test]
    train = blocks[n_val + n_test:]
    flat = lambda bs: [p for b in bs for p in b]
    return flat(train), flat(val), flat(test)


def _evaluate(model, dl, device, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            n += len(y)
            ys += y.cpu().tolist()
            ps += pred.cpu().tolist()
    return loss_sum / n, correct / n, ys, ps


def _features(model, items, device):
    """Fitur 2048-d (keluaran avgpool ResNet-50) untuk daftar (path, label)."""
    body = nn.Sequential(*list(model.children())[:-1]).eval()
    dl = DataLoader(FaceDS(items, TF_EVAL), batch_size=config.BATCH_SIZE, num_workers=0)
    feats, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            feats.append(body(x.to(device)).flatten(1).cpu())
            ys += y.tolist()
    return torch.cat(feats), torch.tensor(ys)


def calibrate_threshold(gen, imp):
    """Pilih ambang kemiripan. gen = skor pasangan wajah SAMA, imp = skor wajah ORANG LAIN.
    Biaya = 2*FAR + FRR (salah menerima orang lain dihukum 2x lebih berat)."""
    import numpy as np
    gen, imp = np.asarray(gen, dtype=float), np.asarray(imp, dtype=float)
    best_t, best_cost = None, None
    for t in np.unique(np.concatenate([gen, imp])):
        far, frr = float((imp >= t).mean()), float((gen < t).mean())
        cost = 2 * far + frr
        if best_cost is None or cost <= best_cost:
            best_t, best_cost = float(t), cost
    far, frr = float((imp >= best_t).mean()), float((gen < best_t).mean())
    if far == 0 and frr == 0 and gen.min() > imp.max():
        best_t = float((gen.min() + imp.max()) / 2)   # tengah-tengah celah pemisah
    far, frr = float((imp >= best_t).mean()), float((gen < best_t).mean())
    return best_t, far, frr


def train(progress=None):
    def report(**kw):
        if progress:
            progress(**kw)

    classes, files = collect_dataset()
    if len(classes) < 2:
        raise RuntimeError(
            f"Minimal 2 karyawan dengan >= {config.MIN_FOTO_TRAIN} foto wajah. "
            f"Saat ini yang memenuhi: {len(classes)}."
        )

    tr, va, te = [], [], []
    for label, kode in enumerate(classes):
        a, b, c = split_blocks(files[kode])
        tr += [(p, label) for p in a]
        va += [(p, label) for p in b]
        te += [(p, label) for p in c]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report(state="running", percent=2,
           message=f"Perangkat: {device} | train={len(tr)}, val={len(va)}, test={len(te)}")

    dl_tr = DataLoader(FaceDS(tr, TF_TRAIN), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(FaceDS(va, TF_EVAL), batch_size=config.BATCH_SIZE, num_workers=0)
    dl_te = DataLoader(FaceDS(te, TF_EVAL), batch_size=config.BATCH_SIZE, num_workers=0)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, len(classes)))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_epochs = config.EPOCH_FREEZE + config.EPOCH_FINETUNE
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = -1, None
    t0 = time.time()

    for epoch in range(total_epochs):
        if epoch == 0:   # Tahap 1: freeze backbone, latih fc saja
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.fc.parameters(), lr=config.LR_FREEZE)
            fase = "Tahap 1 (freeze)"
        if epoch == config.EPOCH_FREEZE:   # Tahap 2: unfreeze semua, lr kecil
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=config.LR_FINETUNE)
            fase = "Tahap 2 (fine-tune)"

        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)

        v_loss, v_acc, _, _ = _evaluate(model, dl_va, device, criterion)
        hist["train_loss"].append(loss_sum / n)
        hist["train_acc"].append(correct / n)
        hist["val_loss"].append(v_loss)
        hist["val_acc"].append(v_acc)
        if v_acc >= best_acc:
            best_acc = v_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        report(state="running", percent=int(5 + 85 * (epoch + 1) / total_epochs),
               message=f"{fase} | epoch {epoch + 1}/{total_epochs} | "
                       f"train_acc={correct / n:.3f} val_acc={v_acc:.3f}")

    # ---- Evaluasi akhir pada data TEST dengan model terbaik ----
    report(message="Evaluasi pada data test...", percent=92)
    model.load_state_dict(best_state)
    t_loss, t_acc, ys, ps = _evaluate(model, dl_te, device, criterion)
    labels = list(range(len(classes)))
    rep = classification_report(ys, ps, labels=labels, target_names=classes,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(ys, ps, labels=labels)

    # ---- Template wajah (centroid fitur ResNet-50) + kalibrasi ambang verifikasi ----
    report(message="Membuat template wajah dan kalibrasi ambang...", percent=95)
    n_cls = len(classes)
    f_tr, y_tr = _features(model, tr, device)
    f_mean = f_tr.mean(0)
    nrm = lambda f: (f - f_mean) / (f - f_mean).norm(dim=1, keepdim=True).clamp_min(1e-8)
    n_tr = nrm(f_tr)
    centroids = torch.stack([n_tr[y_tr == c].mean(0) for c in range(n_cls)])
    centroids = centroids / centroids.norm(dim=1, keepdim=True)
    f_ev, y_ev = _features(model, va + te, device)
    sims = nrm(f_ev) @ centroids.T
    rows = torch.arange(len(y_ev))
    gen = sims[rows, y_ev].numpy()
    mask = torch.ones_like(sims, dtype=torch.bool)
    mask[rows, y_ev] = False
    imp = sims[mask].numpy()
    sim_thr, far, frr = calibrate_threshold(gen, imp)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save({"model": best_state, "classes": classes, "centroids": centroids,
                "mean": f_mean, "sim_threshold": sim_thr}, config.MODEL_PATH)
    _plot_history(hist)
    _plot_cm(cm, classes)

    metrics = {
        "classes": classes,
        "jumlah_data": {"train": len(tr), "val": len(va), "test": len(te)},
        "test_accuracy": t_acc,
        "precision": rep["macro avg"]["precision"],
        "recall": rep["macro avg"]["recall"],
        "f1": rep["macro avg"]["f1-score"],
        "per_kelas": {k: rep[k] for k in classes},
        "sim_threshold": sim_thr,
        "far": far,
        "frr": frr,
        "history": hist,
        "durasi_detik": round(time.time() - t0),
        "waktu": time.strftime("%d-%m-%Y %H:%M:%S"),
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _plot_history(h):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, h["train_acc"], label="Train")
    ax[0].plot(ep, h["val_acc"], label="Validasi")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, h["train_loss"], label="Train")
    ax[1].plot(ep, h["val_loss"], label="Validasi")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.MODEL_DIR, "history.png"), dpi=130)
    plt.close(fig)


def _plot_cm(cm, classes):
    size = max(4, len(classes))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya"); ax.set_title("Confusion Matrix (data test)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(config.MODEL_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    def cb(**kw):
        if "message" in kw:
            print(kw["message"])
    r = train(cb)
    print(f"Selesai. Test accuracy = {r['test_accuracy']:.4f}")