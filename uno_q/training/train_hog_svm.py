"""Train the CareGesture HOG + linear SVM gesture detector ON THE UNO Q.

Run on the board (inside the app runtime image, which has numpy + OpenCV;
scikit-learn comes from /home/arduino/cg-pylibs):

  docker run --rm --user 1000:1000 -e HOME=/tmp -e PYTHONPATH=/pylibs \
    -v /home/arduino/cg-pylibs:/pylibs -v /home/arduino/ArduinoApps/caregesture:/app -w /app \
    --entrypoint python3 ghcr.io/arduino/app-bricks/python-apps-base:0.13.2 training/train_hog_svm.py

Input: the Edge Impulse data export in this app folder (raw 320x320 RGB images
+ bounding boxes only; the cloud model files next to it are NOT read).
Output:
  python/ondevice_model/gesture_hog_svm.npz          weights used by main.py
  python/ondevice_model/gesture_hog_svm_sklearn.joblib  the fitted scikit-learn object
  training/evidence/train_log.txt, metrics.json, provenance.json
"""

import glob
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(APP, "python"))

import gesture_hog as gh  # noqa: E402
from sklearn import __version__ as sklearn_version  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

SEED = 42
C = 0.01
VAL_FRACTION = 0.2
GROUP_GAP = 20            # consecutive sample ids closer than this are one capture burst (kept on one side of the split)
THRESHOLD = 0.6           # min window probability to report a gesture
JITTER_COPIES = 2
NEG_PER_POS_IMAGE = 4
NEG_PER_EMPTY_IMAGE = 12
HARD_NEG_PER_IMAGE = 10

# Edge Impulse numbers labels 1..n in alphabetical order of the class names.
TRAIN_LABELS = {1: "assistance", 2: "comfort", 3: "water_request"}
# The EI test export indexes a 7-class list; 1, 2 and 7 are the same three gestures (by the same
# alphabetical rule and the box shapes). Only used for a secondary check.
TEST_LABELS = {1: "assistance", 2: "comfort", 7: "water_request"}

OUT_MODEL = os.path.join(APP, "python", "ondevice_model")
OUT_EVID = os.path.join(HERE, "evidence")
os.makedirs(OUT_MODEL, exist_ok=True)
os.makedirs(OUT_EVID, exist_ok=True)

_log = open(os.path.join(OUT_EVID, "train_log.txt"), "w")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    _log.write(line + "\n")
    _log.flush()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(pattern):
    hits = glob.glob(os.path.join(APP, "**", pattern), recursive=True)
    if not hits:
        sys.exit(f"dataset file {pattern} not found under {APP}")
    return hits[0]


def load_split(name):
    X = np.load(find(f"ei-*-image-X_{name}*.npy"), mmap_mode="r")
    y = json.load(open(find(f"ei-*-image-y_{name}*.json")))["samples"]
    assert len(X) == len(y), (X.shape, len(y))
    side = int(round((X.shape[1] / 3) ** 0.5))
    log(f"{name}: X {X.shape} {X.dtype} -> {len(y)} images {side}x{side}x3, value range "
        f"{float(X[0].min()):.3f}..{float(X[0].max()):.3f}")
    grays = np.stack([gh.to_gray(np.asarray(X[i]).reshape(side, side, 3), rgb=True) for i in range(len(y))])
    return grays, y


def square_box(b, scale=1.1):
    side = max(b["w"], b["h"]) * scale
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    return cx - side / 2, cy - side / 2, side


def to_xyxy(b):
    return [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]


def make_crops(grays, samples, idx, labels, rng):
    feats, ys = [], []
    for i in idx:
        g = grays[i]
        boxes = [b for b in samples[i]["boundingBoxes"] if b["label"] in labels]
        for b in boxes:
            cls = gh.CLASSES.index(labels[b["label"]])
            x, y, s = square_box(b)
            views = [(x, y, s)]
            for _ in range(JITTER_COPIES):
                s2 = s * rng.uniform(0.9, 1.1)
                views.append((x + (s - s2) / 2 + rng.uniform(-0.08, 0.08) * s, y + (s - s2) / 2 + rng.uniform(-0.08, 0.08) * s, s2))
            for vx, vy, vs in views:
                feats.append(gh.window_features(g, vx, vy, vs)); ys.append(cls)
                flipped = g[:, ::-1].copy()  # mirror image = the other hand
                feats.append(gh.window_features(flipped, gh.FRAME - vx - vs, vy, vs)); ys.append(cls)
        gt = [to_xyxy(b) for b in samples[i]["boundingBoxes"]]
        want = NEG_PER_POS_IMAGE if gt else NEG_PER_EMPTY_IMAGE
        got = 0
        for _ in range(want * 15):
            if got >= want:
                break
            s = float(rng.choice(gh.WINDOW_SIDES))
            x, y = rng.uniform(0, gh.FRAME - s), rng.uniform(0, gh.FRAME - s)
            if all(gh.iou([x, y, x + s, y + s], t) < 0.2 for t in gt):
                feats.append(gh.window_features(g, x, y, s)); ys.append(0); got += 1
    return np.array(feats, dtype=np.float32), np.array(ys)


def fit(X, y):
    clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
    clf.fit(X, y)
    return clf


def fit_temperature(X, y):
    """Scale for softmax(decision * T), fitted on out-of-fold decision values (3-fold CV)."""
    scores = cross_val_predict(LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=SEED),
                               X, y, cv=3, method="decision_function")
    best = None
    for T in np.logspace(-0.5, 1.5, 81):
        p = gh.softmax(scores * T)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if best is None or nll < best[1]:
            best = (float(T), float(nll))
    return best


def detector_from(clf, T):
    return gh.GestureDetector(clf.coef_, clf.intercept_, T, THRESHOLD)


def hard_negatives(det, grays, samples, idx):
    feats = []
    for i in idx:
        gt = [to_xyxy(b) for b in samples[i]["boundingBoxes"]]
        f, boxes = gh.dense_windows(grays[i])
        p = det.probabilities(f)
        cls, conf = p.argmax(1), p.max(1)
        cand = [j for j in np.argsort(-conf) if cls[j] > 0 and conf[j] >= 0.5
                and all(gh.iou(boxes[j], t) < 0.2 for t in gt)][:HARD_NEG_PER_IMAGE]
        feats.extend(f[cand])
    return np.array(feats, dtype=np.float32).reshape(-1, gh.FEAT_DIM)


def confusion(y_true, y_pred, names):
    m = np.zeros((len(names), len(names)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[names.index(t), names.index(p)] += 1
    return m


def print_confusion(m, names, title):
    log(title + "  (rows = truth, cols = predicted)")
    log("                " + " ".join(f"{n[:12]:>12}" for n in names))
    for n, row in zip(names, m):
        log(f"{n[:14]:>14}  " + " ".join(f"{v:>12}" for v in row))


def image_eval(det, grays, samples, idx, labels, title, threshold=THRESHOLD):
    names = ["none"] + list(gh.CLASSES[1:])
    t_true, t_pred, located, n_gesture, times = [], [], 0, 0, []
    for i in idx:
        boxes = [b for b in samples[i]["boundingBoxes"] if b["label"] in labels]
        gt_label = labels[max(boxes, key=lambda b: b["w"] * b["h"])["label"]] if boxes else "none"
        t0 = time.perf_counter()
        dets = det.detect_gray(grays[i], threshold=threshold)
        times.append(time.perf_counter() - t0)
        pred = dets[0][0] if dets else "none"
        t_true.append(gt_label); t_pred.append(pred)
        if boxes:
            n_gesture += 1
            if dets and pred == gt_label and max(gh.iou(dets[0][2], to_xyxy(b)) for b in boxes) >= 0.3:
                located += 1
    m = confusion(t_true, t_pred, names)
    acc = float(np.trace(m) / m.sum())
    res = {
        "images": len(idx),
        "accuracy": round(acc, 4),
        "gesture_images": n_gesture,
        "correct_label_and_box_iou_ge_0.3": round(located / max(1, n_gesture), 4),
        "confusion_rows_truth_cols_pred": {"labels": names, "matrix": m.tolist()},
        "per_class_recall": {n: round(float(m[k, k] / m[k].sum()), 4) if m[k].sum() else None for k, n in enumerate(names)},
        "ms_per_frame_mean": round(1000 * float(np.mean(times)), 1),
    }
    log(f"{title}: {len(idx)} images, top-detection accuracy {acc:.3f}, "
        f"label+box(IoU>=0.3) {res['correct_label_and_box_iou_ge_0.3']:.3f} of {n_gesture} gesture images, "
        f"{res['ms_per_frame_mean']} ms/frame")
    print_confusion(m, names, f"{title} confusion")
    return res


def main():
    t_start = time.time()
    started = datetime.now(timezone.utc).isoformat()
    model_name = ""
    try:
        model_name = open("/proc/device-tree/model").read().strip("\x00\n")
    except OSError:
        pass
    log(f"=== CareGesture on-device training, started {started}")
    log(f"host: {platform.node()} {platform.machine()} {platform.system()} {platform.release()} | board: {model_name}")
    log(f"python {platform.python_version()}, numpy {np.__version__}, scikit-learn {sklearn_version}, opencv {gh.cv2.__version__}, cpus {os.cpu_count()}")
    log("No pretrained weights, no cloud API: features = OpenCV HOG, classifier = scikit-learn LinearSVC from scratch.")

    grays, samples = load_split("training")
    rng = np.random.default_rng(SEED)

    # class stats + box shapes (sanity check of the label order)
    per = defaultdict(list)
    for s in samples:
        for b in s["boundingBoxes"]:
            per[b["label"]].append(b["w"] / b["h"])
    for k in sorted(per):
        log(f"label {k} = {TRAIN_LABELS.get(k, '?')}: {len(per[k])} boxes, median w/h {np.median(per[k]):.2f}")
    log(f"images without a box (background only): {sum(1 for s in samples if not s['boundingBoxes'])}")

    # split by capture burst so near-identical frames never sit on both sides
    order = sorted(range(len(samples)), key=lambda i: samples[i]["sampleId"])
    groups, cur, last = [], [], None
    for i in order:
        sid = samples[i]["sampleId"]
        if last is not None and sid - last > GROUP_GAP:
            groups.append(cur); cur = []
        cur.append(i); last = sid
    groups.append(cur)
    by_label = defaultdict(list)
    for g in groups:
        labs = Counter(b["label"] for i in g for b in samples[i]["boundingBoxes"])
        by_label[labs.most_common(1)[0][0] if labs else 0].append(g)
    train_idx, val_idx = [], []
    for lab, gs in sorted(by_label.items()):
        gs = [gs[k] for k in rng.permutation(len(gs))]
        target, n_val = VAL_FRACTION * sum(len(g) for g in gs), 0
        for g in gs:
            if n_val < target and len(gs) > 1:
                val_idx += g; n_val += len(g)
            else:
                train_idx += g
    log(f"split: {len(groups)} capture groups -> {len(train_idx)} train images, {len(val_idx)} held-out validation images")

    t0 = time.time()
    Xtr, ytr = make_crops(grays, samples, train_idx, TRAIN_LABELS, rng)
    Xva, yva = make_crops(grays, samples, val_idx, TRAIN_LABELS, np.random.default_rng(SEED + 1))
    log(f"HOG features: train {Xtr.shape} {dict(Counter(gh.CLASSES[c] for c in ytr))}, "
        f"val {Xva.shape} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    clf = fit(Xtr, ytr)
    T, nll = fit_temperature(Xtr, ytr)
    log(f"round 1: LinearSVC fit + 3-fold temperature T={T:.2f} (nll {nll:.3f}) in {time.time() - t0:.1f}s")

    t0 = time.time()
    hn = hard_negatives(detector_from(clf, T), grays, samples, train_idx)
    log(f"hard-negative mining on train images: {len(hn)} false-positive windows added ({time.time() - t0:.1f}s)")
    if len(hn):
        Xtr = np.concatenate([Xtr, hn]); ytr = np.concatenate([ytr, np.zeros(len(hn), dtype=ytr.dtype)])
    t0 = time.time()
    clf = fit(Xtr, ytr)
    T, nll = fit_temperature(Xtr, ytr)
    train_seconds = time.time() - t0
    log(f"round 2: LinearSVC fit on {Xtr.shape[0]} windows + temperature T={T:.2f} (nll {nll:.3f}) in {train_seconds:.1f}s")
    det = detector_from(clf, T)

    # --- evaluation -------------------------------------------------------
    names = list(gh.CLASSES)
    pva = clf.predict(Xva)
    m = confusion([names[c] for c in yva], [names[c] for c in pva], names)
    crop_acc = float((pva == yva).mean())
    log(f"held-out validation, window classification: accuracy {crop_acc:.3f} on {len(yva)} windows")
    print_confusion(m, names, "window confusion")
    metrics = {
        "validation_window_classification": {"accuracy": round(crop_acc, 4), "windows": int(len(yva)),
                                             "confusion_rows_truth_cols_pred": {"labels": names, "matrix": m.tolist()}},
        "validation_images": image_eval(det, grays, samples, val_idx, TRAIN_LABELS, "held-out validation images"),
        "validation_threshold_sweep": {},
    }
    for thr in (0.5, 0.7):
        r = image_eval(det, grays, samples, val_idx, TRAIN_LABELS, f"validation images @ threshold {thr}", threshold=thr)
        metrics["validation_threshold_sweep"][str(thr)] = {k: r[k] for k in ("accuracy", "correct_label_and_box_iou_ge_0.3")}

    try:
        tgrays, tsamples = load_split("testing")
        tidx = [i for i, s in enumerate(tsamples)
                if len(s["boundingBoxes"]) <= 1 and all(b["label"] in TEST_LABELS for b in s["boundingBoxes"])]
        metrics["edge_impulse_test_split"] = image_eval(det, tgrays, tsamples, tidx, TEST_LABELS,
                                                        "Edge Impulse test split (never trained on)")
        metrics["edge_impulse_test_split"]["note"] = "label ids 1/2/7 of the 7-class test export mapped to assistance/comfort/water_request"
    except Exception as e:  # secondary check only
        log(f"test split skipped: {e}")

    # --- save -------------------------------------------------------------
    path = os.path.join(OUT_MODEL, "gesture_hog_svm.npz")
    np.savez(path, W=clf.coef_.astype(np.float32), b=clf.intercept_.astype(np.float32),
             classes=np.array(gh.CLASSES), temperature=np.float32(T), threshold=np.float32(THRESHOLD),
             trained_at=np.array(started))
    try:
        import joblib
        joblib.dump(clf, os.path.join(OUT_MODEL, "gesture_hog_svm_sklearn.joblib"))
    except Exception as e:
        log(f"joblib dump skipped: {e}")
    finished = datetime.now(timezone.utc).isoformat()
    metrics.update({
        "model_file": "python/ondevice_model/gesture_hog_svm.npz",
        "model_sha256": sha256(path),
        "model_bytes": os.path.getsize(path),
        "classes": names,
        "features": f"OpenCV HOG 64x64, 9 bins, 8px cells, 16px blocks -> {gh.FEAT_DIM} dims",
        "classifier": f"scikit-learn LinearSVC(C={C}, class_weight=balanced), one-vs-rest, softmax temperature {T:.3f}",
        "training_windows": int(Xtr.shape[0]),
        "threshold": THRESHOLD,
    })
    json.dump(metrics, open(os.path.join(OUT_EVID, "metrics.json"), "w"), indent=2)
    data_files = {os.path.relpath(p, APP): {"bytes": os.path.getsize(p)} for p in
                  [find("ei-*-image-X_training*.npy"), find("ei-*-image-y_training*.json"),
                   find("ei-*-image-X_testing*.npy"), find("ei-*-image-y_testing*.json")]}
    for p in data_files:
        if p.endswith(".json"):
            data_files[p]["sha256"] = sha256(os.path.join(APP, p))
    json.dump({
        "started_utc": started, "finished_utc": finished, "wall_seconds": round(time.time() - t_start, 1),
        "board": model_name, "host": platform.node(), "machine": platform.machine(),
        "kernel": platform.release(), "python": platform.python_version(),
        "numpy": np.__version__, "scikit_learn": sklearn_version, "opencv": gh.cv2.__version__,
        "inputs": data_files,
        "not_used": ["*.eim (Edge Impulse executables)", "*.lite (Edge Impulse TFLite model)", "any network / cloud API", "any pretrained weights"],
        "outputs": ["python/ondevice_model/gesture_hog_svm.npz", "python/ondevice_model/gesture_hog_svm_sklearn.joblib",
                    "training/evidence/metrics.json", "training/evidence/train_log.txt"],
    }, open(os.path.join(OUT_EVID, "provenance.json"), "w"), indent=2)
    log(f"saved {path} ({os.path.getsize(path)} bytes, sha256 {metrics['model_sha256'][:16]}...)")
    log(f"=== finished {finished}, total {time.time() - t_start:.1f}s on {platform.machine()}")


if __name__ == "__main__":
    main()
