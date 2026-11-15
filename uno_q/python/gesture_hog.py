"""On-device gesture detector: HOG features + linear SVM, trained on the UNO Q.

Shared by training (training/train_hog_svm.py) and live inference (main.py), so
both compute exactly the same features. Needs only numpy and OpenCV, which the
app runtime already has. No Edge Impulse model, no cloud, no pretrained weights.

Detection is a sliding-window search: the frame is resized at several scales,
OpenCV computes HOG for every 64x64 window on an 8 px grid, and one matrix
product scores all windows against the 4 SVM classes (background + 3 gestures).
"""

import os

import cv2
import numpy as np

WIN = 64                       # window side in pixels, after scaling
STRIDE = 8                     # window step in pixels, after scaling
FRAME = 320                    # frames are center-cropped and resized to FRAME x FRAME (the dataset size)
WINDOW_SIDES = (72, 96, 128, 160, 208, 256)  # window sizes searched, in FRAME pixels
CLASSES = ("background", "assistance", "comfort", "water_request")
NMS_IOU = 0.3

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ondevice_model", "gesture_hog_svm.npz")


def make_hog() -> cv2.HOGDescriptor:
    # 64x64 window, 16x16 blocks, 8x8 stride and cells, 9 bins -> 7*7*36 = 1764 features
    return cv2.HOGDescriptor((WIN, WIN), (16, 16), (8, 8), (8, 8), 9)


_hog = make_hog()
FEAT_DIM = _hog.getDescriptorSize()


def to_gray(img: np.ndarray, rgb: bool) -> np.ndarray:
    """uint8 grayscale from a float 0..1 or uint8 colour image (RGB dataset, BGR camera)."""
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.5 else img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)


def square_frame(gray: np.ndarray) -> np.ndarray:
    """Center-crop to a square and resize to FRAME x FRAME (a 640x480 camera frame keeps its aspect)."""
    h, w = gray.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = gray[y0:y0 + s, x0:x0 + s]
    if s != FRAME:
        crop = cv2.resize(crop, (FRAME, FRAME), interpolation=cv2.INTER_AREA)
    return crop


def window_features(gray: np.ndarray, x: float, y: float, side: float) -> np.ndarray:
    """HOG of one square window (FRAME coordinates); out-of-frame parts are edge-replicated."""
    pad = int(side) + 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x0, y0 = int(round(x)) + pad, int(round(y)) + pad
    s = max(8, int(round(side)))
    crop = padded[y0:y0 + s, x0:x0 + s]
    crop = cv2.resize(crop, (WIN, WIN), interpolation=cv2.INTER_AREA)
    return _hog.compute(crop).reshape(-1)


def dense_windows(gray: np.ndarray, sides=WINDOW_SIDES):
    """HOG of every window at every scale. Returns (features [N, FEAT_DIM], boxes [N, 4] xyxy in FRAME px)."""
    feats, boxes = [], []
    for side in sides:
        size = int(round(FRAME * WIN / side))
        if size < WIN:
            continue
        img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        desc = _hog.compute(img, winStride=(STRIDE, STRIDE), padding=(0, 0))
        n = (size - WIN) // STRIDE + 1
        desc = desc.reshape(n * n, FEAT_DIM)
        k = side / WIN
        iy, ix = np.divmod(np.arange(n * n), n)  # OpenCV orders windows row by row
        x0, y0 = ix * STRIDE * k, iy * STRIDE * k
        feats.append(desc)
        boxes.append(np.stack([x0, y0, x0 + side, y0 + side], axis=1))
    return np.concatenate(feats), np.concatenate(boxes).astype(np.float32)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class GestureDetector:
    """Loads the npz written by training/train_hog_svm.py and finds gestures in frames."""

    def __init__(self, weights: np.ndarray, bias: np.ndarray, temperature: float, threshold: float):
        self.W = weights.astype(np.float32)
        self.b = bias.astype(np.float32)
        self.T = float(temperature)
        self.threshold = float(threshold)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "GestureDetector":
        m = np.load(path, allow_pickle=False)
        assert tuple(m["classes"].tolist()) == CLASSES, "model classes do not match gesture_hog.CLASSES"
        return cls(m["W"], m["b"], float(m["temperature"]), float(m["threshold"]))

    def probabilities(self, feats: np.ndarray) -> np.ndarray:
        return softmax((feats @ self.W.T + self.b) * self.T)

    def detect_gray(self, gray320: np.ndarray, threshold=None, max_results: int = 3):
        """[(label, confidence, box xyxy in FRAME px)] best first, after NMS."""
        thr = self.threshold if threshold is None else threshold
        feats, boxes = dense_windows(gray320)
        probs = self.probabilities(feats)
        cls = probs.argmax(axis=1)
        conf = probs[np.arange(len(probs)), cls]
        keep = np.where((cls > 0) & (conf >= thr))[0]
        keep = keep[np.argsort(-conf[keep])]
        results = []
        for i in keep:
            if all(iou(boxes[i], r[2]) < NMS_IOU for r in results):
                results.append((CLASSES[cls[i]], float(conf[i]), boxes[i].tolist()))
                if len(results) >= max_results:
                    break
        return results

    def detect_frame(self, frame_bgr: np.ndarray, threshold=None):
        """Camera frame (BGR) -> detections with boxes mapped back to frame pixels."""
        gray = to_gray(frame_bgr, rgb=False)
        h, w = gray.shape
        s = min(h, w)
        y0, x0, k = (h - s) / 2, (w - s) / 2, s / FRAME
        out = []
        for label, conf, (a, b, c, d) in self.detect_gray(square_frame(gray), threshold):
            out.append((label, conf, [x0 + a * k, y0 + b * k, x0 + c * k, y0 + d * k]))
        return out
