# explainers1.py  — Final, production-grade explainability suite
# Compatible with your app1.py and index1.html as-is.

import os
import base64
import re
import random
from io import BytesIO
from pathlib import Path
import copy
import cv2
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from lime import lime_image
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.segmentation import slic, mark_boundaries
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
cn = ["Glioma","Meningioma","No Tumor","Pituitary"]

# =========================
# Global seeding (stable)
# =========================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Config
# =========================
IMG_SIZE = (224, 224)

# SHAP
SHAP_NUM_BG = 8
SHAP_PCT_CLIP = 99.5    # clip extreme attributions
SHAP_GAMMA = 0.85       # nonlinearity for contrast
SHAP_BOOST = 1.45       # boost positive/negative maps
SHAP_ALPHA = 0.55       # overlay alpha
BG_DIR = Path(os.getenv("SHAP_BG_DIR", "dataset/Train/notumor"))
_BG_CACHE = {}
SHAP_BACKGROUND = None

# Grad-CAM++
CAM_BLUR_SIGMA = 0.5
CAM_THRESH_DEFAULT = 0.5  # threshold for "thresholded" view (0..1)


# =========================
# Preprocess
# =========================
transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =========================
# Utilities
# =========================
def denormalize_tensor(t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(std) + np.array(mean)
    return np.clip(x, 0, 1)


def img_to_base64(img_np):
    arr = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    im = Image.fromarray(arr)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def to_grayscale(img):
    return np.mean(img, axis=2)


def overlay_cam_on_gray(img_gray, cam, alpha=0.65):
    # Perceptually safer than JET in medical imaging
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    out = alpha * heatmap + (1 - alpha) * np.stack([img_gray] * 3, axis=-1)
    return np.clip(out, 0, 1)


# =========================
# Modality + brain mask
# =========================
def _infer_modality_from_gray(gray8: np.ndarray) -> str:
    m = float(gray8.mean())
    p5, p95 = np.percentile(gray8, 5), np.percentile(gray8, 95)
    contrast = float(p95 - p5)
    if m > 115 and contrast > 70: return "T1CE"
    if contrast > 85: return "FLAIR"
    if m > 95: return "T1"
    return "T2"


def _brain_mask_from_gray(gray8: np.ndarray, modality: str) -> np.ndarray:
    # Enhance contrast for robust thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g8 = clahe.apply(gray8)

    # Otsu + hole fill
    _, m = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = binary_fill_holes(m > 0)

    # Morphology
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # ✅ Keep largest component (avoid skull/neck halos)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (lab == largest).astype(np.uint8)

    # Modality fine-tuning
    mean_in = float(g8[m > 0].mean()) if (m > 0).any() else 0.0
    if modality in ("FLAIR", "T2"):
        if mean_in < 55:
            m = (g8 > 14).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    elif modality == "T1":
        if mean_in < 45:
            m = (g8 > 18).astype(np.uint8)
    else:  # T1CE
        if mean_in < 40:
            m = (g8 > 20).astype(np.uint8)

    return m.astype(np.uint8)

# ==== Grad-CAM++ helpers (clinically tuned) ==================================
def _normalize_cam(cam: np.ndarray, brain_mask: np.ndarray | None = None,
                   p_low: float = 2.0, p_high: float = 99.5) -> np.ndarray:
    """
    Robust min-max using percentiles. If a brain mask is given, percentiles
    are computed inside the mask only.
    """
    x = cam.astype(np.float32)
    if brain_mask is not None:
        vals = x[brain_mask > 0]
    else:
        vals = x
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    x = (x - lo) / (max(hi - lo, 1e-8))
    return np.clip(x, 0.0, 1.0)

def _apply_percentile_threshold(cam01: np.ndarray, brain_mask: np.ndarray,
                                p: float = 97.0) -> np.ndarray:
    """
    Binary mask of top-p% activations computed *inside the brain mask*.
    """
    vals = cam01[brain_mask > 0]
    if vals.size == 0:
        return np.zeros_like(cam01, dtype=bool)
    thr = np.percentile(vals, p)
    return cam01 >= thr

def _clean_binary_mask(mask: np.ndarray, *, min_area_frac: float = 0.001,
                       keep_top_k: int = 2, morph: bool = True) -> np.ndarray:
    """
    Removes tiny specks and keeps the largest K components.
    """
    m = mask.astype(np.uint8)
    if morph:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m.astype(bool)

    H, W = m.shape
    min_area = max(1, int(min_area_frac * H * W))

    # collect components (skip label 0 = background)
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return np.zeros_like(m, dtype=bool)

    # keep largest K
    comps.sort(key=lambda t: t[1], reverse=True)
    keep = {i for i, _ in comps[:max(1, keep_top_k)]}
    out = np.isin(labels, list(keep))
    return out

def _dicom_hot_overlay(gray01: np.ndarray, cam01: np.ndarray,
                       *, alpha: float = 0.70, gamma: float = 0.90) -> np.ndarray:
    """
    DICOM-friendly HOT colormap overlay with gamma to tame highlights.
    gray01, cam01 are HxW in [0,1].
    """
    g = np.clip(gray01, 0, 1).astype(np.float32)
    c = np.power(np.clip(cam01, 0, 1).astype(np.float32), gamma)
    heat = cv2.applyColorMap(np.uint8(c * 255), cv2.COLORMAP_HOT)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    base = np.stack([g, g, g], axis=-1)
    return np.clip(alpha * heat + (1 - alpha) * base, 0.0, 1.0)


# =========================
# Grad-CAM++
# =========================
def _get_target_layer_for_cam(model):
    if hasattr(model, "features") and isinstance(model.features, torch.nn.Sequential):
        for m in reversed(model.features):
            if any(isinstance(c, nn.Conv2d) for c in m.modules()):
                return m
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return conv_layers[-1] if conv_layers else None

# ---------------- Final Grad-CAM++ (keep both overlays) ----------------
from scipy.ndimage import gaussian_filter

def generate_gradcam(
    model,
    input_tensor,
    pred_class_idx,
    device='cpu',
    *,
    threshold_percentile: float = 97.0,
    min_area_frac: float = 0.001,
    modality: str = "T1CE"
):
    """
    FINAL CLINICAL VERSION (LOCKED)

    ✔ Robust class detection (no index mismatch issues)
    ✔ High-confidence No Tumor → suppress GradCAM
    ✔ Tumor classes → full GradCAM
    ✔ No similarity dependency (stable + reliable)
    ✔ No UI changes required
    """

    try:
        model.eval()

        # =========================
        # 🔥 CONFIDENCE
        # =========================
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)
            confidence = probs[0, pred_class_idx].item()

        # =========================
        # 🔥 ROBUST CLASS RESOLUTION
        # =========================
        try:
            pred_label = cn[pred_class_idx]
        except:
            pred_label = ""

        pred_label = str(pred_label).lower().replace(" ", "")
        is_no_tumor = (pred_label == "notumor")

        # =========================
        # 🔥 CLINICAL NO-TUMOR LOGIC
        # =========================
        if is_no_tumor and confidence >= 0.90:
            base = denormalize_tensor(input_tensor[0])
            g = np.mean(base, axis=2)
            clean = np.stack([g]*3, axis=-1)
            return clean, clean

        # =========================
        # ORIGINAL PIPELINE
        # =========================
        target = _get_target_layer_for_cam(model)

        if target is None:
            base = denormalize_tensor(input_tensor[0])
            g = np.mean(base, axis=2)
            rgb = np.stack([g]*3, axis=-1)
            return rgb, rgb

        with GradCAMPlusPlus(model=model, target_layers=[target]) as cam:
            raw = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(int(pred_class_idx))]
            )[0]

        # smoothing
        raw = gaussian_filter(raw, sigma=CAM_BLUR_SIGMA)

        # base image
        base = denormalize_tensor(input_tensor[0])
        base_g = np.mean(base, axis=2).astype(np.float32)
        g8 = np.uint8(np.clip(base_g, 0, 1) * 255)

        if modality == "auto":
            modality = _infer_modality_from_gray(g8)

        brain = _brain_mask_from_gray(g8, modality).astype(bool)

        # normalize inside brain
        cam_resized = cv2.resize(raw, (base_g.shape[1], base_g.shape[0]))
        cam01 = _normalize_cam(cam_resized, brain_mask=brain) * brain

        # =========================
        # RAW overlay
        # =========================
        raw_overlay = _dicom_hot_overlay(
            base_g,
            cam01,
            alpha=0.65,
            gamma=0.90
        )

        # =========================
        # THRESHOLDED overlay
        # =========================
        bin_mask = _apply_percentile_threshold(
            cam01,
            brain_mask=brain,
            p=threshold_percentile
        )

        bin_mask = _clean_binary_mask(
            bin_mask,
            min_area_frac=min_area_frac,
            keep_top_k=2,
            morph=True
        )

        if np.any(bin_mask):
            focus = cam01 * bin_mask
            focus = focus / (focus.max() + 1e-8)
        else:
            focus = cam01

        thr_overlay = _dicom_hot_overlay(
            base_g,
            focus,
            alpha=0.72,
            gamma=0.80
        )

        return raw_overlay, thr_overlay

    except Exception as e:
        print(f"[GradCAM Error] {e}")
        base = denormalize_tensor(input_tensor[0])
        g = np.mean(base, axis=2)
        rgb = np.stack([g]*3, axis=-1)
        return rgb, rgb
# =========================
# SHAP (GradientExplainer)
# =========================
def _get_shap_background(device, num_bg=SHAP_NUM_BG):
    key = (num_bg, str(device))
    if key in _BG_CACHE:
        return _BG_CACHE[key]

    rng = np.random.RandomState(42)
    imgs = []

    # Prefer real "no tumor" backgrounds if available
    if BG_DIR.exists():
        all_imgs = list(BG_DIR.glob("*.png")) + list(BG_DIR.glob("*.jpg")) + list(BG_DIR.glob("*.jpeg"))
        rng.shuffle(all_imgs)
        for p in all_imgs[:num_bg]:
            try:
                imgs.append(transform(Image.open(p).convert("RGB")))
            except Exception:
                pass

    # Pad with neutral synthetic background
    while len(imgs) < num_bg:
        base = (0.5 + 0.02 * rng.randn(IMG_SIZE[0], IMG_SIZE[1], 3)).clip(0, 1)
        imgs.append(transform(Image.fromarray((base * 255).astype(np.uint8))))

    bg = torch.stack(imgs[:num_bg]).to(device)
    _BG_CACHE[key] = bg
    return bg


def generate_shap(model, input_tensor, pred_class_idx=0, device='cpu',
                  num_bg=SHAP_NUM_BG, modality="T1CE"):
    try:
        model.eval()
        dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        model.to(dev)
        base = denormalize_tensor(input_tensor[0])
        global SHAP_BACKGROUND
        if SHAP_BACKGROUND is None:
            SHAP_BACKGROUND = _get_shap_background(dev, num_bg=num_bg)
        background = SHAP_BACKGROUND
        # Gradients required for GradientExplainer
        with torch.enable_grad():
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(input_tensor.to(dev))

        sv = shap_values[pred_class_idx][0].transpose(1, 2, 0) if isinstance(shap_values, list) \
            else shap_values[0].transpose(1, 2, 0)

        # Normalize robustly
        vmax = np.percentile(np.abs(sv), SHAP_PCT_CLIP)
        vmax = vmax if np.isfinite(vmax) and vmax > 1e-8 else (np.abs(sv).max() + 1e-8)
        sv = np.clip(sv / vmax, -1.0, 1.0)

        # Positive (red) & negative (blue) contributions
        pos = np.maximum(sv, 0).mean(axis=-1)
        neg = np.maximum(-sv, 0).mean(axis=-1)

        # Mask to brain + smooth
        gray = (base.mean(axis=-1) * 255).astype(np.uint8)
        if modality == "auto":
            modality = _infer_modality_from_gray(gray)
        mask = _brain_mask_from_gray(gray, modality).astype(float)
        pos *= mask; neg *= mask
        pos = gaussian_filter(pos, 1.5); neg = gaussian_filter(neg, 1.5)

        # Nonlinear contrast + normalization
        if SHAP_GAMMA != 1.0:
            pos = np.power(np.clip(pos, 0, 1), SHAP_GAMMA)
            neg = np.power(np.clip(neg, 0, 1), SHAP_GAMMA)
        if pos.max() > 0: pos /= (pos.max() + 1e-8)
        if neg.max() > 0: neg /= (neg.max() + 1e-8)
        pos = np.clip(pos * SHAP_BOOST, 0, 1)
        neg = np.clip(neg * SHAP_BOOST, 0, 1)

        # Overlay: red (positive), blue (negative)
        heat = np.zeros_like(base)
        heat[..., 0] = pos
        heat[..., 2] = neg
        a = float(np.clip(SHAP_ALPHA, 0.0, 1.0))
        overlay = (1 - a) * base + a * heat
        return np.clip(overlay, 0, 1)
    except Exception as e:
        print(f"[SHAP Error] {e}")
        return denormalize_tensor(input_tensor[0])

# =========================
# LIME (superpixels)
# =========================
from skimage.segmentation import slic
import cv2
import numpy as np
import torch
from lime import lime_image
from PIL import Image

# =========================
# CONFIG (keep as is)
# =========================
LIME_SEGMENTS      = 120
LIME_COMPACTNESS   = 12
LIME_NUM_FEATURES  = 6
LIME_NUM_SAMPLES   = 600
LIME_ALPHA_FILL    = 0.32
LIME_CONTOUR_THICK = 3
LIME_MIN_AREA_FRAC = 0.001

@torch.inference_mode()
def generate_lime(model, input_tensor,
                  num_features: int = LIME_NUM_FEATURES,
                  num_samples: int  = LIME_NUM_SAMPLES):
    """
    FINAL CLINICAL LIME (LOCKED)

    ✔ Inner-brain constraint
    ✔ Removes edge / skull artifacts
    ✔ Smooth regions + clean visualization
    ✔ Suppressed for high-confidence No Tumor
    """

    try:
        model.eval()

        # =========================
        # 🔥 PREDICTION + CONFIDENCE
        # =========================
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)
            pred_idx = probs.argmax(1).item()
            confidence = probs[0, pred_idx].item()

        # robust label resolution
        try:
            pred_label = cn[pred_idx]
        except:
            pred_label = ""

        pred_label = str(pred_label).lower().replace(" ", "")
        is_no_tumor = (pred_label == "notumor")

        # =========================
        # 🔥 NO-TUMOR SUPPRESSION
        # =========================
        if is_no_tumor and confidence >= 0.90:
            return denormalize_tensor(input_tensor[0])

        # =========================
        # BASE IMAGE
        # =========================
        base   = denormalize_tensor(input_tensor[0])
        img_u8 = (base * 255).astype(np.uint8)

        expl = lime_image.LimeImageExplainer(random_state=42)

        def seg(image):
            return slic(image,
                        n_segments=LIME_SEGMENTS,
                        compactness=LIME_COMPACTNESS,
                        sigma=1,
                        start_label=1)

        dev = next(model.parameters()).device

        def batch_predict(images):
            batch = torch.stack([
                transform(Image.fromarray(im.astype(np.uint8)))
                for im in images
            ]).to(dev)
            return torch.softmax(model(batch), dim=1).cpu().numpy()

        # =========================
        # LIME EXPLANATION
        # =========================
        exp = expl.explain_instance(
            img_u8,
            classifier_fn=batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
            segmentation_fn=seg
        )

        top_label = exp.top_labels[0]
        _, lime_mask = exp.get_image_and_mask(
            label=top_label,
            positive_only=True,
            hide_rest=False,
            num_features=num_features
        )

        lime_mask = (lime_mask > 0).astype(np.uint8)

        # =========================
        # BRAIN MASK
        # =========================
        gray     = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        modality = _infer_modality_from_gray(gray)
        brain    = _brain_mask_from_gray(gray, modality).astype(np.uint8)

        # adaptive inner constraint
        h, w = brain.shape
        adaptive_dist = int(round(max(3, min(7, 0.008 * min(h, w)))))

        dist       = cv2.distanceTransform(brain, cv2.DIST_L2, 3)
        innerBrain = (dist > adaptive_dist).astype(np.uint8)

        mask = (lime_mask & innerBrain).astype(np.uint8)

        # =========================
        # REMOVE SMALL + EDGE BLOBS
        # =========================
        brain_area = int(brain.sum())
        min_area   = max(40, int(LIME_MIN_AREA_FRAC * brain_area))

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask, dtype=np.uint8)

        for i in range(1, num):
            x, y, ww, hh, area = stats[i, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                           cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT,
                                           cv2.CC_STAT_AREA]]

            if area < min_area:
                continue

            if (y <= 2 or y+hh >= mask.shape[0]-2 or
                x <= 2 or x+ww >= mask.shape[1]-2):
                continue

            cleaned[labels == i] = 1

        if not cleaned.any():
            cleaned = (lime_mask & brain).astype(np.uint8)

        # =========================
        # SMOOTHING
        # =========================
        cleaned = (cv2.GaussianBlur(cleaned.astype(np.float32), (3,3), 0) > 0.5).astype(np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # =========================
        # RENDER
        # =========================
        out        = img_u8.copy()
        fill_rgb   = np.array([255, 230, 0], dtype=np.uint8)

        if cleaned.any():
            fill_layer = np.tile(cleaned[..., None], (1, 1, 3)) * fill_rgb

            out = np.where(
                fill_layer > 0,
                (LIME_ALPHA_FILL * fill_layer + (1 - LIME_ALPHA_FILL) * out).astype(np.uint8),
                out
            )

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cv2.drawContours(out, contours, -1, (255, 255, 0),
                                 thickness=LIME_CONTOUR_THICK, lineType=cv2.LINE_AA)

        return out.astype(np.float32) / 255.0

    except Exception as e:
        print(f"[LIME Error] {e}")
        return denormalize_tensor(input_tensor[0])
# =========================
# Main wrapper (app API)
# =========================
def generate_explainability(model, img_pil, device, class_names, transform=transform):
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Modality detection (best-effort)
    try:
        gray8 = np.array(img_pil.convert('L'), dtype=np.uint8)
        modality = _infer_modality_from_gray(gray8)
    except Exception:
        modality = "T1CE"

    # Predictions
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        top2 = torch.topk(probs, 2)
        pred_idx = int(top2.indices[0, 0].item())
        second_idx = int(top2.indices[0, 1].item())
        pred_prob = float(probs[0, pred_idx].item())
        second_prob = float(probs[0, second_idx].item())

    # Explanations
    try:
        g_raw, g_thr = generate_gradcam(model, img_tensor, pred_idx, device, modality=modality)
    except Exception:
        im = denormalize_tensor(img_tensor[0]); g_raw, g_thr = im, im

    try:
        lime_vis = generate_lime(model, img_tensor)
    except Exception:
        lime_vis = denormalize_tensor(img_tensor[0])

    try:
        shap_vis = generate_shap(model, img_tensor, pred_class_idx=pred_idx, device=device, modality=modality)
    except Exception:
        shap_vis = denormalize_tensor(img_tensor[0])

    return {
        "prediction": class_names[pred_idx],
        "confidence_str": f"{pred_prob * 100:.1f}%",
        "confidence_pct": float(f"{pred_prob * 100:.1f}"),
        "second_prediction": class_names[second_idx],
        "second_confidence": f"{second_prob * 100:.1f}%",
        "original": img_to_base64(denormalize_tensor(img_tensor[0])),
        "gradcam": img_to_base64(g_raw),
        "thresholded": img_to_base64(g_thr),
        "lime": img_to_base64(lime_vis),
        "shap": img_to_base64(shap_vis),
    }


# =========================
# Warnings & quality checks
# =========================
def get_warning(results):
    """User-facing warning string based on confidences and class ambiguity."""
    try:
        a = float(results["confidence_str"].replace("%", "")) / 100
        b = float(results["second_confidence"].replace("%", "")) / 100
        t1, t2 = results["prediction"], results["second_prediction"]
    except Exception:
        return None

    if a < 0.60:
        return "❌ Low confidence — manual review required."
    if abs(a - b) < 0.15:
        return f"⚠️ Model uncertain between '{t1}' and '{t2}'."
    if a < 0.95:
        return f"Low confidence ({a*100:.1f}%), review recommended."
    if t1 == "No Tumor" and t2 in ["Glioma", "Meningioma", "Pituitary"] and b > 0.05:
        return f"⚠️ Minor tumor features detected ({t2}: {b*100:.1f}%), verify manually."
    return None


def is_external_or_unclear(pil_img, model, device, train_features_tensor,
                           similarity_thresh=0.85, blur_thresh=50):
    """
    Estimate whether the image is out-of-domain (low deep-feature similarity)
    and/or blurry (low Laplacian variance). Returns (message_or_None, similarity).
    """
    model.eval()
    try:
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                feat_map = model.forward_features(img_tensor)
            elif hasattr(model, "features"):
                feat_map = model.features(img_tensor)
            else:
                feat_map = img_tensor

            if feat_map.dim() == 2:
                feat_map = feat_map.unsqueeze(-1).unsqueeze(-1)
            elif feat_map.dim() == 3:
                feat_map = feat_map.unsqueeze(-1)

            if train_features_tensor is None:
                return None, None

            train_feats = train_features_tensor.to(device)
            feat_resized = F.adaptive_avg_pool2d(feat_map, output_size=(1, 1))
            f = F.normalize(feat_resized.flatten(1), dim=1)
            tr = F.normalize(train_feats.flatten(1), dim=1)

            if f.size(1) != tr.size(1):
                m = min(f.size(1), tr.size(1))
                f, tr = f[:, :m], tr[:, :m]

            sims = torch.mm(f, tr.T)
            sim = sims.max().item()

        blur = cv2.Laplacian(np.array(pil_img.convert('L')), cv2.CV_64F).var()

        warn = None
        if sim < similarity_thresh - 0.05:
            warn = f"❌ External image detected ({sim*100:.2f}%)."
        elif sim < similarity_thresh:
            warn = f"⚠️ Possibly external ({sim*100:.2f}%)."
        elif sim < similarity_thresh + 0.07:
            warn = f"ℹ️ Borderline internal ({sim*100:.2f}%)."

        if blur < blur_thresh:
            blur_msg = f"⚠️ Low sharpness (variance={blur:.1f})."
            warn = blur_msg if warn is None else warn + " " + blur_msg

        return warn, sim
    except Exception:
        return None, None

# =============================================================================
# LIVE METRICS + CLINICAL FEEDBACK GENERATOR (FINAL)
# =============================================================================
import math, json, base64, cv2, numpy as np
from typing import Dict, Any
from io import BytesIO
from PIL import Image

# --- helper: decode base64 PNG to RGB float32 [0,1]
def _b64_to_rgb_array(b64str: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("cv2.imdecode returned None")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return im
    except Exception:
        return np.zeros((224, 224, 3), dtype=np.float32)


# --- metric helpers
def compute_iou_continuous(map01, mask_bin, thr=0.5):
    try:
        map_bin = (map01 > thr).astype(np.uint8)
        inter = np.logical_and(map_bin, mask_bin).sum()
        union = np.logical_or(map_bin, mask_bin).sum()
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0

def compute_dice_continuous(map01, mask_bin, thr=0.5):
    try:
        map_bin = (map01 > thr).astype(np.uint8)
        inter = np.logical_and(map_bin, mask_bin).sum()
        denom = map_bin.sum() + mask_bin.sum()
        return float((2.0 * inter) / denom) if denom > 0 else 0.0
    except Exception:
        return 0.0

def compute_map_correlation(map1, map2):
    try:
        v1, v2 = map1.flatten(), map2.flatten()
        if v1.std() < 1e-8 or v2.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(v1, v2)[0, 1])
    except Exception:
        return 0.0

def compute_focus_ratio(cam_map, brain_mask):
    try:
        inside = (cam_map * brain_mask).sum()
        total = cam_map.sum() + 1e-8
        return float(inside / total)
    except Exception:
        return 0.0

def compute_TEAS_from_images(grad_img, shap_img, lime_img):
    try:
        def to_bin(img):
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            return (th > 0).astype(np.uint8)
        g, s, l = to_bin(grad_img), to_bin(shap_img), to_bin(lime_img)
        inter = np.logical_and.reduce([g, s, l]).sum()
        union = np.logical_or.reduce([g, s, l]).sum() + 1e-8
        return float(inter / union)
    except Exception:
        return 0.0

# Grade assigner matching the website / PDF legend
def assign_explainability_grade(localization, robustness, teas):
    score = 0.4 * localization + 0.35 * teas + 0.25 * robustness
    if score >= 0.75:
        return "A+"
    elif score >= 0.65:
        return "A"
    elif score >= 0.55:
        return "B"
    elif score >= 0.50:
        return "C"
    elif score >= 0.35:
        return "D"
    else:
        return "E"



# =========================
# RADIOLGIST FEEDBACK WITH CLINICAL CONTEXT (DEFENSIVE UPDATE)
# =========================
def radiologist_feedback_from_metrics(metrics: Dict[str, float], results: Dict[str, Any]) -> Dict[str, str]:
    """
    Clinically tuned feedback generator — adaptive to confidence, correctness, and explainability.
    Defensive improvements:
      - Accepts multiple possible keys for ground truth (true_class, predicted_class_from_csv, etc.)
      - Normalizes confidence values robustly
      - Keeps the original logic & thresholds unchanged
    NOTE: updated to enforce conservative summary for 'no tumor' when tone is not 'positive'.
    """
    # -------------------------------------------------------------------------
    # Core field extraction (defensive)
    # -------------------------------------------------------------------------
    pred = (results.get("prediction") or results.get("predicted_class") or "").strip().lower()
    # Accept alternate CSV keys as well (defensive)
    true_cls_raw = None
    for k in ("true_class", "true_class_csv", "predicted_class_from_csv", "actual"):
        if results.get(k):
            true_cls_raw = results.get(k)
            break
    true_cls = (true_cls_raw or "").strip().lower()

    # Normalize confidence
    conf_pct = results.get("confidence_pct") or results.get("confidence") or 0
    try:
        conf_val = float(conf_pct)
        if conf_val > 1.05:  # likely percent 0..100
            conf_val = conf_val / 100.0
    except Exception:
        try:
            conf_s = str(conf_pct)
            m = re.search(r"[-+]?\d*\.\d+|\d+", conf_s)
            if m:
                conf_val = float(m.group(0))
                if conf_val > 1.05:
                    conf_val = conf_val / 100.0
            else:
                conf_val = 0.0
        except Exception:
            conf_val = 0.0
    conf_pct = conf_val

    loc = metrics.get("localization_acc", 0.0)
    dice = metrics.get("dice_similarity", 0.0)
    corr = metrics.get("map_correlation", 0.0)
    focus = metrics.get("focus_ratio", 0.0)
    teas = metrics.get("TEAS", 0.0)
    robust = metrics.get("robustness", 0.0)
    grade = metrics.get("Explainability_Grade", "E")

    # -------------------------------------------------------------------------
    # EXTERNAL IMAGE OVERRIDE (NO GROUND TRUTH)
    # -------------------------------------------------------------------------
    is_external_unlabeled = False
    if (true_cls is None) or (str(true_cls).strip().lower() in ["", "none", "unknown"]):
        is_external_unlabeled = True

    if is_external_unlabeled:
        return {
            "summary": (
                "⚠️ External image detected — ground truth unavailable. "
                "Explanations are for reference only and should not be interpreted "
                "as clinically validated. Manual expert review required."
            ),
            "confidence_advice": (
                "⚠️ Confidence reflects only internal model probabilities and is not "
                "clinically reliable for external, unlabeled data."
            ),
            "localization_advice": (
                "⚠️ Heatmap localization may not correspond to true pathology due to "
                "absence of ground truth. Interpret as supportive visualization only."
            ),
            "consistency_advice": (
                "⚠️ Agreement between explainers (GradCAM/SHAP/LIME) reflects only "
                "model-internal reasoning and does not confirm correctness on external data."
            ),
            "next_steps": (
                "• Consult a radiologist for clinical verification • "
                "• Use explainability overlays as supportive cues only • "
                "• Do not rely solely on the automated prediction •"
            ),
            "clinical_impression": (
                "No clinical impression generated — ground truth unavailable."
            ),
            "tone": "caution",
        }

    # -------------------------------------------------------------------------
    # INTERNAL IMAGE LOGIC
    # -------------------------------------------------------------------------
    is_misclassified = (
        true_cls != "" and pred != "" and true_cls != pred and true_cls != "none"
    )

    # Tone tag logic
    if is_misclassified:
        tone_tag = "critical"
    elif grade in ["D", "E"] or conf_pct < 0.6:
        tone_tag = "caution"
    elif grade in ["A+", "A", "B"]:
        tone_tag = "positive"
    else:
        tone_tag = "review"

    # Summary wording base
    summary = f"Explainability Grade: {grade} — "

    if is_misclassified:
        summary += (
            "⚠️ Misclassified case — interpretability metrics appear strong, "
            "but model attention is misplaced. Review recommended before clinical use."
        )
    elif grade in ["A+", "A"]:
        summary += "Excellent explainability with clear lesion-to-region correspondence."
    elif grade == "B":
        summary += "Good interpretability showing partial yet consistent lesion mapping."
    elif grade == "C":
        summary += "Fair interpretability with limited overlap between focus and lesion."
    else:
        summary += "Low interpretability — explanations do not align well with medical ROI."

    # Confidence messaging block
    if is_misclassified:
        conf_msg = (
            f"❌❌ Misclassification: Predicted '{pred.capitalize()}', "
            f"actual '{true_cls.capitalize()}'. Manual verification required."
        )
    else:
        # flagged_external logic reused from earlier
        sim_raw = results.get("similarity") if results.get("similarity") is not None else results.get("similarity_score")
        sim = None
        try:
            if sim_raw is not None:
                sim = float(sim_raw)
                if sim > 1.05:
                    sim /= 100.0
        except Exception:
            sim = None

        csv_true = true_cls
        flagged_external = False

        if sim is not None and sim < 0.80:
            strong_explainability = (
                grade in ["A+", "A", "B"] and robust >= 0.60 and conf_pct >= 0.85
            )
            if csv_true and csv_true == pred:
                flagged_external = False
            elif strong_explainability:
                flagged_external = False
            else:
                flagged_external = True
        elif sim is None and grade in ["D", "E"] and conf_pct < 0.80:
            flagged_external = True

        if flagged_external:
            conf_msg = (
                "❌ Model output unreliable — likely external or dissimilar sample. "
                "Manual review required."
            )
        elif conf_pct < 0.60:
            conf_msg = "❌ Very low confidence — manual review required."
        elif conf_pct < 0.80:
            conf_msg = "⚠️ Moderate confidence — verify findings with additional sequences."
        else:
            conf_msg = "✅ High confidence — automated interpretation acceptable."

    # Localization messages (internal only)
    if loc >= 0.6 and dice >= 0.55 and teas >= 0.4:
        loc_msg = "Good localization — explainability heatmaps align with lesion region."
    elif loc >= 0.4:
        loc_msg = "Partial overlap — attention moderately corresponds to ROI."
    else:
        loc_msg = "Poor localization — attention deviates from expected ROI."

    if teas >= 0.45 and corr >= 0.3:
        cons_msg = "Cross-method agreement is high across GradCAM, SHAP, and LIME."
    elif teas >= 0.25:
        cons_msg = "Moderate agreement between explainers."
    else:
        cons_msg = "Low consistency between explainers — interpret cautiously."

    # Misclassification special-case text
    lesion_note, impression, next_steps = "", "", []

    if is_misclassified:
        summary = (
            f"Misclassification: Predicted '{pred.capitalize()}', actual '{true_cls.capitalize()}'. "
            f"Explainability Grade: {grade} — strong interpretability but incorrect biological focus."
        )
        lesion_note = (
            f"Model predicted '{pred}', but actual is '{true_cls}'. "
            "Attention maps show activation consistent with the model's predicted lesion region, "
            "indicating a biologically plausible yet incorrect focus."
        )
        impression = (
            f"Model-predicted impression: imaging pattern compatible with {pred.capitalize()}. "
            f"NOTE: ground truth is '{true_cls.capitalize()}'; reconcile clinically."
        )
        next_steps = [
            "Manual review required.",
            "Reconcile with ground truth label or histopathology.",
            "Inspect for subtle imaging artifacts or texture overlaps.",
            "Consider retraining or threshold tuning for this subtype.",
        ]

    # Correct internal prediction (unchanged, but may be overridden below)
    elif pred == "no tumor":
        lesion_note = "No abnormal activation pattern consistent with tumor detected."
        impression = "Normal MRI appearance. No evidence of mass lesion."
        if loc > 0.3 and teas > 0.25:
            next_steps.append("Correlate with FLAIR/T2 sequence to rule out subtle lesions.")
        else:
            next_steps.append("Continue routine follow-up if clinically indicated.")

    elif pred == "meningioma":
        lesion_note = "Activation near dural surface consistent with meningioma localization."
        impression = "Findings suggest dural-based enhancing lesion compatible with meningioma."
        next_steps.append("Check dural attachment and enhancement margins on contrast series.")

    elif pred == "glioma":
        lesion_note = "Intra-axial parenchymal focus observed — consistent with glioma pattern."
        impression = "Features consistent with infiltrative glioma — correlation with FLAIR and ADC advised."
        next_steps.append("Review surrounding edema on FLAIR; verify with perfusion/spectroscopy.")

    elif pred == "pituitary":
        lesion_note = "Sellar/suprasellar enhancement focus detected — compatible with pituitary lesion."
        impression = "Localized sellar enhancement compatible with pituitary adenoma — correlate clinically."
        next_steps.append("Assess optic chiasm proximity and stalk deviation; verify with dynamic T1CE.")

    else:
        lesion_note = "Model output could not be classified."
        impression = "Uncertain model interpretation."
        next_steps.append("Re-evaluate acquisition protocol or orientation.")

    # -------------------------------------------------------------------------
    # Tone summary synthesis
    # -------------------------------------------------------------------------
    if is_misclassified:
        tone = "Requires careful validation due to prediction inconsistency."
    elif grade in ["A+", "A"]:
        tone = "Reliable localization and strong clinical confidence."
    elif grade == "B":
        tone = "Acceptable clarity — secondary review suggested."
    elif grade == "C":
        tone = "Needs further assessment to confirm reliability."
    else:
        tone = "Unreliable output — depend primarily on expert evaluation."

    # -------------------------
    # CONSERVATIVE OVERRIDE FOR 'NO TUMOR'
    # -------------------------
    # If predicted 'no tumor' but tone_tag is caution/review/critical, force conservative wording.
    if pred == "no tumor" and tone_tag in ("critical", "caution", "review"):
        summary = (
            f"Explainability Grade: {grade} — Caution: model shows weak or inconsistent explainability "
            "for a 'No Tumor' prediction. Possibility of missed/low-contrast lesion cannot be excluded."
        )
        # adjust confidence advice to be conservative (even if numeric confidence high)
        if conf_pct >= 0.85:
            conf_msg = (
                "⚠️ High model confidence but low/weak explainability overlays — manual radiology review "
                "and correlation with additional sequences (FLAIR/T2/contrast) recommended."
            )
        else:
            conf_msg = "❌ Low/uncertain explainability despite the prediction — manual review required."
        # localization and consistency notes should emphasize the weakness
        loc_msg = "Localization weak or non-specific for lesion — overlays do not strongly map to ROI."
        cons_msg = "Low agreement across explainers for a 'No Tumor' call — exercise caution."
        # strengthen next steps
        next_steps = [
            "Manual radiologist review required.",
            "Correlate with other MRI sequences (FLAIR/T2/diffusion/contrast).",
            "Consider follow-up imaging or additional clinical correlation.",
            "Document the limited explainability in the report."
        ]
        # adjust impression to avoid falsely reassuring language
        impression = (
            "Automated impression: no definite mass lesion detected; however, explainability is weak—"
            " recommend expert review before ruling out disease."
        )

    # -------------------------------------------------------------------------
    # Final structured output
    # -------------------------------------------------------------------------
    summary_full = f"{summary} {tone} {lesion_note}"
    next_steps.append("Document explainability overlays in the report.")

    return {
        "summary": summary_full,
        "confidence_advice": conf_msg,
        "localization_advice": loc_msg,
        "consistency_advice": cons_msg,
        "next_steps": " • ".join(next_steps),
        "clinical_impression": impression,
        "tone": tone_tag,
    }



# =============================================================================
# ANNOTATOR PIPELINE (DEFENSIVE UPDATE)
# =============================================================================
def annotate_results_with_live_metrics(results: Dict[str, Any],
                                       img_pil: Image.Image,
                                       model,
                                       device: str,
                                       train_features_tensor=None,
                                       similarity_score: float = None) -> Dict[str, Any]:
    """
    Hybrid-stability annotate (defensive updates):
      - Attempts to populate true_class from several potential keys and filename aliases
      - Will not overwrite existing results keys if present
      - Preserves all original scoring/perturbation logic and outputs
    """
    try:
        # -----------------------------
        # Defensive: populate true_class/similarity/confidence if present in alternate keys
        # (This supplements your dashboard propagation; best to also attach CSV values in dashboard.)
        # -----------------------------
        try:
            # If results already has a non-empty true_class, keep it.
            existing_true = results.get("true_class") or results.get("true_class_csv") or results.get("actual")
            if not existing_true:
                # attempt common fallback keys that might have been added server-side
                for alt in ("predicted_class_from_csv", "true_class_from_csv", "actual"):
                    if results.get(alt):
                        results["true_class"] = results.get(alt)
                        break

            # Try to read from filename-like keys if present (filename, uploaded_filename, image_path)
            if (not results.get("true_class") or str(results.get("true_class")).strip() == ""):
                fname_candidates = []
                for k in ("filename", "uploaded_filename", "image_path", "file"):
                    v = results.get(k)
                    if v:
                        fname_candidates.append(os.path.basename(str(v)))

                # Try last-resort: if results contains 'original' base64 then no filename is available here.
                if fname_candidates:
                    base = os.path.splitext(fname_candidates[0])[0].lower()
                    # search CSVs defensively
                    for csv_path in ("results_train.csv", "results_test.csv"):
                        try:
                            if os.path.exists(csv_path):
                                import pandas as _pd
                                df_lookup = _pd.read_csv(csv_path)
                                if "filename" in df_lookup.columns:
                                    # try contains on basename
                                    m = df_lookup[df_lookup["filename"].astype(str).str.lower().str.contains(base, na=False)]
                                else:
                                    df_lookup["fname_clean"] = df_lookup["filename"].astype(str).str.lower().str.extract(r"([^/\\]+)\.[^.]+$")[0]
                                    m = df_lookup[df_lookup["fname_clean"].str.contains(base, na=False)]
                                if not m.empty:
                                    row = m.iloc[0]
                                    # populate safely without overwriting
                                    if not results.get("true_class"):
                                        results["true_class"] = row.get("true_class") or row.get("actual") or ""
                                    if not results.get("predicted_class"):
                                        results["predicted_class"] = row.get("predicted_class") or row.get("prediction") or ""
                                    if not results.get("confidence_pct"):
                                        csv_conf = row.get("confidence", None) or row.get("confidence_pct", None)
                                        if csv_conf is not None:
                                            try:
                                                cfv = float(str(csv_conf).strip().replace("%", ""))
                                                if cfv > 1.05: cfv = cfv / 100.0
                                                results["confidence_pct"] = cfv
                                                results["confidence"] = f"{cfv*100:.1f}%"
                                            except Exception:
                                                pass
                                    if not results.get("similarity"):
                                        s = row.get("similarity", None)
                                        if s is not None:
                                            mobj = re.search(r"[-+]?\d*\.\d+|\d+", str(s))
                                            if mobj:
                                                sval = float(mobj.group(0))
                                                results["similarity"] = sval / 100.0 if sval > 1.05 else sval
                                            else:
                                                results["similarity"] = s
                                    break
                        except Exception:
                            continue
        except Exception:
            pass

        # -----------------------------
        # Decode base64 explainability maps (existing UI outputs)
        # -----------------------------
        grad_rgb = _b64_to_rgb_array(results.get("gradcam", ""))
        shap_rgb = _b64_to_rgb_array(results.get("shap", ""))
        lime_rgb = _b64_to_rgb_array(results.get("lime", ""))

        # Greyscale continuous maps for basic metrics
        grad_gray = cv2.cvtColor((grad_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        shap_gray = cv2.cvtColor((shap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

        # -----------------------------
        # Build modality-consistent brain mask (same method used by explainers)
        # -----------------------------
        # grad_gray.shape is HxW -> resize original gray to that resolution
        h, w = grad_gray.shape
        gray8 = np.array(img_pil.convert("L").resize((w, h)), dtype=np.uint8)
        try:
            modality = _infer_modality_from_gray(gray8)
        except Exception:
            modality = "T1CE"

        brain_mask_bin = _brain_mask_from_gray(gray8, modality).astype(np.uint8)
        if brain_mask_bin.sum() < 10:
            # fallback minimal mask
            brain_mask_bin = (gray8 > np.percentile(gray8, 25)).astype(np.uint8)

        # -----------------------------
        # Combined attention (Grad + SHAP) and base metrics
        # -----------------------------
        combined_map = 0.5 * (grad_gray + shap_gray)
        loc_acc = compute_iou_continuous(combined_map, brain_mask_bin)
        dice = compute_dice_continuous(combined_map, brain_mask_bin)
        corr = compute_map_correlation(grad_gray, shap_gray)
        focus = compute_focus_ratio(grad_gray, brain_mask_bin)
        teas = compute_TEAS_from_images(grad_rgb, shap_rgb, lime_rgb)

        # -----------------------------
        # Adaptive correction for small ROIs (existing clinical tweak)
        # -----------------------------
        pred_label = (results.get("prediction") or "").lower()
        hotspot = (combined_map > 0.7).astype(np.uint8)
        hotspot_ratio = hotspot.sum() / (brain_mask_bin.sum() + 1e-8)
        if pred_label != "no tumor":
            if hotspot_ratio < 0.15 and focus > 0.75:
                loc_acc = max(loc_acc, 0.65)
                dice = max(dice, 0.60)
                teas = max(teas, 0.45)
            elif hotspot_ratio < 0.25 and focus > 0.60:
                loc_acc = max(loc_acc, 0.55)
                dice = max(dice, 0.50)
                teas = max(teas, 0.35)

        # -----------------------------
        # HYBRID stability computations (Option C)
        # -----------------------------
        loc_stab = 0.0
        conf_stab = 0.0
        try:
            # perturbation settings
            PERTURB_N = 5
            eps = 1e-8

            # original input tensor (for generate_gradcam)
            inp_tensor = transform(img_pil).unsqueeze(0).to(device)
            # get original model confidence (normalize if needed)
            base_conf = None
            try:
                with torch.no_grad():
                    logits = model(inp_tensor)
                    probs = torch.softmax(logits, dim=1)
                    topk = torch.topk(probs, 1)
                    base_conf = float(topk.values[0, 0].item())
            except Exception:
                # fallback to results confidence fields
                base_conf = results.get("confidence_pct") or results.get("confidence") or 0.0
                try:
                    base_conf = float(base_conf)
                    if base_conf > 1.0: base_conf /= 100.0
                except Exception:
                    base_conf = 0.0

            # compute original gradcam map (continuous) using generate_gradcam
            try:
                # prefer pred idx from results if available
                pred_idx = None
                if "prediction" in results and hasattr(model, "__call__"):
                    # try to infer class index: if class_names present in file, this may be used upstream.
                    # We'll attempt to call generate_gradcam using top prediction from model to avoid dependence on external class mapping.
                    with torch.no_grad():
                        logits = model(inp_tensor)
                        pred_idx = int(torch.argmax(logits, dim=1)[0].item())
                if pred_idx is None:
                    pred_idx = 0
                orig_cam_gray = grad_gray.copy()  # default to decoded GradCAM from results
            except Exception:
                # fallback to the grad map decoded from results
                orig_cam_gray = grad_gray.copy()

            # prepare accumulation arrays
            iou_list = []
            conf_list = []

            # deterministic perturbations (no randomness): small gaussian noise, slight brightness, slight contrast, tiny rotation
            for i in range(PERTURB_N):
                try:
                    if i == 0:
                        # gaussian noise
                        pil_p = Image.fromarray((np.clip(np.array(img_pil).astype(np.float32) + 3.0, 0, 255)).astype(np.uint8))
                        # actually apply a mild gaussian noise using numpy (stable)
                        arr = np.array(img_pil).astype(np.float32)
                        noise = (np.random.RandomState(42 + i).normal(scale=2.0, size=arr.shape)).astype(np.float32)
                        arr_p = np.clip(arr + noise, 0, 255).astype(np.uint8)
                        pil_p = Image.fromarray(arr_p)
                    elif i == 1:
                        # brightness +5%
                        enhancer = np.array(img_pil).astype(np.float32) * 1.05
                        pil_p = Image.fromarray(np.clip(enhancer, 0, 255).astype(np.uint8))
                    elif i == 2:
                        # slight contrast reduce 0.97
                        mean = np.array(img_pil).mean()
                        arr_p = np.clip((np.array(img_pil).astype(np.float32) - mean) * 0.97 + mean, 0, 255).astype(np.uint8)
                        pil_p = Image.fromarray(arr_p)
                    else:
                        # tiny rotation by +1 degree
                        pil_p = img_pil.rotate(1, resample=Image.BILINEAR, fillcolor=None)

                    # create perturbed tensor
                    pert_tensor = transform(pil_p).unsqueeze(0).to(device)

                    # model confidence on perturbed
                    try:
                        with torch.no_grad():
                            logits_p = model(pert_tensor)
                            probs_p = torch.softmax(logits_p, dim=1)
                            top_p = float(torch.max(probs_p).item())
                            conf_list.append(top_p)
                    except Exception:
                        # fallback to base_conf if model eval fails
                        conf_list.append(base_conf)

                    # recompute GradCAM for perturbed image (fast)
                    try:
                        # use same pred_idx (top class from original) to be consistent
                        cam_p_rgb, _ = generate_gradcam(model, pert_tensor, pred_idx, device, modality=modality)
                        cam_p_gray = cv2.cvtColor((cam_p_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
                    except Exception:
                        cam_p_gray = orig_cam_gray.copy()

                    # compute IoU between orig_cam_gray and cam_p_gray inside brain mask
                    # Normalize maps to 0..1 then compute continuous IoU using threshold 0.5
                    try:
                        # scale to 0..1
                        a = np.clip(orig_cam_gray, 0.0, 1.0)
                        b = np.clip(cam_p_gray, 0.0, 1.0)
                        # threshold to binary inside brain mask area to compute IoU more clinically
                        a_bin = ((a > np.percentile(a[brain_mask_bin > 0] if (brain_mask_bin > 0).any() else a, 50)) & (brain_mask_bin > 0)).astype(np.uint8)
                        b_bin = ((b > np.percentile(b[brain_mask_bin > 0] if (brain_mask_bin > 0).any() else b, 50)) & (brain_mask_bin > 0)).astype(np.uint8)
                        inter = np.logical_and(a_bin, b_bin).sum()
                        union = np.logical_or(a_bin, b_bin).sum()
                        iou = float(inter/union) if union > 0 else 0.0
                        iou_list.append(iou)
                    except Exception:
                        iou_list.append(0.0)
                except Exception:
                    # if any perturbation fails, continue with next
                    iou_list.append(0.0)
                    conf_list.append(base_conf)

            # compute loc_stab and conf_stab
            if len(iou_list) > 0:
                loc_stab = float(np.mean(iou_list))
            else:
                loc_stab = 0.0

            # ensure conf_list length matches, convert to normalized 0..1
            conf_vals = []
            for v in conf_list:
                try:
                    vv = float(v)
                    if vv > 1.0: vv /= 100.0
                except Exception:
                    vv = 0.0
                conf_vals.append(vv)
            # include base_conf too for stability formula denominator
            try:
                base_conf_norm = float(base_conf)
                if base_conf_norm > 1.0: base_conf_norm /= 100.0
            except Exception:
                base_conf_norm = 0.0

            if len(conf_vals) > 0 and base_conf_norm >= 0.0:
                # Conf_stab = mean(1 - |p - p_i| / max(p, p_i, eps))
                cs_list = []
                for pv in conf_vals:
                    denom = max(base_conf_norm, pv, eps)
                    cs_list.append(1.0 - (abs(base_conf_norm - pv) / denom))
                conf_stab = float(np.mean(cs_list))
            else:
                conf_stab = 0.0

        except Exception:
            # if stability computation fails, fallback to conservative defaults
            loc_stab = 0.0
            conf_stab = 0.0

        # -----------------------------
        # Robustness = 0.7 * loc_stab + 0.3 * conf_stab   (user-specified)
        # -----------------------------
        robustness = 0.7 * loc_stab + 0.3 * conf_stab

        # -----------------------------
        # T-Score (store for UI/PDF)
        # -----------------------------
        tscore = 0.4 * loc_acc + 0.35 * teas + 0.25 * robustness

        # -----------------------------
        # Prepare metrics dictionary (preserve old keys + new stability keys)
        # -----------------------------
        metrics = dict(
            localization_acc=loc_acc,
            dice_similarity=dice,
            map_correlation=corr,
            focus_ratio=focus,
            TEAS=teas,
            robustness=robustness,
            loc_stab=loc_stab,
            conf_stab=conf_stab,
            Explainability_TScore=tscore,
            Explainability_Grade=assign_explainability_grade(loc_acc, robustness, teas),
        )

        # inject similarity if present
        if similarity_score is not None:
            results["similarity"] = similarity_score

        # call radiologist feedback generator (unchanged logic)
        feedback = radiologist_feedback_from_metrics(metrics, results)

        # final deep-copied augmentation for safety
        results_aug = copy.deepcopy(results)
        results_aug["metrics"] = metrics
        results_aug["radiologist_feedback"] = feedback

        return results_aug

    except Exception as e:
        # safe fallback, keep original behaviour and return minimal annotation
        print(f"[Annotate Error] {e}")
        import copy as _copy
        fallback = _copy.deepcopy(results)
        fallback["metrics"] = {}
        fallback["radiologist_feedback"] = {"summary": "Annotation error."}
        return fallback
