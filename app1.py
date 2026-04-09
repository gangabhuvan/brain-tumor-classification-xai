#app1.py
import os, re, base64, sqlite3, traceback
from functools import wraps
from io import BytesIO
import cv2
from flask import (
    Flask, render_template, request, send_file, redirect,
    url_for, session, flash, g, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException  # <-- important
from PIL import Image
import tempfile
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import json
# Headless plotting
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image as PILImage
# Torch
import torch
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import unicodedata
# Local explainers
from utils import explainers1 as explainers
import gdown
torch.set_num_threads(os.cpu_count())
TEMP_RESULTS_CACHE = {}

# =============================================================================
# Flask setup
# =============================================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace-with-a-secure-key")

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---- Debug helpers (single canonical definitions) ----
LAST_ERROR = {"trace": None}

@app.errorhandler(Exception)
def on_exception(e):
    """Convert only unexpected exceptions to 500. Let HTTPException (404/405/…) pass through."""
    if isinstance(e, HTTPException):
        # Let Flask handle proper HTTP status codes like 404 instead of converting to 500
        return e
    trace = traceback.format_exc()
    LAST_ERROR["trace"] = trace
    app.logger.error("\n" + "="*80 + "\n💥 UNHANDLED EXCEPTION\n" + "="*80 + f"\n{trace}\n" + "="*80)
    if app.debug:
        return f"<pre>{trace}</pre>", 500
    return "Internal Server Error", 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True)

@app.route("/_last_error", methods=["GET"])
def last_error_endpoint():
    return jsonify(LAST_ERROR)

@app.route("/routes", methods=["GET"])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        methods = ",".join(sorted(m for m in rule.methods if m in {"GET", "POST"}))
        routes.append({"rule": str(rule), "endpoint": rule.endpoint, "methods": methods})
    routes.sort(key=lambda x: x["rule"])
    return jsonify(routes)

# Serve a favicon to stop 404→500 noise in logs
@app.route('/favicon.ico')
def favicon():
    fav_path = os.path.join(app.static_folder, 'favicon.ico')
    if os.path.exists(fav_path):
        return send_from_directory(app.static_folder, 'favicon.ico')
    # No favicon file? Return "no content" so browsers stop retrying
    return ('', 204)


# =============================================================================
# Auth (SQLite)
# =============================================================================
DB_PATH = "users.db"

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


def init_user_db():
    """Create users table + default reviewer account"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    # 🔥 Create default user (for reviewers / first run)
    from werkzeug.security import generate_password_hash

    c.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if not c.fetchone():
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash("admin123"))
        )

    conn.commit()
    conn.close()


# 🔥 ENSURE DB EXISTS BEFORE ANY REQUEST
@app.before_request
def ensure_db():
    init_user_db()


@app.teardown_appcontext
def close_connection(exc):
    db = getattr(g, "_database", None)
    if db:
        db.close()

def create_user(username, password):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def validate_user(username, password):
    conn = get_db()
    cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return bool(row and check_password_hash(row["password_hash"], password))

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if session.get("user") is None:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","")
        if not u or not p:
            flash("Username and password required.", "danger")
        elif create_user(u, p):
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash("Username already exists.", "danger")
    return render_template("register.html", user=session.get("user"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","")
        if validate_user(u, p):
            session["user"] = u
            flash(f"Welcome, {u}!", "login_welcome")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html", user=session.get("user"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "info")
    return redirect(url_for("login"))

@app.route("/")
def root():
    return redirect(url_for("login"))


# =============================================================================
# Model & transforms
# =============================================================================
SAFE_MODE = False  # set True to test UI without loading the model

device = torch.device("cpu")
num_classes = 4
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model = None
train_features_tensor = None

def get_model():
    global model
    if model is None:
        print("Lazy loading model...")
        _load_model()
    return model


def get_train_features():
    return None

def _load_model():
    global model
    if SAFE_MODE:
        model = None
        return

    base = convnext_tiny(weights=None)

    clf = base.classifier[2]
    if isinstance(clf, torch.nn.Sequential):
        lin = next((m for m in clf if isinstance(m, torch.nn.Linear)), None)
        in_features = lin.in_features if lin is not None else 768
    else:
        in_features = getattr(clf, "in_features", 768)

    base.classifier[2] = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2693893223768286),
        torch.nn.Linear(in_features, num_classes)
    )

    path = os.path.join("model", "final_convnext_aq.pth")
    if not os.path.exists(path):
        print("[INFO] Model not found locally. Downloading from Google Drive...")
        os.makedirs("model", exist_ok=True)
        file_id = "1NXQFVXNcGgVn28ZTBWIGq0Rt9r0Ov1-l"
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, path, quiet=False)
            print("[INFO] Model downloaded successfully.")
        except Exception as e:
            print("[ERROR] Model download failed:", e)
            raise RuntimeError("Model could not be downloaded. Check internet connection.")
    else:
        print("[INFO] Model already exists. Skipping download.")
    try:
        state = torch.load(path, map_location="cpu")
        base.load_state_dict(state)
    except Exception as e:
        print("[ERROR] Model loading failed:", e)
        raise RuntimeError("Failed to load model weights.")
    base = base.to(device)
    base.eval()
    base.float()
    model = base

def _load_train_features():
    return None

# =============================================================================
# Metrics from CSV (robust)
# =============================================================================
def compute_dataset_metrics(csv_path, cm_filename):
    labels = ["Glioma","Meningioma","No Tumor","Pituitary"]
    class_rows, avg_rows, accuracy = [], [], 0.0
    cm = np.zeros((4,4), dtype=int)

    try:
        df = pd.read_csv(csv_path)
        if not {"true_class","predicted_class"}.issubset(set(df.columns)):
            raise ValueError(f"{csv_path} missing required columns true_class/predicted_class")
        y_true = df["true_class"].astype(str).values
        y_pred = df["predicted_class"].astype(str).values

        from sklearn.metrics import classification_report, confusion_matrix
        rep = classification_report(y_true, y_pred, labels=labels,
                                    output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).transpose().reset_index().rename(columns={"index":"class"})
        class_rows = rep_df[~rep_df["class"].isin(["accuracy","macro avg","weighted avg"])] \
                        .to_dict(orient="records")
        avg_rows   = rep_df[ rep_df["class"].isin(["macro avg","weighted avg"]) ] \
                        .to_dict(orient="records")
        accuracy   = float(rep.get("accuracy", 0.0))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    except Exception as e:
        print(f"[WARN] metrics for {csv_path} failed:", e)

    cm_path = os.path.join(STATIC_FOLDER, cm_filename)
    try:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
        plt.savefig(cm_path, bbox_inches="tight"); plt.close()
    except Exception as e:
        print("[WARN] saving confusion matrix failed:", e)

    return class_rows, avg_rows, accuracy, cm_path

train_class_rows, train_avg_rows, train_accuracy_row, train_cm_path = compute_dataset_metrics(
    "results_train.csv", "confusion_matrix_train.png")
test_class_rows, test_avg_rows, test_accuracy_row, test_cm_path = compute_dataset_metrics(
    "results_test.csv", "confusion_matrix_test.png")


# =============================================================================
# Warning message helper
# =============================================================================
def _normalize_pct(v):
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    return f * 100.0 if f <= 1.05 else f

def warning_msg_and_type(confidence_frac, similarity_frac, true_class=None, predicted_class=None):
    """Generates a reliable high-level interpretability warning message for the Live Message & PDF."""
    conf = _normalize_pct(confidence_frac) or 100.0
    sim = _normalize_pct(similarity_frac) or 100.0

    # Thresholds
    LOW, HIGH, SIM_T = 90.0, 95.0, 95.0

    # --- Case 1: Misclassification always critical ---
    if true_class and predicted_class and str(true_class).lower() != str(predicted_class).lower():
        return (f"❌ Misclassification: Predicted '{predicted_class}', actual '{true_class}'. "
                f"Confidence {conf:.1f}% — verify manually."), "high_conf_misclass"

    # --- Case 2: Perfect or high-confidence correct predictions ---
    if conf >= 98.0 and sim >= 95.0:
        return "✅ Normal: Automated prediction completed safely.", "normal"

    # --- Case 3: High-confidence but slightly low similarity ---
    if conf >= HIGH and sim < SIM_T:
        return (f"⚠️ High confidence ({conf:.1f}%) but slightly low similarity ({sim:.1f}%) — "
                "verify contextually."), "review"

    # --- Case 4: Low similarity (external / dissimilar image) ---
    if sim < 80.0:
        return (f"❌ External/dissimilar image detected (similarity {sim:.1f}%). "
                "Manual verification required."), "critical"

    # --- Case 5: Moderate confidence range ---
    if conf < LOW:
        return (f"⚠️ Moderate confidence ({conf:.1f}%) — review advisable."), "moderate"

    # --- Default safe case ---
    return "✅ Normal: Automated prediction completed safely.", "normal"



# =============================================================================
# Dashboard (Final Robust Clinical Version)
# =============================================================================
# Replace your existing dashboard route with this function in app1.py
@app.route("/dashboard", methods=["GET","POST"])
@login_required
def dashboard():
    try:
        results = None
        results_aug = None
        warning_message = None
        warning_type = "normal"
        similarity_score = None
        top_conf_frac = 1.0
        conf_str = None
        radiologist_feedback = None
        metrics = None
        actual_label = None

        if request.method == "POST":
            file = request.files.get("file")
            if file and file.filename:
                filename = secure_filename(file.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                results = {}
                results["filename"] = filename
                session["last_uploaded_file"] = filename
                pil = Image.open(path).convert("RGB").resize((224,224))

                # Safe-mode fallback UI (unchanged)
                if SAFE_MODE:
                    dummy = Image.new("RGB", (512,512), (232,238,247))
                    b = BytesIO(); dummy.save(b, format="PNG"); b64 = base64.b64encode(b.getvalue()).decode("utf-8")
                    results = {
                        "prediction":"No Tumor",
                        "confidence":"100.0%",
                        "confidence_pct":1.0,
                        "second_prediction":"Glioma",
                        "second_confidence":"0.0%",
                        "original":b64,"gradcam":b64,"thresholded":b64,"lime":b64,"shap":b64
                    }
                    conf_str = "100.0%"
                    top_conf_frac = 1.0
                    similarity_score = 1.0
                else:
                    model = get_model()
                    import gc
                    gc.collect()
                    results = explainers.generate_explainability(
                        model, pil, device, class_names, transform=transform
                    )
                    gc.collect()
                    try:
                        os.remove(path)
                    except:
                        pass
                    # 2) Robust confidence parsing from results (support both confidence and confidence_pct)
                    conf_str = None
                    try:
                        if results.get("confidence"):
                            conf_str = str(results["confidence"]).strip()
                            top_conf_frac = float(conf_str.replace("%",""))/100.0
                        elif results.get("confidence_pct") is not None:
                            cp = float(results["confidence_pct"])
                            top_conf_frac = cp if 0.0 <= cp <= 1.05 else (cp/100.0)
                            conf_str = f"{top_conf_frac*100:.1f}%"
                        else:
                            top_conf_frac = 1.0
                            conf_str = "100.0%"
                    except Exception:
                        top_conf_frac = 1.0
                        conf_str = "100.0%"

                    # 3) CSV lookup for filename,true_class,predicted_class,confidence,similarity (prefer CSV when present)
                    csv_actual_label, csv_similarity, csv_pred, csv_conf = None, None, None, None
                    try:
                        for csv in ("results_train.csv","results_test.csv"):
                            if not os.path.exists(csv):
                                continue
                            df = pd.read_csv(csv)
                            # normalize filename search
                            if "filename" in df.columns:
                                m = df[df["filename"].astype(str).str.contains(filename, case=False, na=False)]
                            else:
                                m = pd.DataFrame()
                            if not m.empty:
                                row = m.iloc[0]
                                csv_actual_label = row.get("true_class")
                                csv_pred = row.get("predicted_class")
                                csv_conf = row.get("confidence", None) or row.get("confidence_pct", None)
                                s = row.get("similarity", None)
                                if s is not None:
                                    # extract numeric part robustly
                                    mobj = re.search(r"[-+]?\d*\.\d+|\d+", str(s))
                                    if mobj:
                                        sval = float(mobj.group(0))
                                        csv_similarity = sval/100.0 if sval > 1.05 else sval
                                break
                    except Exception as e:
                        print("[WARN] CSV lookup failed:", e)

                    # --- Propagate CSV-derived fields into `results` so annotator sees them ---
                    try:
                        print("[DEBUG] CSV lookup ->", {"csv_actual_label": csv_actual_label, "csv_pred": csv_pred, "csv_conf": csv_conf, "csv_similarity": csv_similarity})

                        if csv_actual_label:
                            # attach canonical and alias keys expected by explainers/radiologist feedback
                            results["true_class"] = str(csv_actual_label)
                            results["true_class_csv"] = str(csv_actual_label)

                        if csv_pred:
                            results["predicted_class_from_csv"] = str(csv_pred)

                        if csv_conf is not None:
                            try:
                                cf = float(str(csv_conf).strip().replace("%",""))
                                if cf > 1.05:
                                    cf = cf / 100.0
                                # store both normalized and display forms (do not clobber if present)
                                if not results.get("confidence_pct"):
                                    results["confidence_pct"] = cf
                                if not results.get("confidence"):
                                    results["confidence"] = f"{cf*100:.1f}%"
                                print("[DEBUG] propagated confidence:", results.get("confidence_pct"), results.get("confidence"))
                            except Exception as _e:
                                print("[DEBUG] confidence parse failed:", _e)

                        if csv_similarity is not None and not results.get("similarity"):
                            try:
                                sval = float(csv_similarity)
                                if sval > 1.05:
                                    sval = sval / 100.0
                                results["similarity"] = sval
                                print("[DEBUG] propagated similarity:", results.get("similarity"))
                            except Exception as _e:
                                print("[DEBUG] similarity parse failed:", _e)
                    except Exception as _e:
                        print("[WARN] failed to attach CSV metadata to results:", _e)

                    # 4) Feature-based similarity fallback if CSV similarity missing
                    sim_warn, feat_sim = None, None
                    similarity_score = csv_similarity
                    if similarity_score is None:
                        similarity_score = 1.0

                    # 5) Annotate live metrics & radiologist feedback (this is your annotated pipeline)
                    try:
                        results_aug = explainers.annotate_results_with_live_metrics(
                            results, pil, model, device, train_features_tensor, similarity_score=similarity_score
                            )

                    except Exception as e:
                        print("[WARN] annotate_results_with_live_metrics failed:", e)
                        results_aug = results.copy()
                        results_aug["metrics"] = {}
                        results_aug["radiologist_feedback"] = {"summary": "Annotation error."}

                    # ensure results_aug has confidence_pct and strings in standard form
                    try:
                        if "confidence_pct" not in results_aug and results.get("confidence_pct") is not None:
                            results_aug["confidence_pct"] = results["confidence_pct"]
                        if "confidence" not in results_aug and results.get("confidence"):
                            results_aug["confidence"] = results["confidence"]
                    except Exception:
                        pass

                    # 6) Reconcile CSV-derived signals with annotator feedback to decide the final tone and live message.
                    # Compute baseline warning via your existing helper
                    msg, wtype = warning_msg_and_type(top_conf_frac, similarity_score, csv_actual_label, results.get("prediction"))

                    # get annotator feedback + metrics
                    annot_fb = results_aug.get("radiologist_feedback", {})
                    metrics = results_aug.get("metrics", {})

                    # If CSV indicates misclassification -> force critical
                    forced_tone = None
                    forced_message_prefix = None
                    if csv_actual_label and results.get("prediction") and str(csv_actual_label).strip().lower() != str(results.get("prediction")).strip().lower():
                        forced_tone = "critical"
                        forced_message_prefix = f"❌ Misclassification: Predicted '{results.get('prediction')}', actual '{csv_actual_label}'."
                    elif similarity_score is not None and similarity_score < 0.80:
                        # low similarity -> external/unreliable
                        forced_tone = "critical"
                        forced_message_prefix = f"❌ External/dissimilar image detected (similarity {similarity_score*100:.1f}%)."
                    elif sim_warn and "External" in (sim_warn or ""):
                        forced_tone = "critical"
                        forced_message_prefix = sim_warn

                    # Compose final radiologist_feedback using annotator feedback as base but override tone/summary if forced
                    final_fb = {
                        "summary": annot_fb.get("summary", "No annotator summary."),
                        "confidence_advice": annot_fb.get("confidence_advice", ""),
                        "localization_advice": annot_fb.get("localization_advice", ""),
                        "consistency_advice": annot_fb.get("consistency_advice", ""),
                        "next_steps": annot_fb.get("next_steps", ""),
                        "clinical_impression": annot_fb.get("clinical_impression", ""),
                        "tone": None
                    }

                    # If forced, prepend forced message and set critical tone
                    if forced_tone == "critical":
                        # prefix the summary and bump tone
                        prefix = forced_message_prefix + " "
                        final_fb["summary"] = prefix + final_fb["summary"]
                        final_fb["tone"] = "critical"
                        # Ensure localization and consistency advice are not overwritten for critical cases
                        final_fb["localization_advice"] = annot_fb.get("localization_advice", "")
                        final_fb["consistency_advice"] = annot_fb.get("consistency_advice", "")

                        # if CSV misclassification, update confidence advice to highlight mislabel
                        if "Misclassification" in (forced_message_prefix or "") and not final_fb.get("confidence_advice"):
                            final_fb["confidence_advice"] = (
                                f"❌❌ Misclassification: Predicted '{results.get('prediction')}', actual '{csv_actual_label}'. "
                                "Manual verification required."
                                )




                    else:
                        # Map annotator grade -> tone if no forced condition
                        grade = metrics.get("Explainability_Grade", None)
                        if grade in ["A+", "A"]:
                            final_fb["tone"] = "positive"
                        elif grade in ["B", "C"]:
                            final_fb["tone"] = "review"
                        elif grade in ["D", "E"]:
                            final_fb["tone"] = "critical"
                        else:
                            # fallback to earlier warning type mapping
                            if wtype in ["high_conf_misclass","unreliable","external","explainer"]:
                                final_fb["tone"] = "critical"
                            elif wtype == "low_conf":
                                final_fb["tone"] = "caution"
                            else:
                                final_fb["tone"] = "normal"

                    # Final live warning message - keep consistent with radiologist feedback
                    # If forced critical, use forced message; otherwise, prefer explainer's warning if present, then msg.

                    if final_fb["tone"] == "critical":
                        tone_tag = "critical"
                        if forced_message_prefix:
                            warning_message = forced_message_prefix + " Verify manually."
                            warning_type = "critical"
                        else:
                            warning_message = msg
                            warning_type = wtype
                    elif final_fb["tone"] == "caution":
                        warning_message = msg
                        warning_type = "low_conf"
                    elif final_fb["tone"] == "review":
                        try:
                            conf_val = float(top_conf_frac) * 100.0
                            sim_val = float(similarity_score or 1.0) * 100.0
                        except Exception:
                            conf_val, sim_val = 100.0, 100.0
                            # High-confidence, correct => normal
                        if conf_val >= 95.0 and (csv_actual_label is None or str(csv_actual_label).lower() == str(results.get("prediction")).lower()):
                            warning_message = "✅ Normal: Automated prediction completed safely."
                            warning_type = "normal"
                        else:
                            warning_message = "⚠️ Moderate interpretability — manual review advised."
                            warning_type = "moderate"
                    else:
                        warning_message = "✅ Normal: Automated prediction completed safely."
                        warning_type = "normal"


                    # --- after you've computed final_fb, warning_message, metrics etc. ---
                    #Attach the final feedback and metrics for template
                    radiologist_feedback = final_fb
                    metrics = metrics or results_aug.get("metrics", {})
                    # Ensure results_aug is updated with final feedback and warning before caching
                    results_aug["radiologist_feedback"] = final_fb
                    # store human-readable warning_text (prefer forced prefix if present)
                    results_aug["warning"] = (forced_message_prefix + " Verify manually.") if forced_message_prefix else (msg or "")
                    # Also keep normalized fields (confidence strings) in cached results
                    if "confidence_pct" not in results_aug and results.get("confidence_pct") is not None:
                        results_aug["confidence_pct"] = results["confidence_pct"]
                    if "confidence" not in results_aug and results.get("confidence"):
                        results_aug["confidence"] = results["confidence"]
                    # Cache the final augmented result (server-side cache to avoid large session cookies)
                    cache_id = str(uuid.uuid4())
                    TEMP_RESULTS_CACHE[cache_id] = results_aug
                    session["last_results_id"] = cache_id
                    session.modified = True

        # Render template (unchanged fields)
        return render_template(
            "index1.html",
            title="Brain Tumor MRI Classification",
            user=session.get("user"),
            # result values
            results=results,
            prediction=(results or {}).get("prediction"),
            confidence=(conf_str or (results or {}).get("confidence") or ""),
            second_prediction=(results or {}).get("second_prediction"),
            second_confidence=(results or {}).get("second_confidence"),
            similarity_score=similarity_score,
            warning=warning_message,
            warning_type=warning_type,
            # images
            original=(results or {}).get("original"),
            gradcam=(results or {}).get("gradcam"),
            thresholded=(results or {}).get("thresholded"),
            lime=(results or {}).get("lime"),
            shap=(results or {}).get("shap"),
            # metrics & radiologist feedback for side panels / feedback card
            radiologist_feedback=radiologist_feedback,
            metrics=metrics,
            train_class_rows=train_class_rows,
            train_avg_rows=train_avg_rows,
            train_accuracy_row=train_accuracy_row,
            train_cm_path=train_cm_path,
            test_class_rows=test_class_rows,
            test_avg_rows=test_avg_rows,
            test_accuracy_row=test_accuracy_row,
            test_cm_path=test_cm_path,
        )
    except Exception:
        LAST_ERROR["trace"] = traceback.format_exc()
        if app.debug:
            return f"<pre>{LAST_ERROR['trace']}</pre>", 500
        return "Internal Server Error", 500


# ================================================================
#  SAFE SANITIZER
# ================================================================
def sanitize_text(text: str) -> str:
    """Fully safe text sanitizer for FPDF (handles all Unicode and encoding edge cases)."""
    if not text:
        return "N/A"
    try:
        text = unicodedata.normalize("NFKC", str(text))
        replacements = {
            "✅": "[OK]", "❌": "[X]", "⚠️": "[!]", "🧠": "[Brain]",
            "📍": "[Loc]", "🩺": "[Clinical]", "🔴": "[Critical]",
            "🟢": "[Good]", "🟡": "[Review]", "🔁": "[Repeat]", "ℹ️": "[Info]",
            "•": "-", "●": "-", "▪": "-", "|": " | ", "\u00A0": " ",
            "→": "->", "≥": ">=", "≤": "<=", "↔": "<->"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r"[\u200B-\u200D\uFEFF\u2028\u2029\u00AD]", "", text)
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = []
        for w in text.split():
            if len(w) > 30:
                for i in range(0, len(w), 30):
                    words.append(w[i:i + 30] + "-")
            else:
                words.append(w)
        safe = " ".join(words)
        return safe.encode("latin-1", "replace").decode("latin-1")
    except Exception as e:
        print(f"[Sanitize Error] {e}")
        return str(text).encode("latin-1", "replace").decode("latin-1")


# ================================================================
#  SAFE MULTICELL WRAPPER
# ================================================================
def safe_multicell(pdf, width, height, text, **kwargs):
    """A wrapper that guarantees safe multicell rendering for any text length."""
    try:
        if not width or width <= 0:
            width = pdf.w - pdf.l_margin - pdf.r_margin
        cleaned = sanitize_text(text or "")
        pdf.multi_cell(width, height, cleaned, new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kwargs)
    except Exception as e:
        print(f"[safe_multicell warn] {e}")
        try:
            truncated = sanitize_text(str(text)[:400]) + " [...]"
            pdf.multi_cell(width, height, truncated, new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kwargs)
        except Exception as e2:
            print(f"[safe_multicell fallback] {e2}")
            pdf.cell(width, height, "Text render error", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def get_local_time(tz_name='Asia/Kolkata'):
    from datetime import datetime
    from pytz import timezone
    return datetime.now(timezone(tz_name)).strftime('%Y-%m-%d %H:%M:%S')

# ================================================================
#  CLINICAL REPORT PDF CLASS (Optimized Layout)
# ================================================================
class ClinicalReportPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_page_rendered = False

    def header(self):
        """Custom header with conditional title and logo placement."""
        self.draw_page_border()

        if not self._first_page_rendered:
            try:
                logo_path = os.path.join(app.root_path, "static", "nmit_logo.jpeg")
                if os.path.exists(logo_path):
                    # perfectly aligned: slightly higher and right to title
                    self.image(logo_path, x=self.w - 36, y=7, w=16)
            except Exception:
                pass

            self.set_font("Helvetica", "B", 15)
            self.cell(0, 10, "Brain Tumor MRI Explainability Report",
                      align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            self.set_draw_color(180, 180, 180)
            self.set_line_width(0.2)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(2)

            self._first_page_rendered = True
        else:
            self.set_y(self.t_margin + 4)

    def footer(self):
        """Neat footer with page number."""
        self.set_y(-12)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")

    def draw_page_border(self):
        """Draw subtle light-gray border inside page edges."""
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.4)
        margin = 6
        self.rect(margin, margin, self.w - 2 * margin, self.h - 2 * margin)

    def draw_section_line(self, y=None):
        """Draw subtle gray line (used under image titles)."""
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        if y is None:
            y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)

def find_image_metadata(filename: str):
    """
    Looks for an image in internal train/test CSVs and returns its ground truth, predicted class, and source.
    """
    try:
        import pandas as pd, os
        if not filename:
            return None, None, "External upload"

        base = os.path.splitext(os.path.basename(filename))[0].lower()

        for csv_path, source in [("results_train.csv", "Train Dataset"), ("results_test.csv", "Test Dataset")]:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["fname_clean"] = df["filename"].astype(str).str.lower().str.extract(r"([^/\\]+)\.[^.]+$")[0]
                match = df[df["fname_clean"].str.contains(base, na=False)]
                if not match.empty:
                    row = match.iloc[0]
                    true_class = row.get("true_class") or row.get("actual") or "Unknown"
                    pred_class = row.get("predicted_class") or row.get("prediction") or "Unknown"
                    return str(true_class), str(pred_class), source

        return None, None, "External upload"

    except Exception as e:
        print(f"[Metadata lookup error] {e}")
        return None, None, "External upload"
    

# ================================================================
#  FINAL DOWNLOAD_PDF ROUTE
# ================================================================
@app.route("/download_pdf", methods=["GET", "POST"])
@login_required
def download_pdf():
    """Final production-grade clinical PDF generator."""
    try:
        results = {}
        cache_id = session.get("last_results_id")
        if cache_id and cache_id in TEMP_RESULTS_CACHE:
            results = TEMP_RESULTS_CACHE[cache_id]

        if not results and request.method == "POST":
            metrics_obj = json.loads(request.form.get("metrics")) if request.form.get("metrics") else {}
            fb_obj = json.loads(request.form.get("radiologist_feedback")) if request.form.get("radiologist_feedback") else {}
            results = {
                "prediction": request.form.get("prediction"),
                "confidence": request.form.get("confidence"),
                "confidence_pct": request.form.get("confidence_pct"),
                "original": request.form.get("original"),
                "gradcam": request.form.get("gradcam"),
                "thresholded": request.form.get("thresholded"),
                "lime": request.form.get("lime"),
                "shap": request.form.get("shap"),
                "metrics": metrics_obj,
                "radiologist_feedback": fb_obj,
                "warning": request.form.get("warning", "None"),
            }

        if not results:
            return "No recent results available.", 404

        metrics = results.get("metrics", {}) or {}
        feedback = results.get("radiologist_feedback", {}) or {}

        # --- ensure latest live warning message is reflected ---
        warning_text = sanitize_text(
            results.get("warning") or session.get("last_warning") or "None"
        )

        user = session.get("user", "Radiologist")

        # === Initialize PDF ===
        pdf = ClinicalReportPDF(orientation="P", unit="mm", format="A4")
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "", 11)

        # ---- Header Meta ----
        safe_multicell(pdf, 0, 6, f"Generated for: {user}")
        safe_multicell(pdf, 0, 6, "Institution: NMIT-ISE-FYP-2025-G10")
        safe_multicell(pdf, 0, 6, f"Date & Time: {get_local_time()}")
        safe_multicell(pdf, 0, 6, f"Warning: {warning_text}")
        pdf.ln(4)

        # ---- Feedback Summary ----
        pdf.set_font("Helvetica", "B", 13)
        safe_multicell(pdf, 0, 8, "Radiologist Feedback Summary")
        pdf.set_font("Helvetica", "", 11)

        predicted_class = results.get("prediction", "N/A")
        # --- Try to find actual source from internal results CSVs ---
        uploaded_filename = results.get("filename") or results.get("image_path") or session.get("last_uploaded_file")
        true_class, csv_pred_class, img_source = find_image_metadata(uploaded_filename or predicted_class)

        # --- Use true_class only if found ---
        if true_class and true_class.lower() not in ["none", "unknown", ""]:
            safe_multicell(pdf, 0, 6, f"Prediction Results: Predicted — {predicted_class} | True — {true_class}")
            source_info = f"{img_source} (ground truth verified)"
        else:
            safe_multicell(pdf, 0, 6, f"Prediction Results: Predicted — {predicted_class} (ground truth unavailable)")
            source_info = "External upload"

        pdf.set_font("Helvetica", "I", 9)
        safe_multicell(pdf, 0, 5, f"Image Source: {source_info}")
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(3)


        safe_multicell(pdf, 0, 6, f"Summary: {feedback.get('summary', 'N/A')}")
        clinical_imp = feedback.get("clinical_impression")
        if clinical_imp:
            pdf.set_font("Helvetica", "I", 11)
            safe_multicell(pdf, 0, 6, f"Clinical Impression: {clinical_imp}")
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(2)

        for label, key in [
            ("Confidence", "confidence_advice"),
            ("Localization", "localization_advice"),
            ("Consistency", "consistency_advice"),
            ("Next Steps", "next_steps"),
        ]:
            val = feedback.get(key)
            if val:
                safe_multicell(pdf, 0, 6, f"{label}: {val}")
                pdf.ln(1)
        pdf.ln(4)

        # ---- Metrics Summary ----
        pdf.set_font("Helvetica", "B", 12)
        safe_multicell(pdf, 0, 8, "Explainability Metrics (Key Values)")
        pdf.set_font("Helvetica", "", 11)
        if metrics:
            keys = ["localization_acc", "dice_similarity", "map_correlation", "focus_ratio", "TEAS", "robustness"]
            parts = []
            for k in keys:
                v = metrics.get(k)
                try:
                    parts.append(f"{k.split('_')[0].capitalize()}: {float(v):.2f}")
                except Exception:
                    if v:
                        parts.append(f"{k.split('_')[0].capitalize()}: {sanitize_text(str(v))}")
            grade = metrics.get("Explainability_Grade", "N/A")
            safe_multicell(pdf, 0, 6, " | ".join(parts) + f" | Grade: {grade}")
        else:
            safe_multicell(pdf, 0, 6, "Metrics unavailable.")
        pdf.ln(4)

        # ---- Metric Interpretation Table ----
        pdf.set_font("Helvetica", "B", 12)
        safe_multicell(pdf, 0, 8, "Explainability Metric Scale & Interpretation")
        pdf.ln(2)
        widths = [48, 30, 60, 52]
        headers = ["Metric", "Range / Scale", "Evaluates", "Interpretation Guidelines"]
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(242, 242, 242)
        for i, h in enumerate(headers):
            pdf.cell(widths[i], 8, sanitize_text(h), border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        rows = [
            ("IoU (localization_acc)", "0.0 to 1.0", "Overlap of heatmap with ROI", "≥0.6 good; 0.4 to 0.6 partial; <0.4 poor"),
            ("Dice similarity", "0.0 to 1.0", "Overlap consistency", "≥0.55 good; 0.35 to 0.55 moderate; <0.35 poor"),
            ("TEAS", "0.0 to 1.0", "Agreement across explainers", "≥0.45 good; 0.25 to 0.45 moderate; <0.25 low"),
            ("Map Corr", "-1.0 to 1.0", "Correlation between maps", "≥0.3 reasonable; lower = disagreement"),
            ("Focus ratio", "0.0 to 1.0", "Concentration of activation in ROI", "≥0.6 preferred"),
            ("Robustness", "0.0 to 1.0", "Confidence+localization composite", "≥0.7 robust"),
        ]
        for row in rows:
            y0 = pdf.get_y()
            heights = []
            for i, col in enumerate(row):
                lines = pdf.multi_cell(widths[i], 5, sanitize_text(col), border=0, align="L", dry_run=True, output="LINES")
                heights.append(len(lines) * 5)
            max_h = max(heights)
            pdf.set_xy(pdf.l_margin, y0)
            for i, col in enumerate(row):
                safe_multicell(pdf, widths[i], 5, col, border=1, align="L")
                pdf.set_xy(pdf.l_margin + sum(widths[:i + 1]), y0)
            pdf.set_y(y0 + max_h)
        pdf.ln(5)

        # ---- Explainability Scoring Info ----
        pdf.set_font("Helvetica", "B", 12)
        safe_multicell(pdf, 0, 8, "Explainability T-Score and Grade Calculation")
        pdf.set_font("Helvetica", "", 10)
        safe_multicell(pdf, 0, 5,
               "The final explainability grade (A to E) is computed using weighted IoU, Dice, Focus, TEAS, and Robustness. "
               "Higher agreement yields higher grades:")

        safe_multicell(pdf, 0, 5, "A+ ≥ 0.75 → Exceptional explainability and focus alignment")
        safe_multicell(pdf, 0, 5, "A = 0.65 to 0.74 → Very good interpretability; reliable localization")
        safe_multicell(pdf, 0, 5, "B = 0.55 to 0.64 → Good/Acceptable interpretability")
        safe_multicell(pdf, 0, 5, "C = 0.50 to 0.54 → Moderate interpretability; limited focus overlap")
        safe_multicell(pdf, 0, 5, "D = 0.35 to 0.49 → Weak localization; manual review required")
        safe_multicell(pdf, 0, 5, "E < 0.35 → Poor alignment; not suitable for autonomous interpretation.")
        pdf.ln(5)

        # ---- Image Pages ----
        image_keys = [
            ("Original MRI Image", results.get("original")),
            ("Grad-CAM++ (Raw)", results.get("gradcam")),
            ("Grad-CAM++ (Thresholded)", results.get("thresholded")),
            ("LIME Explanation", results.get("lime")),
            ("SHAP Explanation", results.get("shap")),
        ]
        tmp_files = []
        present_images = [(t, b) for (t, b) in image_keys if b]

        for idx, (title, b64) in enumerate(present_images):
            try:
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmpf.write(img_bytes)
                tmpf.close()
                tmp_files.append(tmpf.name)

                pdf.add_page()
                pdf.set_font("Helvetica", "B", 12)
                safe_multicell(pdf, 0, 8, title, align="C")
                pdf.draw_section_line()
                pdf.ln(4)

                pdf.set_y(35)
                with PILImage.open(tmpf.name) as im:
                    w_px, h_px = im.size
                    max_w, max_h = pdf.w - 30, pdf.h - 60
                    aspect = w_px / float(h_px)
                    w_pdf = max_w
                    h_pdf = w_pdf / aspect
                    if h_pdf > max_h:
                        h_pdf = max_h
                        w_pdf = h_pdf * aspect

                pdf.image(tmpf.name, x=(pdf.w - w_pdf) / 2, y=pdf.get_y(), w=w_pdf, h=h_pdf)
                pdf.ln(h_pdf + 6)

                if idx == len(present_images) - 1:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(120, 120, 120)
                    safe_multicell(pdf, 0, 6,
                                   "Disclaimer: This explainability report is intended for decision support only. "
                                   "Final interpretation must be verified by a qualified radiologist.")
                    pdf.set_text_color(0, 0, 0)
            except Exception as e:
                app.logger.warning(f"[PDF] Image embed failed for '{title}': {e}")

        pdf_bytes = pdf.output()  # FPDF2 now returns bytes directly
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            pdf_bytes = str(pdf_bytes).encode("latin-1", "replace")


        for f in tmp_files:
            try:
                os.remove(f)
            except Exception:
                pass
        if cache_id:
            TEMP_RESULTS_CACHE.pop(cache_id, None)

        return send_file(BytesIO(pdf_bytes),
                         mimetype="application/pdf",
                         as_attachment=True,
                         download_name=f"Explainability_Report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf")

    except Exception as e:
        app.logger.error(f"[PDF Error] {e}\n{traceback.format_exc()}")
        return f"<pre>{traceback.format_exc()}</pre>", 500



# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
