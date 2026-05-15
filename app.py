"""
=============================================================================
  CardioScan AI — Combined Heart Disease Detection
  Features: Numerical (XGBoost) + ECG (LightResNet1D) + Combined Score
  Run: python app.py  →  http://localhost:5000
=============================================================================
"""

import os, io, tempfile, traceback, base64
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
from scipy.signal import butter, filtfilt
from xgboost import XGBClassifier
import wfdb

app = Flask(__name__, template_folder='.')
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=15, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            ) if stride != 1 or in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class LightResNet1D(nn.Module):
    def __init__(self, num_leads=12, num_classes=1, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 32, 15, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
        self.layer1 = ResBlock1D(32,  64,  stride=2, dropout=dropout)
        self.layer2 = ResBlock1D(64,  128, stride=2, dropout=dropout)
        self.layer3 = ResBlock1D(128, 256, stride=2, dropout=dropout)
        self.layer4 = ResBlock1D(256, 256, stride=2, dropout=dropout)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(self.gap(x))

    def get_last_conv_features(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.layer4(x)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

XGB_MODEL = None
ECG_MODEL = None

def load_models():
    global XGB_MODEL, ECG_MODEL

    try:
        m = XGBClassifier()
        m.load_model("xgboost_heart_model.json")
        XGB_MODEL = m
        print("✓ Clinical model loaded")
    except Exception as e:
        print(f"⚠️  Clinical model not loaded: {e}")

    try:
        ckpt = torch.load("best_ecg_model.pt", map_location="cpu")
        m    = LightResNet1D()
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        ECG_MODEL = m
        print(f"✓ ECG model loaded — val AUC {ckpt['val_auc']:.4f}")
    except Exception as e:
        print(f"⚠️  ECG model not loaded: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ecg(signal, fs=100):
    nyq = fs / 2.0
    b, a = butter(2, 0.5/nyq, btype="high")
    signal = filtfilt(b, a, signal, axis=0).astype(np.float32)
    b, a = butter(4, [0.5/nyq, 40.0/nyq], btype="band")
    signal = filtfilt(b, a, signal, axis=0).astype(np.float32)
    mean = signal.mean(axis=0, keepdims=True)
    std  = signal.std(axis=0, keepdims=True) + 1e-8
    signal = (signal - mean) / std
    T = signal.shape[0]
    if T >= 1000: signal = signal[:1000]
    else:
        signal = np.vstack([signal, np.zeros((1000-T, 12), dtype=np.float32)])
    return torch.tensor(signal.T[np.newaxis], dtype=torch.float32)


LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def ecg_to_base64(x_tensor, prob, threshold=0.3):
    sig  = x_tensor.squeeze().numpy()
    time = np.arange(sig.shape[1]) / 100
    is_mi = prob >= threshold
    color = "#ef4444" if is_mi else "#22c55e"

    fig, axes = plt.subplots(6, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#060d1a")
    for i, (ax, name) in enumerate(zip(axes.flatten(), LEAD_NAMES)):
        ax.set_facecolor("#0d1f35")
        ax.plot(time, sig[i], color=color, linewidth=0.8)
        ax.set_ylabel(name, fontsize=9, color="#7ab3d4",
                      rotation=0, labelpad=22, va="center")
        ax.tick_params(colors="#475569", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.grid(True, alpha=0.15, color="#475569"); ax.set_yticks([])
    for ax in axes.flatten()[-2:]:
        ax.set_xlabel("Time (s)", fontsize=8, color="#7ab3d4")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#060d1a")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def get_risk_factors(data):
    factors = []
    if float(data.get('BMI', 0)) > 30:        factors.append(f"High BMI ({data['BMI']})")
    if int(data.get('Smoking', 0)):            factors.append("Smoking")
    if int(data.get('Stroke', 0)):             factors.append("Prior stroke")
    if int(data.get('DiffWalking', 0)):         factors.append("Difficulty walking")
    if int(data.get('Diabetic', 0)) > 0:        factors.append("Diabetes")
    if int(data.get('KidneyDisease', 0)):       factors.append("Kidney disease")
    if float(data.get('SleepTime', 8)) < 6:    factors.append("Insufficient sleep")
    if not int(data.get('PhysicalActivity', 1)):factors.append("Inactive lifestyle")
    if int(data.get('PhysicalHealth', 0)) > 15:factors.append("Poor physical health")
    return factors


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           xgb_ready=(XGB_MODEL is not None),
                           ecg_ready=(ECG_MODEL is not None))


@app.route("/predict_numerical", methods=["POST"])
def predict_numerical():
    if XGB_MODEL is None:
        return jsonify({"error": "Clinical model not loaded. Check xgboost_heart_model.json exists."}), 500
    try:
        data = {
            'BMI'             : request.form['BMI'],
            'Smoking'         : request.form['Smoking'],
            'Stroke'          : request.form['Stroke'],
            'PhysicalHealth'  : request.form['PhysicalHealth'],
            'DiffWalking'     : request.form['DiffWalking'],
            'Sex'             : request.form['Sex'],
            'AgeCategory'     : request.form['AgeCategory'],
            'Diabetic'        : request.form['Diabetic'],
            'PhysicalActivity': request.form['PhysicalActivity'],
            'SleepTime'       : request.form['SleepTime'],
            'KidneyDisease'   : request.form['KidneyDisease'],
        }
        df_input = pd.DataFrame({k: [float(v)] for k, v in data.items()})
        risk     = float(XGB_MODEL.predict_proba(df_input)[0][1])
        pred     = 1 if risk >= 0.3 else 0
        factors  = get_risk_factors(data)

        return jsonify({
            "prediction"     : pred,
            "probability"    : round(risk, 4),
            "probability_pct": round(risk * 100, 1),
            "risk_factors"   : factors,
            "label"          : "High Risk" if pred == 1 else "Low Risk",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_ecg", methods=["POST"])
def predict_ecg():
    if ECG_MODEL is None:
        return jsonify({"error": "ECG model not loaded. Check best_ecg_model.pt exists."}), 500
    threshold = float(request.form.get("threshold", 0.3))
    try:
        if "hea_file" in request.files and request.files["hea_file"].filename:
            hea_file = request.files["hea_file"]
            dat_file = request.files.get("dat_file")
            with tempfile.TemporaryDirectory() as tmpdir:
                # Keep original filename so wfdb can find matching .dat
                hea_filename = hea_file.filename
                record_name  = hea_filename.replace(".hea", "")
                hea_file.save(os.path.join(tmpdir, hea_filename))
                if dat_file and dat_file.filename:
                    dat_file.save(os.path.join(tmpdir, dat_file.filename))
                signal, fields = wfdb.rdsamp(os.path.join(tmpdir, record_name))
            x_tensor = preprocess_ecg(signal, fs=fields["fs"])

        elif "npy_file" in request.files and request.files["npy_file"].filename:
            signal = np.load(io.BytesIO(request.files["npy_file"].read()))
            if signal.ndim == 2 and signal.shape[0] == 12:
                signal = signal.T
            x_tensor = preprocess_ecg(signal.astype(np.float32))
        else:
            return jsonify({"error": "No ECG file uploaded."}), 400

        with torch.no_grad():
            prob = torch.sigmoid(ECG_MODEL(x_tensor).squeeze()).item()

        is_mi = prob >= threshold
        confidence = (
            "Very High" if abs(prob-0.5) > 0.4 else
            "High"      if abs(prob-0.5) > 0.3 else
            "Medium"    if abs(prob-0.5) > 0.15 else "Low"
        )
        ecg_img = ecg_to_base64(x_tensor, prob, threshold)

        return jsonify({
            "probability"    : round(prob, 4),
            "probability_pct": round(prob * 100, 1),
            "prediction"     : "MI" if is_mi else "NORM",
            "label"          : "Myocardial Infarction Detected" if is_mi else "Normal ECG",
            "confidence"     : confidence,
            "is_mi"          : is_mi,
            "ecg_plot"       : ecg_img,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_combined", methods=["POST"])
def predict_combined():
    """
    Runs both models and fuses predictions into one combined risk score.
    Weights: 40% clinical + 60% ECG (ECG is more direct cardiac evidence).
    """
    if XGB_MODEL is None or ECG_MODEL is None:
        return jsonify({"error": "Both models must be loaded for combined analysis."}), 500

    threshold = float(request.form.get("threshold", 0.3))

    # ── Run clinical model ──────────────────────────────────────────────
    try:
        data = {
            'BMI'             : request.form['BMI'],
            'Smoking'         : request.form['Smoking'],
            'Stroke'          : request.form['Stroke'],
            'PhysicalHealth'  : request.form['PhysicalHealth'],
            'DiffWalking'     : request.form['DiffWalking'],
            'Sex'             : request.form['Sex'],
            'AgeCategory'     : request.form['AgeCategory'],
            'Diabetic'        : request.form['Diabetic'],
            'PhysicalActivity': request.form['PhysicalActivity'],
            'SleepTime'       : request.form['SleepTime'],
            'KidneyDisease'   : request.form['KidneyDisease'],
        }
        df_input    = pd.DataFrame({k: [float(v)] for k, v in data.items()})
        num_risk    = float(XGB_MODEL.predict_proba(df_input)[0][1])
        risk_factors = get_risk_factors(data)
    except Exception as e:
        return jsonify({"error": f"Clinical model error: {str(e)}"}), 500

    # ── Run ECG model ───────────────────────────────────────────────────
    try:
        if "hea_file" in request.files and request.files["hea_file"].filename:
            hea_file = request.files["hea_file"]
            dat_file = request.files.get("dat_file")
            with tempfile.TemporaryDirectory() as tmpdir:
                hea_filename = hea_file.filename
                record_name  = hea_filename.replace(".hea", "")
                hea_file.save(os.path.join(tmpdir, hea_filename))
                if dat_file and dat_file.filename:
                    dat_file.save(os.path.join(tmpdir, dat_file.filename))
                signal, fields = wfdb.rdsamp(os.path.join(tmpdir, record_name))
            x_tensor = preprocess_ecg(signal, fs=fields["fs"])
        elif "npy_file" in request.files and request.files["npy_file"].filename:
            signal = np.load(io.BytesIO(request.files["npy_file"].read()))
            if signal.ndim == 2 and signal.shape[0] == 12:
                signal = signal.T
            x_tensor = preprocess_ecg(signal.astype(np.float32))
        else:
            return jsonify({"error": "No ECG file uploaded for combined analysis."}), 400

        with torch.no_grad():
            ecg_prob = torch.sigmoid(ECG_MODEL(x_tensor).squeeze()).item()

        ecg_img = ecg_to_base64(x_tensor, ecg_prob, threshold)
        ecg_label = "Myocardial Infarction Detected" if ecg_prob >= threshold else "Normal ECG"
        ecg_confidence = (
            "Very High" if abs(ecg_prob-0.5) > 0.4 else
            "High"      if abs(ecg_prob-0.5) > 0.3 else
            "Medium"    if abs(ecg_prob-0.5) > 0.15 else "Low"
        )
    except Exception as e:
        return jsonify({"error": f"ECG model error: {str(e)}"}), 500

    # ── Fuse predictions ────────────────────────────────────────────────
    # Weighted average: 40% clinical, 60% ECG
    combined_prob = 0.4 * num_risk + 0.6 * ecg_prob
    is_high       = combined_prob >= threshold

    return jsonify({
        "combined_prob"    : round(combined_prob * 100, 1),
        "clinical_prob"    : round(num_risk * 100, 1),
        "ecg_prob"         : round(ecg_prob * 100, 1),
        "prediction"       : 1 if is_high else 0,
        "is_high"          : is_high,
        "risk_factors"     : risk_factors,
        "ecg_plot"         : ecg_img,
        "ecg_label"        : ecg_label,
        "ecg_confidence"   : ecg_confidence,
        "fusion_note"      : "Combined score = 40% Clinical + 60% ECG analysis",
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  CardioScan AI — Heart Attack Detection")
    print("=" * 55)
    load_models()
    print("\n→ http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)