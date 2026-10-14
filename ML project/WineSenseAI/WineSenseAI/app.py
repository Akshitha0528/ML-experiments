"""
WineSense AI — Backend API

Serves the trained Decision Tree / Random Forest models (see train.py) over a
small REST API, matching the interface specified in the project brief:

    POST /predict
    {
      "fixed_acidity": number, "volatile_acidity": number, "citric_acid": number,
      "residual_sugar": number, "chlorides": number, "free_sulfur_dioxide": number,
      "total_sulfur_dioxide": number, "density": number, "ph": number,
      "sulphates": number, "alcohol": number
    }
    -> { "prediction": "Good", "quality_score": 7, "model": "Random Forest", ... }

Run:
    pip install -r requirements.txt
    python train.py     # only needed once, or whenever you retrain — creates wine_model.joblib
    python app.py        # starts the API on http://localhost:5000
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "wine_model.joblib")

app = Flask(__name__)
CORS(app)  # allow the frontend (opened from file:// or a different origin) to call this API

_bundle = None

def load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "wine_model.joblib not found. Run `python train.py` first to train and save the model."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle

# Maps the JSON field names used by the API to the column names/order the model was trained on
FIELD_TO_COLUMN = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "ph": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}
REQUIRED_FIELDS = list(FIELD_TO_COLUMN.keys())


@app.route("/health", methods=["GET"])
def health():
    try:
        load_bundle()
        return jsonify({"status": "ok", "model_loaded": True})
    except FileNotFoundError as e:
        return jsonify({"status": "error", "model_loaded": False, "message": str(e)}), 503


@app.route("/predict", methods=["POST"])
def predict():
    try:
        bundle = load_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        x = [[float(data[f]) for f in REQUIRED_FIELDS]]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    rf = bundle["random_forest"]
    dt = bundle["decision_tree"]

    rf_pred = rf.predict(x)[0]
    rf_proba = rf.predict_proba(x)[0]
    dt_pred = dt.predict(x)[0]

    confidence = round(float(np.max(rf_proba)) * 100, 1)
    score = bundle["representative_score"][rf_pred]

    return jsonify({
        "prediction": rf_pred,
        "quality_score": score,
        "model": "Random Forest",
        "confidence_percent": confidence,
        "class_probabilities": {cls: round(float(p), 4) for cls, p in zip(rf.classes_, rf_proba)},
        "decision_tree_prediction": dt_pred,
        "models_agree": bool(dt_pred == rf_pred),
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Accepts {"samples": [ {...}, {...} ]} and returns a prediction per sample —
    used by the table-based Predict dashboard when the backend is enabled."""
    try:
        bundle = load_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    body = request.get_json(silent=True)
    if not body or "samples" not in body:
        return jsonify({"error": 'Request body must be {"samples": [...]}'}), 400

    rf = bundle["random_forest"]
    dt = bundle["decision_tree"]
    results = []
    for i, sample in enumerate(body["samples"]):
        missing = [f for f in REQUIRED_FIELDS if f not in sample]
        if missing:
            results.append({"row": i, "error": f"Missing fields: {', '.join(missing)}"})
            continue
        try:
            x = [[float(sample[f]) for f in REQUIRED_FIELDS]]
        except (TypeError, ValueError):
            results.append({"row": i, "error": "All fields must be numeric."})
            continue
        rf_pred = rf.predict(x)[0]
        rf_proba = rf.predict_proba(x)[0]
        dt_pred = dt.predict(x)[0]
        results.append({
            "row": i,
            "prediction": rf_pred,
            "quality_score": bundle["representative_score"][rf_pred],
            "model": "Random Forest",
            "confidence_percent": round(float(np.max(rf_proba)) * 100, 1),
            "decision_tree_prediction": dt_pred,
            "models_agree": bool(dt_pred == rf_pred),
        })
    return jsonify({"results": results})


@app.route("/metrics", methods=["GET"])
def metrics():
    path = os.path.join(APP_DIR, "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "metrics.json not found. Run train.py first."}), 503
    return send_from_directory(APP_DIR, "metrics.json")


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "WineSense AI backend",
        "endpoints": ["GET /health", "POST /predict", "POST /predict/batch", "GET /metrics"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
