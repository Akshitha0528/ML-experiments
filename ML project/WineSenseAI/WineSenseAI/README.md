# WineSense AI — Wine Quality Prediction

Two ways to run predictions, both using the **same real trained model**:

1. **Client-side (no setup)** — `index.html` has the trained Random Forest embedded as JSON
   and re-implements sklearn's tree traversal in JavaScript. Just open the file in a browser.
   Verified to match scikit-learn's own `.predict()` output exactly.
2. **Backend API (optional)** — `app.py` is a Flask server that loads the same model via
   joblib and exposes it over HTTP, matching the `POST /predict` spec from the original brief.
   Useful if you want a real client/server architecture to demonstrate, or want to call the
   model from something other than the browser (e.g. Postman, another app).

## What's real
- `winequality-red.csv` — the actual UCI Wine Quality (red wine) dataset, 1,599 samples.
- `train.py` — trains a Decision Tree and Random Forest with GridSearchCV (5-fold) + 5-fold
  cross-validation, evaluates both on a held-out test split (accuracy, precision, recall, F1,
  confusion matrix, ROC/AUC), and produces:
  - `wine_model.joblib` — the trained sklearn models, for the Flask backend.
  - `model_bundle.json` — the same Random Forest (100 trees) + Decision Tree, serialized as
    plain arrays for the browser-side JS engine (already embedded in `index.html`).
  - `metrics.json` — the real evaluation numbers shown on the Results page / served by `/metrics`.

Quality classes: raw scores (3–8) are grouped into Poor (≤4), Average (5–6), Good (≥7).
"Quality score" on a prediction is the average actual rating of training samples in that
class — a real statistic, not a per-sample regression.

## Running the backend

```bash
pip install -r requirements.txt
python train.py        # trains the models and writes wine_model.joblib, metrics.json, etc.
python app.py           # starts the API on http://localhost:5000
```

### Endpoints

**GET `/health`**
```json
{ "status": "ok", "model_loaded": true }
```

**POST `/predict`** — single wine sample
```json
// Request
{
  "fixed_acidity": 7.9, "volatile_acidity": 0.35, "citric_acid": 0.46,
  "residual_sugar": 3.6, "chlorides": 0.078, "free_sulfur_dioxide": 15,
  "total_sulfur_dioxide": 37, "density": 0.9973, "ph": 3.35,
  "sulphates": 0.86, "alcohol": 12.8
}
```
```json
// Response
{
  "prediction": "Good",
  "quality_score": 7,
  "model": "Random Forest",
  "confidence_percent": 78.2,
  "class_probabilities": { "Average": 0.2176, "Good": 0.7819, "Poor": 0.0005 },
  "decision_tree_prediction": "Good",
  "models_agree": true
}
```

**POST `/predict/batch`** — multiple samples at once (`{"samples": [ {...}, {...} ]}`),
returns one result per row — matches the table-based Predict page in `index.html`.

**GET `/metrics`** — the real evaluation numbers as JSON.

## Connecting the frontend to the backend
`index.html` works fully standalone with no backend. If you'd rather have it call the live
Flask API instead of running the model in-browser, replace the `predictRow()` function's body
with a `fetch('http://localhost:5000/predict', {...})` call using the field names shown above —
the response shape already matches what the UI expects (`prediction`, `quality_score`, `model`).
Note the backend must be running locally (`python app.py`) for this to work, and the page must
be served over http(s) rather than opened via `file://` for the request to succeed in most browsers.

## Re-running training
Re-run `python train.py` any time — e.g. after swapping in the combined red+white dataset, a
different binning scheme, or your own data. It regenerates `wine_model.joblib`, `model_bundle.json`,
and `metrics.json` together, so the backend and the browser-side model always agree. If you're using
the client-side version, re-embed the two updated JSON files into `index.html` in place of the
`<script type="application/json">` blocks near the end of the file.
