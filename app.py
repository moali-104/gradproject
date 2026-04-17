from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# ---- Patch لحل مشكلة sklearn version compatibility ----
def patch_model(model):
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            if hasattr(est, 'tree_') and not hasattr(est, 'monotonic_cst'):
                est.monotonic_cst = None
    return model

# ---- تحميل الموديلز والـ Scaler ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Loading models...")

scaler              = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
adaboost_model      = patch_model(joblib.load(os.path.join(BASE_DIR, "top3_adaboost.pkl")))
random_forest_model = patch_model(joblib.load(os.path.join(BASE_DIR, "top3_random_forest.pkl")))

try:
    gradient_boosting_model = patch_model(joblib.load(os.path.join(BASE_DIR, "top3_gradient_boosting.pkl")))
    GRADIENT_AVAILABLE = True
    print("Gradient Boosting loaded!")
except Exception as e:
    GRADIENT_AVAILABLE = False
    print(f"Warning: Gradient Boosting not loaded: {e}")

print("Models ready!")

FEATURE_NAMES = [
    'A1', 'A2', 'A3', 'A4', 'A5',
    'A6', 'A7', 'A8', 'A9', 'A10',
    'Age', 'Sex', 'Jauundice', 'Family_ASD'
]

app = Flask(__name__)

def extract_features(data):
    features, missing = [], []
    for f in FEATURE_NAMES:
        if f not in data:
            missing.append(f)
        else:
            features.append(float(data[f]))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    raw = np.array(features).reshape(1, -1)
    return scaler.transform(raw)  # ✅ تطبيق الـ Scaler

def make_prediction(model, features):
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0].tolist()
    return {
        "prediction": prediction,
        "label": "ASD Positive" if prediction == 1 else "ASD Negative",
        "probability": {"negative": round(proba[0], 4), "positive": round(proba[1], 4)}
    }

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ASD Prediction API",
        "endpoints": {
            "POST /predict/adaboost":          "AdaBoost",
            "POST /predict/random_forest":     "Random Forest",
            "POST /predict/gradient_boosting": "Gradient Boosting",
            "POST /predict/all":               "All models + Majority Vote"
        },
        "required_features": FEATURE_NAMES
    })

@app.route("/predict/adaboost", methods=["POST"])
def predict_adaboost():
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No JSON body"}), 400
        result = make_prediction(adaboost_model, extract_features(data))
        return jsonify({"model": "AdaBoost", **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/random_forest", methods=["POST"])
def predict_random_forest():
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No JSON body"}), 400
        result = make_prediction(random_forest_model, extract_features(data))
        return jsonify({"model": "Random Forest", **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/gradient_boosting", methods=["POST"])
def predict_gradient_boosting():
    if not GRADIENT_AVAILABLE:
        return jsonify({"error": "Gradient Boosting not available. Need scikit-learn==1.3.0"}), 503
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No JSON body"}), 400
        result = make_prediction(gradient_boosting_model, extract_features(data))
        return jsonify({"model": "Gradient Boosting", **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/all", methods=["POST"])
def predict_all():
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "No JSON body"}), 400
        features = extract_features(data)
        results = {
            "adaboost":      make_prediction(adaboost_model, features),
            "random_forest": make_prediction(random_forest_model, features),
        }
        if GRADIENT_AVAILABLE:
            results["gradient_boosting"] = make_prediction(gradient_boosting_model, features)
        votes = [r["prediction"] for r in results.values()]
        majority = 1 if votes.count(1) > votes.count(0) else 0
        return jsonify({
            "results": results,
            "majority_vote": {"prediction": majority, "label": "ASD Positive" if majority == 1 else "ASD Negative"}
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5000)