import os
import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from flask import Flask, render_template, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook"
    "/master/breast-cancer-data.csv"
)

FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst"
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")
app = Flask(__name__)

def build_candidates() -> dict:
    # return the pool of candidates models to compare.

    model_dt = DecisionTreeClassifier(random_state=42)
    return {
        "Random Forest Model": RandomForestClassifier(
            n_estimators=250,
            random_state=42
        ),

        "Support Vector Machine (SVM) Model": SVC(
            kernel='rbf',
            probability=True
        ),

        "Decision Tree Model": model_dt,

        "AdaBoost Model": AdaBoostClassifier(
            estimator=model_dt,
            learning_rate=0.1,
            n_estimators=120,
            random_state=42
        ),

        "CatBoost Model": CatBoostClassifier(
            iterations=400,
            learning_rate=0.1,
            verbose=False
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=400,
            early_stopping=True,
            random_state=42
        )
    }

def select_best_model():
    if os.path.exists(MODEL_PATH):
        logger.info('Loading Cached Best Model From %s', MODEL_PATH)
        cached = joblib.load(MODEL_PATH)
        return cached['model'], cached['name'], cached['leaderboard']

    logger.info('Training and Comparing Candidate Models (First Run)...')
    df = pd.read_csv(DATASET_URL)

    # M = Malignant; B = Benign
    df['diagnosis'] = df['diagnosis'].map(
        {
            'M': 1,
            'B': 0
        }
    )

    X = df[FEATURES]
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=42, stratify=y)
    leaderboard = {}
    fitted_models = {}
    for name, clf in build_candidates().items():
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        leaderboard[name] = accuracy
        fitted_models[name] = clf
        logger.info('%-15s test accuracy: %.3f%%', name, accuracy * 100)


    best_name = max(leaderboard, key=leaderboard.get)
    best_model = fitted_models[best_name]
    logger.info(
        "Best Model: %s (%.3f%%) - Deploying This One.",
        best_name,
        leaderboard[best_name] * 100
    )


    joblib.dump(
        {
            "model": best_model,
            "name": best_name,
            "leaderboard": leaderboard
        }, MODEL_PATH
    )
    return best_model, best_name, leaderboard


# Train/compare once at startup, not on every request.
model, model_name, leaderboard = select_best_model()

def parse_inputs(form) -> tuple[list[float] | None, str | None]:
    # Validate and parse the five numeric fields from the submitted form.
    values = []
    for i in range(1, 15):
        raw = form.get(f"query{i}", "").strip()
        try:
            values.append(float(raw))
        except ValueError:
            return None, f"'{FEATURES[i-1]}' must be a number (got: '{raw}')"
    return values, None


@app.route("/", methods=["GET"])
def index():
    return render_template(
        'home.html',
        model_name=model_name,
        leaderboard=leaderboard
    )

@app.route("/", methods=["POST"])
def predict():
    form_values = {f"query{i}": request.form.get(f"query{i}", "") for i in range(1, 15)}
    values, error = parse_inputs(request.form)

    if error:
        return render_template(
            "home.html",
            error=error,
            model_name=model_name,
            leaderboard=leaderboard,
            **form_values
        )

    input_df = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0,1]

    if prediction == 1:
        output1 = "The Patient is Diagnosed With Breast Cancer"
        output2 = f"Confidence: {probability * 100:.2f}%"
    else:
        output1 = "The Patient is Not Diagnosed With Breast Cancer"
        output2 = f"Confidence: {(1 - probability) * 100:.2f}%"

    return render_template(
        "home.html",
        output1=output1,
        output2=output2,
        model_name=model_name,
        leaderboard=leaderboard,
        **form_values
    )


if __name__ == "__main__":
    # Debug = False for Anything Beyond Local Testing
    app.run(host="127.0.0.1", port=5000, debug=True)