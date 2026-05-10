# =========================================================
# TRANSFORMER OIL AI ANALYZER
# SIMULATION + REALTIME MODE
# RANDOM FOREST REGRESSION
# RENDER READY
# =========================================================

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import requests
import random
import joblib
import os

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("clustered_regression_output.csv")

print("\nDataset Loaded Successfully")

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[['Light', 'Temp', 'Cluster']]

y = df['Mix']

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

X = X.fillna(X.mean())

# =========================================================
# SCALE FEATURES
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# RANDOM FOREST REGRESSION
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== MODEL PERFORMANCE =====")

print(f"R2 Score : {r2:.4f}")

print(f"MAE      : {mae:.4f}")

print(f"RMSE     : {rmse:.4f}")

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(model, "oil_quality_model.pkl")

joblib.dump(scaler, "oil_scaler.pkl")

print("\nAI Model Saved Successfully")

# =========================================================
# ESP32 URL
# =========================================================

ESP32_URL = "http://10.128.96.179/"

# =========================================================
# CURRENT MODE
# =========================================================

CURRENT_MODE = "simulation"

# =========================================================
# SIMULATION DATA GENERATOR
# =========================================================

def generate_fake_data():

    oil_type = random.choice([
        "PURE",
        "LOW",
        "MODERATE",
        "HIGH"
    ])

    # PURE OIL
    if oil_type == "PURE":

        light = random.randint(950, 1300)
        temp = random.randint(35, 50)

    # LOW ADULTERATION
    elif oil_type == "LOW":

        light = random.randint(700, 950)
        temp = random.randint(40, 55)

    # MODERATE ADULTERATION
    elif oil_type == "MODERATE":

        light = random.randint(450, 700)
        temp = random.randint(45, 60)

    # HIGH ADULTERATION
    else:

        light = random.randint(150, 450)
        temp = random.randint(50, 75)

    return light, temp


# =========================================================
# GET REALTIME ESP32 DATA
# =========================================================

def get_realtime_data():

    try:

        response = requests.get(
            ESP32_URL,
            timeout=3
        )

        data = response.json()

        light = float(data['light'])

        temp = float(data['temp'])

        return light, temp

    except Exception as e:

        print("ESP32 ERROR :", e)

        return None, None


# =========================================================
# AUTO CLUSTER
# =========================================================

def get_cluster(light):

    if light > 1000:

        return 0

    elif light > 800:

        return 1

    elif light > 600:

        return 2

    elif light > 400:

        return 3

    else:

        return 4


# =========================================================
# QUALITY CLASSIFICATION
# =========================================================

def classify_quality(predicted_mix):

    if predicted_mix < 10:

        return "PURE OIL"

    elif predicted_mix < 30:

        return "LOW ADULTERATION"

    elif predicted_mix < 60:

        return "MODERATE ADULTERATION"

    else:

        return "HIGH ADULTERATION"


# =========================================================
# HOME PAGE
# =========================================================

@app.route('/')

def home():

    return render_template(
        "index.html",
        mode=CURRENT_MODE
    )


# =========================================================
# CHANGE MODE
# =========================================================

@app.route('/set_mode/<mode>')

def set_mode(mode):

    global CURRENT_MODE

    CURRENT_MODE = mode

    return jsonify({
        "mode": CURRENT_MODE
    })


# =========================================================
# ANALYZE OIL SAMPLE
# =========================================================

@app.route('/analyze')

def analyze():

    # -----------------------------------------------------
    # SIMULATION MODE
    # -----------------------------------------------------

    if CURRENT_MODE == "simulation":

        light, temp = generate_fake_data()

    # -----------------------------------------------------
    # REALTIME MODE
    # -----------------------------------------------------

    else:

        light, temp = get_realtime_data()

        if light is None:

            return jsonify({

                "status": "error",

                "message": "ESP32 NOT CONNECTED"

            })

    # -----------------------------------------------------
    # AUTO CLUSTER
    # -----------------------------------------------------

    cluster = get_cluster(light)

    # -----------------------------------------------------
    # PREPARE MODEL INPUT
    # -----------------------------------------------------

    input_df = pd.DataFrame(
        [[light, temp, cluster]],
        columns=['Light', 'Temp', 'Cluster']
    )

    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    scaled_input = scaler.transform(input_df)

    # -----------------------------------------------------
    # AI PREDICTION
    # -----------------------------------------------------

    predicted_mix = model.predict(
        scaled_input
    )[0]

    # -----------------------------------------------------
    # QUALITY CLASSIFICATION
    # -----------------------------------------------------

    quality = classify_quality(
        predicted_mix
    )

    # -----------------------------------------------------
    # RETURN RESPONSE
    # -----------------------------------------------------

    return jsonify({

        "status": "ok",

        "mode": CURRENT_MODE,

        "light": round(light, 2),

        "temp": round(temp, 2),

        "cluster": cluster,

        "mix": round(predicted_mix, 2),

        "quality": quality,

        "time": datetime.now().strftime("%H:%M:%S"),

        "r2_score": round(r2, 4),

        "mae": round(mae, 4),

        "rmse": round(rmse, 4)

    })


# =========================================================
# RUN APP
# =========================================================

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )