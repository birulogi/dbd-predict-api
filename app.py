from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # agar bisa diakses dari HTML luar

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API Prediksi DBD Aktif"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([
            float(data["curah"]),
            float(data["suhu"]),
            float(data["penduduk"]),
            float(data["genangan"])
        ]).reshape(1, -1)

        hasil = model.predict(features)[0]
        return jsonify({"hasil": str(hasil)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()
