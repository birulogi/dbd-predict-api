from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')  # file hasil training

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        fitur = [[
            float(data['curah']),
            float(data['suhu']),
            float(data['penduduk']),
            int(data['genangan'])
        ]]
        hasil = model.predict(fitur)[0]
        return jsonify({'hasil': 'Risiko DBD Tinggi' if hasil == 1 else 'Risiko DBD Rendah'})
    except Exception as e:
        return jsonify({'error': str(e)})
