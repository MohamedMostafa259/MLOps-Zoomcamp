from flask import Flask, request, jsonify
import joblib 
import pandas as pd

lr_model, encoder = joblib.load("lin_reg.bin")

def predict(features):
    features_df = pd.DataFrame([features])
    X = encoder.transform(features_df)
    y_pred = lr_model.predict(X)
    return y_pred[0]

app = Flask('duration-prediction')

@app.route('/predict', methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    pred = predict(ride)
    result = {
        'duration': pred
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)