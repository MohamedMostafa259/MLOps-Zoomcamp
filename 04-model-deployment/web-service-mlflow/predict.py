from flask import Flask, request, jsonify
import mlflow 
import joblib
import pandas as pd

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
RUN_ID = "e415ff245ee741548cfd0578fea1d936"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
rf_pipeline = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/models')


def predict(features):
    features_df = pd.DataFrame([features])
    print("Received features:", features_df)
    y_pred = rf_pipeline.predict(features_df)
    return y_pred[0]

app = Flask('duration-prediction')

@app.route('/predict', methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    pred = predict(ride)
    result = {
        'duration': pred, 
        'model_run_id': RUN_ID
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)