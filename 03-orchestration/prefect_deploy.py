import argparse
import warnings

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from prefect import flow, task
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

@task
def read_dataframe(year, month):
	url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
	df = pd.read_parquet(url).sample(frac=0.01, random_state=42)

	print("Records:", df.shape[0] * 100)

	df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
	df = df.loc[(df['duration'] >= 1) & (df['duration'] <= 60)]

	return df

@task
def preprocessing(df):
	encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
	df_encoded = encoder.fit_transform(df[['PULocationID', 'DOLocationID']])

	X, y = df_encoded, df['duration'].values

	return X, y, encoder

@task
def train_model(X_train, y_train, X_val, y_val, encoder):
	with mlflow.start_run() as run:
		train = xgb.DMatrix(X_train, label=y_train)
		val = xgb.DMatrix(X_val, label=y_val)

		best_params = {
			'colsample_bytree': 0.811456438126501,
			'learning_rate': 0.23687492640963337,
			'max_depth': 17,
			'min_child_weight': 0.6041334208397435,
			'objective': 'reg:squarederror',
			'reg_alpha': 0.05731229437746139,
			'reg_lambda': 0.020764792800741835,
			'seed': 42,
			'subsample': 0.7453194874678659
		}

		mlflow.log_params(best_params)

		booster = xgb.train(
			params=best_params, 
			dtrain=train, 
			num_boost_round=30, 
			evals=[(val, 'validation')], 
			early_stopping_rounds=10, 
			# verbose_eval=10
		)

		y_pred = booster.predict(val)
		rmse = root_mean_squared_error(y_val, y_pred)
		mlflow.log_metric('rmse_val', rmse)

		joblib.dump(encoder, 'encoder.joblib')
		mlflow.log_artifact('encoder.joblib', artifact_path="preprocessing")
		mlflow.xgboost.log_model(booster, artifact_path="models")

		return run.info.run_id

@task
def run(year, month):
	df_train = read_dataframe(year, month)

	next_year = year if month < 12 else year + 1
	next_month = month + 1 if month < 12 else 1 
	df_val = read_dataframe(next_year, next_month)

	X_train, y_train, encoder = preprocessing(df_train)
	X_val, y_val, _ = preprocessing(df_val)

	run_id = train_model(X_train, y_train, X_val, y_val, encoder)
	print(f"MLflow run_id: {run_id}")
	return run_id


@flow
def main(year: int = 2023, month: int = 1):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    print("Tracking URI:", mlflow.get_tracking_uri())
    run_id = run(year=year, month=month)
    print("Run ID:", run_id)

# Follow this link for orchestrating your pipeline using prefect (Prefect section)
# https://github.com/hannarud/mlops-zoomcamp-2025/blob/main/week3_notes.md