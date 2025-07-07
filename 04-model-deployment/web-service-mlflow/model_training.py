import argparse
import warnings

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


def read_dataframe(year, month):
	url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
	df = pd.read_parquet(url).sample(frac=0.01, random_state=42)

	print(f"Records (Year: {year}, Month: {month}):", df.shape[0] * 100)

	df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
	df = df.loc[(df['duration'] >= 1) & (df['duration'] <= 60)]

	return df

def X_y_split(df):
	X = df[['PULocationID', 'DOLocationID']]
	y = df['duration']
	return X, y

def train_model(X_train, y_train, X_val, y_val):
    with mlflow.start_run() as run:
        pipeline = make_pipeline(
            OneHotEncoder(drop='first', handle_unknown='ignore'),
            RandomForestRegressor(random_state=42)
        )
        pipeline.fit(X_train, y_train)  

        y_pred = pipeline.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse_val", rmse)

        mlflow.sklearn.log_model(pipeline, artifact_path="models")
        return run.info.run_id

def run(year, month):
	df_train = read_dataframe(year, month)

	next_year = year if month < 12 else year + 1
	next_month = month + 1 if month < 12 else 1 
	df_val = read_dataframe(next_year, next_month)

	X_train, y_train = X_y_split(df_train)
	X_val, y_val = X_y_split(df_val)

	run_id = train_model(X_train, y_train, X_val, y_val)
	print(f"MLflow run_id: {run_id}")
	return run_id

if __name__ == "__main__":
	mlflow.autolog()
	parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
	parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
	parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
	args = parser.parse_args()

	run_id = run(year=args.year, month=args.month)
	print("Run ID:", run_id)
