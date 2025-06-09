import argparse
import warnings

import mlflow
import pandas as pd
import prefect
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")


def read_dataframe(year, month):
	url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
	df = pd.read_parquet(url).sample(frac=0.01, random_state=42)
	
	print("Records:", df.shape[0] * 100)

	df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
	df = df.loc[(df['duration'] >= 1) & (df['duration'] <= 60)]

	return df

def preprocessing(df):
	encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
	df_encoded = encoder.fit_transform(df[['PULocationID', 'DOLocationID']])

	X, y = df_encoded, df['duration'].values

	return X, y, encoder

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
			verbose_eval=10
		)

		y_pred = booster.predict(val)
		rmse = root_mean_squared_error(y_val, y_pred)
		mlflow.log_metric('rmse', rmse)

		mlflow.sklearn.log_model(encoder, artifact_path="artifacts")
		mlflow.xgboost.log_model(booster, artifact_path="artifacts")

		return run.info.run_id

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
	parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
	parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
	args = parser.parse_args()

	run_id = run(year=args.year, month=args.month)

	with open('run_id.txt', 'w') as f:
		f.write(run_id)


# Now, we have an ML pipeline; even though we didn't use any
# workflow orchestrator here So some
# things that can happen is um let's say u
# network is down and we cannot read the
# file So
# um in order to uh deal with this problem
# we need to d to to add some sort of
# retrial mechanism right
# or I don't know what else could can go
# wrong or for example here again network
# is down and we cannot log the parameter
# so we need again to add some retrial
# mechanism here we don't want to
# implement it ourselves so we can uh
# relate we can rely on um workflow
# extensation tool to do it for us So if
# it sees that uh read data frame fails it
# would just re-execute it and then maybe
# if it uh executes it three times and all
# three times fails that it fails the
# entire job Um so yeah we have the
# foundation ready to actually use a
# workflow orchestrator if we want but if
# we don't we already have a training
# pipeline that we can just execute and
# get the model Okay so that's it And um
# um yeah right now in the next video um
# so if you're watching this in 2025
# uh we don't have a designated special
# orchestrator for this year for this
# cohort Uh which means that you can
