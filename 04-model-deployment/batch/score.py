import argparse
import uuid
from datetime import datetime

import joblib
import mlflow
import pandas as pd
from dateutil.relativedelta import relativedelta
from prefect import flow, get_run_logger, task


@task
def generate_uuids(n: int):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


@task
def read_dataframe(input_file):
	df = pd.read_parquet(input_file).sample(frac=0.01, random_state=42)

	print(f"Records: (input_file: {input_file})", df.shape[0] * 100)

	df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
	df = df.loc[(df['duration'] >= 1) & (df['duration'] <= 60)]

	return df


@task
def load_model(run_id):
    # The model should be fetched from a remote shared storage like an s3 instance:
    # logged_model = f's3://mlflow-models-alexey/1/{run_id}/artifacts/model'
    # model = mlflow.sklearn.load_model(logged_model)
    model = joblib.load('lin_reg.bin')
    return model


@task
def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    df_result['tpep_dropoff_datetime'] = df['tpep_dropoff_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_parquet(output_file, index=False)


@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()

    logger.info(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    df_preprocessed = df[['PULocationID', 'DOLocationID']]

    logger.info(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)

    logger.info(f'applying the model...')
    y_pred = model.predict(df_preprocessed)

    logger.info(f'saving the result to {output_file}...')

    save_results(df, y_pred, run_id, output_file)
    return output_file

@task
def get_paths(run_date, taxi_type, run_id):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month 

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month:02d}.parquet'
    output_file = f'./trip-duration-prediction/taxi_type={taxi_type}/year={year:04d}/month={month:02d}.parquet'

    return input_file, output_file


@flow
def ride_duration_prediction(taxi_type: str, run_id: str, run_date: datetime):
    input_file, output_file = get_paths(run_date, taxi_type, run_id)
    apply_model(input_file=input_file, run_id=run_id, output_file=output_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Apply a model to predict taxi trip duration.')
	parser.add_argument('--taxi_type', type=str, required=True, help='Taxi type of the data to train on (green or yellow)')
	parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
	parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
	# we'll skip this for now because we're not using an s3 bucket 
    # parser.add_argument('--run_id', type=str, required=True, help='RUN ID of the model')
	args = parser.parse_args()

    run_date = datetime(year=args.year, month=args.month, day=1)
    ride_duration_prediction(taxi_type=args.taxi_type, run_id=None, run_date=run_date)

