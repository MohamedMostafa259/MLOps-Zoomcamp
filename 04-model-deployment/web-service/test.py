import requests

# features
ride = {
    'PULocationID': 10, 
    'DOLocationID': 50
}

url = 'http://0.0.0.0:9696/predict'
response = requests.post(url, json=ride)
print(response.json())