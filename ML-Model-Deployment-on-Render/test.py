import requests

url = 'http://127.0.0.1:5000/recommend'
data = {'movie_name': 'Inception'}  
response = requests.post(url, json=data)

print(response.json())
