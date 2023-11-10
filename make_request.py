import requests

url = "http://127.0.0.1:5000/get_recommendations"
params = {"title": "A Promised Land", "num_recommendations": 15}

response = requests.get(url, params=params)

if response.status_code == 200:
    recommendations = response.json()["recommendations"]
    for i, book in enumerate(recommendations, start=1):
        print(f"Book suggested #{i}: {book}")
else:
    print(f"Error: {response.status_code}")
