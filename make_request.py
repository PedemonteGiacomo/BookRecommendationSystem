import requests

url = "https://blooming-ridge-95700-a04b44e99aa6.herokuapp.com/get_recommendations"
params = {"title": "A Promised Land", "num_recommendations": 15}

response = requests.get(url, params=params)

if response.status_code == 200:
    recommendations = response.json()["recommendations"]
    for i, book in enumerate(recommendations, start=1):
        print(f"Book suggested #{i}: {book}")
else:
    print(f"Error: {response.status_code}")
