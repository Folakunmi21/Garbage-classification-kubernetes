import requests

url = 'http://localhost:8080/predict'

request = {
    "url": "https://upload.wikimedia.org/wikipedia/commons/e/e3/Bouteille.jpg"
}

response = requests.post(url, json=request)

try:
    result = response.json()
except Exception:
    print(f"Error! Status Code: {response.status_code}")
    print("Full Response Content:")
    print(response.text) # This will show the actual Python error traceback
    exit()

print(f"Top prediction: {result['top_class']} ({result['top_probability']:.2%})")
print(f"\nAll predictions:")
for cls, prob in result['predictions'].items():
    print(f"  {cls:12s}: {prob:.2%}")