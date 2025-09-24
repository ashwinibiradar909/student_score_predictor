# predict_test.py
import requests, json

url = "http://127.0.0.1:5000/predict"
payload = {
    "features": {
        "age": 17,
        "sex": "F",
        "studytime": 2,
        "G1": 10,
        "G2": 11,
        "traveltime": 1,
        "famsize": "GT3",
        "Pstatus": "T"
    }
}
resp = requests.post(url, json=payload)
print("status:", resp.status_code)
print("json:", resp.json())
