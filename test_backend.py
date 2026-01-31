import requests
import base64
import json

# Configuration
API_URL = "http://127.0.0.1:8000/api/analyze"

# Sample 1x1 pixel red dot image (Base64 encoded)
SAMPLE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

def test_analyze():
    print(f"Testing API at: {API_URL}")
    
    payload = {
        "image": SAMPLE_IMAGE_B64,
        "description": "I have a mild headache and some fever.",
        "uid": "test_user_123", # Optional: Replace with a real UID if you want to test Supabase fetching
        "language": "english"
    }

    try:
        response = requests.post(API_URL, json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("\nSuccess! Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("\nError Response:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server.")
        print("Make sure the backend is running (uvicorn app:app --reload)")

if __name__ == "__main__":
    test_analyze()
