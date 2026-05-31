
import requests
import os

url = 'http://127.0.0.1:5000/upload'
file_path = r'c:\Users\Amir khan\Downloads\final project\data\ECHONET-Dynamic\Videos\0X43A5C5FAAFCA1B2C.avi'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, 'rb') as f:
    files = {'video': f}
    try:
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
