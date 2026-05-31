import requests
import os

url = "http://127.0.0.1:5000/upload"
file_path = "uploads/0X100E491B3CD58DE2.avi"

if not os.path.exists(file_path):
    print(f"File {file_path} not found.")
else:
    with open(file_path, 'rb') as f:
        print(f"Uploading {file_path}...")
        files = {'video': (os.path.basename(file_path), f, 'video/avi')}
        try:
            r = requests.post(url, files=files)
            print(f"Status Code: {r.status_code}")
            try:
                print("Response JSON:", r.json())
            except:
                print("Response Text:", r.text)
        except Exception as e:
            print(f"Request failed: {e}")
