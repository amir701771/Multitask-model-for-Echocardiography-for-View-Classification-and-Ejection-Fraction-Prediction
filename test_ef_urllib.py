import urllib.request
import urllib.parse
import json
import os
import mimetypes
import uuid

url = "http://127.0.0.1:5000/upload"
file_path = "uploads/0X100E491B3CD58DE2.avi"

boundary = uuid.uuid4().hex
headers = {
    "Content-Type": f"multipart/form-data; boundary={boundary}",
    "User-Agent": "Python-urllib-test"
}

def encode_multipart_formdata(fields, files, boundary):
    body = []
    for key, value in fields.items():
        body.append(f'--{boundary}'.encode('utf-8'))
        body.append(f'Content-Disposition: form-data; name="{key}"'.encode('utf-8'))
        body.append(''.encode('utf-8'))
        body.append(value.encode('utf-8'))
    
    for key, (filename, file_handle, content_type) in files.items():
        body.append(f'--{boundary}'.encode('utf-8'))
        body.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"'.encode('utf-8'))
        body.append(f'Content-Type: {content_type}'.encode('utf-8'))
        body.append(''.encode('utf-8'))
        body.append(file_handle.read())
    
    body.append(f'--{boundary}--'.encode('utf-8'))
    body.append(''.encode('utf-8'))
    return b'\r\n'.join(body)

if not os.path.exists(file_path):
    print(f"File {file_path} not found.")
else:
    # print(f"Uploading {file_path}...")
    with open(file_path, 'rb') as f:
        data = encode_multipart_formdata({}, {'video': (os.path.basename(file_path), f, 'video/avi')}, boundary)
        
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            # print(f"Status Code: {response.status}")
            response_body = response.read().decode('utf-8')
            data = json.loads(response_body)
            print(f"EF Value: {data.get('ef_value')}")
            print(f"EF Category: {data.get('ef_category')}")
            print(f"View: {data.get('label')}")
    except Exception as e:
        print(f"FAILED: {e}")
