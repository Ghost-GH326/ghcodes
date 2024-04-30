import requests

url = "https://api.synclabs.so/lipsync/2d42a0f1-9636-4e09-8eeb-c0ab343f68ed"

headers = {"x-api-key": "2d4d8c36-4ec3-4acd-8b47-3c4a9144d79d"}

response = requests.request("GET", url, headers=headers)

print(response.text)


# 
# 