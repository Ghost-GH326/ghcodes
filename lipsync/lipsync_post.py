import requests

url = "https://api.synclabs.so/lipsync"

payload = {
  "synergize": True,
  "videoUrl": "https://aiplatform-databackups.oss-cn-hangzhou.aliyuncs.com/AIGC_VIDEOS/tmp_videos/3.12_11.mp4?OSSAccessKeyId=LTAI4GDxGEBdhZiuziL4j5cB&Expires=1710231322&Signature=sfEqv5YrN844fsP5Ztz9VPs00CU%3D",
  "audioUrl": "https://aiplatform-databackups.oss-cn-hangzhou.aliyuncs.com/AIGC_VIDEOS/tmp_videos/3.12_1.mp3?OSSAccessKeyId=LTAI4GDxGEBdhZiuziL4j5cB&Expires=1710231218&Signature=Byx2uEQpz7rgyVxOozG2kA5o44g%3D"
}

headers = {
    "x-api-key": "2d4d8c36-4ec3-4acd-8b47-3c4a9144d79d",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)