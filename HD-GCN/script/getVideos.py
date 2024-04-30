import requests
from bs4 import BeautifulSoup
import re
import os

# 腾讯视频网页链接
url = "https://v.qq.com/x/cover/mzc001002m4o16l/o00308f9sjw.html"

# 发送 GET 请求
response = requests.get(url)
html_content = response.text

# 使用 BeautifulSoup 解析 HTML
soup = BeautifulSoup(html_content, "html.parser")

# 找到视频标题
title_tag = soup.find("h1", class_="video_title")
video_title = title_tag.text.strip()

# 找到视频播放地址
video_play_url = soup.find("video", id="txp_video_player").get("src")

# 找到字幕文件地址
subtitle_url = soup.find("div", id="mod_cover_subtitle").find("a").get("href")

# 下载视频
video_response = requests.get(video_play_url, stream=True)
with open(f"{video_title}.mp4", "wb") as video_file:
    for chunk in video_response.iter_content(chunk_size=1024):
        if chunk:
            video_file.write(chunk)

# 下载字幕文件
subtitle_response = requests.get(subtitle_url)
with open(f"{video_title}.srt", "wb") as subtitle_file:
    subtitle_file.write(subtitle_response.content)

print("视频和字幕下载完成！")
