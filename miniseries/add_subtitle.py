import os
import cv2
import ffmpeg
import numpy as np

 
def add_subtitle_with_effects(video_path, output_path, fontsize, fontcolor, effect):

    command = f'ffmpeg -i {video_path} -vf "subtitles={effect}:force_style=\'Alignment=2,MarginV=50,Fontsize={fontsize},PrimaryColour=&H{fontcolor[1:]}&\'" -c:a copy {output_path}'
    #command = f'ffmpeg -i {video_path} -vf "subtitles={effect}:force_style=\'Alignment=2, MarginV=1900, Fontsize={fontsize}, PrimaryColour=&H{fontcolor[1:]}&, Fontname=Arial\'" -c:a copy {output_path}'
    #Alignment=2 用于设置字体的对齐方式，2 表示底部居中对齐，
    #Alpha=0.5 的选项来设置字体的透明度为 0.5，
    #Fontname=Arial
    #MarginV表示字幕到视频底部的距离
    os.system(command)

video_path = '/home/guohao826/miniseries/processing_video/nixi_nosubtitle/nixi_5.mp4'
  # 字幕
effect = '/home/guohao826/miniseries/material/'  # 入场/出场特效，例如fade表示淡入淡出效果
effect_path = os.path.join(effect,video_path.rsplit("/")[-1].split('.')[0]+'.ass')
print(effect_path)
#水印字体大小
fontsize = 4
#字幕字体大小
fontsizezimu = 12
fontcolor = '#FFFFFF'  # 字体颜色，使用十六进制表示，例如白色为#FFFFFF
add_subtitle_with_effects(video_path, "/home/guohao826/miniseries/processed_video/nixi_5.mp4", fontsizezimu, fontcolor, effect_path)
