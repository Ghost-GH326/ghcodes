from paddleocr import PaddleOCR, draw_ocr
import ffmpeg
import subprocess
import cv2
import os

video_path = '/home/guohao826/miniseries/material/nixi/nixi_5.mp4'
images_folder = '/home/guohao826/miniseries/frames/nixi_5'


video = cv2.VideoCapture(video_path)
ff_video = ffmpeg.probe(video_path)
# 获取视频的帧率
fps = int(video.get(cv2.CAP_PROP_FPS))
# fps = int(int(ff_video['streams'][0]['r_frame_rate'].split('/')[0]) / int(ff_video['streams'][0]['r_frame_rate'].split('/')[1]))
# 获取视频的宽度和高度
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("height:",height,"width:",width)
#记录视频帧数
frame_count = 0
#设置抽帧
print(f"帧率：{fps}")
interval = int(fps)

#遍历视频
while True:
    ret, frame = video.read()
    # 如果没有帧了，退出循环
    if not ret:
        break
    # 抽帧
    if frame_count % interval == 0:
        # 保存帧
        frame_filename = f"nixi_5_{frame_count//fps}.jpg"
        frame_path = os.path.join(images_folder, frame_filename)
        cropped_frame = frame[int(height/2):height,0:width]
        cv2.imwrite(frame_path, cropped_frame)
    frame_count+=1



# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
images = os.listdir(images_folder)
subtitle_list = []
for imagename in images:
    img_path = os.path.join(images_folder, imagename)
    result = ocr.ocr(img_path, cls=True)
    time = int(imagename.rsplit('_')[-1].split('.')[0])
    print(time)
    if time//60 < 10:
        minute = f'0{time//60}'
    else:
        minute = f'{time//60}'
    if time%60 < 10:
        second = f'0{time%60}'
    else:
        second = f'{time%60}'

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            # print(f"{minute}:{second}_{imagename}:",line[1][0])
            if time%60+1 < 10:
                subtitle_list.append( f'Dialogue: 0,0:{minute}:{second}.00,0:{minute}:0{time%60+1}.00,Default,,0,0,0,,{{\\fad(1,1)}}{line[1][0]}')
            elif time%60 != 59:
                subtitle_list.append( f'Dialogue: 0,0:{minute}:{second}.00,0:{minute}:{time%60+1}.00,Default,,0,0,0,,{{\\fad(1,1)}}{line[1][0]}')
            elif time//60+1 <10:
                subtitle_list.append( f'Dialogue: 0,0:{minute}:{second}.00,0:0{time//60+1}:00.00,Default,,0,0,0,,{{\\fad(1,1)}}{line[1][0]}')
            else:
                subtitle_list.append( f'Dialogue: 0,0:{minute}:{second}.00,0:{time//60+1}:00.00,Default,,0,0,0,,{{\\fad(1,1)}}{line[1][0]}')
print(len(subtitle_list))
for i in subtitle_list:
    print(i)



# ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
# img_path = '/home/guohao826/miniseries/material/ocr_drug.jpg'
# result = ocr.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)



# # 显示结果
# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores)
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')