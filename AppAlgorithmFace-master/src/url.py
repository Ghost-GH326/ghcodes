'''
对外暴露的 url
'''
from src.handler import ping, face, test

url_patterns = [
    (r"/", ping.PingHandler),
    (r"/ping", ping.PingHandler),
    (r"/face/ping", ping.PingHandler),
    # 人脸
    (r"/face/alive/session", face.FaceAliveSessionHandler),
    (r"/face/alive/signup", face.FaceAliveScoreHandler),  # path 不可更改，代码中有判断
    (r"/face/alive/prepare", face.FaceAlivePrepareHandler),
    (r"/face/alive/recognize", face.FaceAliveScoreHandler),
    (r"/face/alive/track", face.FaceAliveTrackHandler),  # 追踪历史信息
    (r"/face/anti/recognize", face.FaceAntiScoreHandler),  # 人脸防伪接口
    # 测试
    (r"/test/feature", test.FaceFeatureHandler),
    (r"/test/video/upload", test.UploadVideoHandler),
]
