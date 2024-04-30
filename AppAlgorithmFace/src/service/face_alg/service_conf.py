from easydict import EasyDict as edict

align_conf = edict()
aligned_im_shape = [112, 112]
# face adn landmarks  detection model path
align_conf.DETECT_MODEL_PATH = '/Users/wxy/work/AppAlgorithmFace/models/detection_model/retina-0000.params'
align_conf.INPUT_SHAPE = [640, 640]
align_conf.ALIGN_SHAPE = aligned_im_shape
align_conf.THRESHOLD = 0.8
align_conf.NETNAME = 'net3'
align_conf.GPUID = -1  # -1 cpu
align_conf.ERROR = 1e-6
align_conf.ALIGN_PATHS = [
    '/Users/wxy/work/AppAlgorithmFace/models/example/test1.jpg',
    '/Users/wxy/work/AppAlgorithmFace/models/example/test2.jpg',
    '/Users/wxy/work/AppAlgorithmFace/models/example/test3.jpg'
]
align_conf.TEST_ALIGNED_IMG = '/Users/wxy/work/AppAlgorithmFace/models/example/test_align.png'

feature_conf = edict()
feature_conf.FEATURE_MODEL_PATH = '/home/guohao826/AppAlgorithmFace/models/feature_model/res50_fullada_msra_045.pth'
feature_conf.MODEL_CONFIG = '/home/guohao826/AppAlgorithmFace/models/feature_model/train_res50.json5'
feature_conf.GPUID = -1  # -1 cpu
feature_conf.BATCH_SIZE = 16
feature_conf.INPUT_SHAPE = aligned_im_shape
feature_conf.POSITIVE_SCORE = 0.4672134015251498
feature_conf.NEGATIVE_SCORE = -0.07803277941195955
feature_conf.ERROR = 1e-6
feature_conf.ALIGN_PATHS = [
    '/Users/wxy/work/AppAlgorithmFace/models/example/test1.jpg',
    '/Users/wxy/work/AppAlgorithmFace/models/example/test2.jpg',
    '/Users/wxy/work/AppAlgorithmFace/models/example/test3.jpg'
]
