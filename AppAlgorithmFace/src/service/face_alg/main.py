# encoding: utf-8
# main file for benchmark

import argparse
from data_process import DataProcess


def get_config():
    parser = argparse.ArgumentParser(description='DataProcess config')
    common = parser.add_argument_group('common')
    common.add_argument('--path', type=str,
                         default='/home/guohao826/AppAlgorithmFace/src/service/face_alg/process_file/imgs', 
                         help='path of file to process')
    common.add_argument('--info-path', type=str,
                         default='write_info', 
                         help='path of info file')
    common.add_argument('--process-num', type=int,
                         default='1', 
                         help='number of process')
    common.add_argument('--line-num', type=int,
                         default=1000,
                         help='number of lines of file')
    common.add_argument('--task-name', type=str,
                         default='align',
                         help='name of task')

    align = parser.add_argument_group('align_face')
    align.add_argument('--save-root', type=str,
                         default='alignedimgs', 
                         help='the path to save aligned images')
    align.add_argument('--align-shape', type=str,
                         default='112,112', 
                         help='shape of  aligned images')
    
    detect = parser.add_argument_group('detect_facc')
    detect.add_argument('--model-path', type=str,
                         default='path/to/mxnet/detect/model', 
                         help='path to detection model')
    detect.add_argument('--input-size', type=str,
                         default='640,640', 
                         help='inpue size of  detection model')
    detect.add_argument('--threshold', type=float,
                         default=0.8, 
                         help='threshold of  detection model')
    detect.add_argument('--gpus', type=str,
                         default='0', 
                         help='avaliable gpu ids')
    detect.add_argument('--network', type=str,
                         default='net3', 
                         help='net name of  detection model')
    detect.add_argument('--batch-size', type=int,
                         default=16, 
                         help='batch size of  detection model')

    args = parser.parse_args()
    args.align_shape = [int(i) for i in args.align_shape.split(',')]
    args.input_size = [int(i) for i in args.input_size.split(',')]
    args.gpus = [int(i) for i in args.gpus.split(',')]
    return args

if __name__ == '__main__':
    conf = get_config()
    dataprocess = DataProcess(conf)
    dataprocess.run()
