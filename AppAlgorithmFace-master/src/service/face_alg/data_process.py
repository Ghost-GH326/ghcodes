# encoding: utf-8
# data process class

import multiprocessing as mp
from multiprocessing import Process, Manager
from align_process import  align_paths2queue, AlignProcess, write_info
from detect_process import get_gpu_queue, init_detectinfer, detect_imgs, detect_path2queue


class DataProcess:
    def __init__(self, conf):
        self._conf = conf
        self._output_queue = Manager().Queue()
        self._num_queue = Manager().Queue()

    def _open_wirte_file(self):
        open(self._conf.info_path, 'w').close()
    
    def _read2queue(self):
        conf = self._conf
        if conf.task_name == 'detect':
            self._path_lst = []
            detect_path2queue(conf.path, self._path_lst, 
                              conf.batch_size, conf.line_num)
        elif conf.task_name == 'align':
            self._input_queue = Manager().Queue()
            align_paths2queue(conf.path, self._input_queue, conf.line_num)

    def _detect(self):
        conf = self._conf
        self._read2queue()
        write_proess = Process(target=write_info, args=[conf.info_path,
                       self._output_queue, self._num_queue, conf.line_num,])
        write_proess.start()
        gpu_ids = get_gpu_queue(conf.gpus, conf.process_num)
        process_pool = mp.Pool(processes=conf.process_num, initializer=init_detectinfer,
                       initargs=(gpu_ids, self._output_queue, self._num_queue, 
                       conf.model_path, conf.input_size, conf.network, conf.threshold))
        process_pool.map(detect_imgs, self._path_lst)
        write_proess.join()

    
    def _align(self):
        conf = self._conf
        self._read2queue()
        process_lst = []
        write_proess = Process(target=write_info, args=[conf.info_path,
                       self._output_queue, self._num_queue, conf.line_num,])
        write_proess.start()
        process_lst.append(write_proess)
        j = 0
        for i in range(conf.process_num):
            j += 1
            process_i = AlignProcess(self._input_queue, self._output_queue, 
                        self._num_queue, conf.save_root, conf.align_shape,j)
            process_i.start()
            process_lst.append(process_i)

        for process_i in process_lst:
            process_i.join()

    def run(self):
        self._open_wirte_file()
        conf = self._conf
        if conf.task_name == 'detect':
            self._detect()
        elif conf.task_name == 'align':
            self._align()

