#!/usr/bin/env python
# coding=utf-8
import mxnet as mx
import numpy as np
from mxnet import ndarray as nd
from multiprocessing import Process


class TestProcess(Process):
      def __init__(self, dev, num):
          Process.__init__(self)
          #self.dev = dev
          self.num = num
          self.model = get_model(dev)
          
      def run(self):
          forward(self.model, self.num)
          
def get_model(dev):
      prefix = './model/retina'
      epoch = 0
      global model
      sym , arg, aux = mx.model.load_checkpoint(prefix, epoch)
      model = mx.mod.Module(symbol=sym, context=mx.gpu(dev), label_names=None)
      model.bind(data_shapes=[('data', (1, 3, 640, 640))], for_training=False)
      model.set_params(arg, aux)
      return model
    
def forward(model, num):
      print('process :{}'.format(num))
      im_tensor = np.zeros((3, 3, 640, 640))
      data = nd.array(im_tensor)
      db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
      model.forward(db, is_train=False)
      net_out = model.get_outputs()
      out = net_out
      print(len(out))
  
if __name__ == '__main__':
      process_ls = []
      dev = 0
      for i in range(1):
          model = get_model(dev)
          process_i = Process(target=forward, args=[model, i,])#TestProcess(0, i)
          process_ls.append(process_i)
          process_i.start()
      for p_i in process_ls:
          p_i.join()
