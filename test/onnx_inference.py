import psutil
import onnxruntime as rt
import numpy
import os


#define the priority order for the execution providers
# prefer CUDA Execution Provider over CPU Execution Provider

class ONNXBase(object):
    def __init__(self, weight_path):
        EP_list = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]

        # initialize the model.onnx

        device_name = 'gpu'

        sess_options = rt.SessionOptions()


        self.sess = rt.InferenceSession(weight_path,sess_options,  providers=EP_list)

        # get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
        self.output_name = [self.sess.get_outputs()[i].name for i in range(len(self.sess.get_outputs()))]
        # get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
        self.input_name = self.sess.get_inputs()[0].name

    def infer(self, image):
        
        result = self.sess.run(self.output_name, {self.input_name: image})

        return result
    
    def preprocess(self, image):
        pass
    def postprocess(self, result):
        pass

if __name__ == "__main__":
    m = ONNXBase('vehicle-7.onnx')
    import numpy as np
    import time

    img = np.zeros((1,3,640,640), dtype=np.float32)
    for i in range(100):
        t1 = time.time()
        pred = m.infer(img)
        print(time.time() - t1)
    print(pred[0].shape)