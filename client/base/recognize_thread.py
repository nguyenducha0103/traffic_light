import threading
import ctypes
import cv2
import time
import numpy as np
from client.vehicle_models.lp_recognizer import LP_RECOGNIZER
from client.vehicle_models.trt_model import TrtModel
# from onnx_face.FaceDetector import FaceDetector
# from onnx_face.FaceExtractor import FaceExtractor
import pycuda.driver as cuda

class ExtractionThread(threading.Thread):
    def __init__(self, queue_vehicle):
        threading.Thread.__init__(self )
        # self.id = self.camera.get_streamID()

        self.stream_capture = ''
        self.name = 'ExtractionThread'
        self.stop_flag = False
        # self.extractor = FaceExtractor()
        # self.extractor = TritonExtractor()
        # _, self.jpeg_stopimage = cv2.imencode('.jpg',self.stopstream_image) 
        self.queue_vehicle = queue_vehicle
        # self.extractor = FaceExtractor()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        self.plugin = '/traffic_light/client/vehicle_models/plugins/libyolo_layer.so'
        self.char_detector = TrtModel(model = '/traffic_light/client/vehicle_models/weights/lp-128x352.trt', plugin=self.plugin, cuda_ctx=self.ctx, category_num=31, letter_box=False)
        self.lp_detector = TrtModel(model = '/traffic_light/client/vehicle_models/weights/lp-480.trt', plugin=self.plugin, cuda_ctx=self.ctx, category_num=2, letter_box=False)
 
        self.lp_recognizer = LP_RECOGNIZER(self.lp_detector, self.char_detector)
    
    def recognize_vehicle(self, vehicle_image, type):
        lp_result = self.lp_recognizer.recognize(vehicle_image, type)
        if lp_result is None:
            return None
        
        lp_img = lp_result['img']
        for a in lp_result['detections']:
            cv2.rectangle(lp_img,(a[2],a[3]),(a[4],a[5]),(255,0,255),1)
            cv2.putText(lp_img, a[0], (a[2], a[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        return lp_result['lp'], lp_img
    
    def run(self):
        while not self.stop_flag:
            if len(self.queue_vehicle):
                vehicle = self.queue_vehicle.popleft()

                image = vehicle.vehicle_image
                typ = vehicle.type
                if 0 not in image.shape:
                    lp_result = self.lp_recognizer.recognize(image, typ)
                    
                    if lp_result is not None:
                        vehicle.lp = lp_result['lp']
                        vehicle.score = lp_result['score']
                    
                        if vehicle.score > 82:
                            vehicle.lp_image = lp_result['img']
                            vehicle.identied = True
            time.sleep(0.001)
        
        # time.sleep(0.01)

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure') 

    def stop(self):
        self.stop_flag = True
        # self.ffmpegrestream.stop()
    # def get_camera_id(self):