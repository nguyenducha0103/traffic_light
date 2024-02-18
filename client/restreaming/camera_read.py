import time
import cv2
import time
import threading
import ctypes
from PIL import Image
# from ffmpeg_restream import FfmpegRestream

class CameraReading(threading.Thread):
    def __init__(self, queue_camera):
        threading.Thread.__init__(self)
        self.queue_camera = queue_camera
        # self.id = self.camera.get_streamID()
        self.name = 'CameraReading'
        self.stop_flag = False
        self.stopstream_image = cv2.imread("images/endstream.jpg")

        self.is_pause = False
        self.source = None
        self.fps = 30
        self.stream_capture = None

    def load_rtsp(self, rtsp):
        if rtsp is None:
            return False
        
        stream_capture = cv2.VideoCapture(rtsp)
        _, frame = stream_capture.read()
        if not _:
            return False

        stream_capture.release()

        self.source = rtsp
        self.restream_flag = False
        self.stream_capture = cv2.VideoCapture(self.source)
        
        self.stream_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.fps = self.stream_capture.get(cv2.CAP_PROP_FPS)
        self.sleep_time = 0.1 if self.fps <= 0 else 1/self.fps

        return True
        
    def capture(self, rtsp):
        print(rtsp)
        if rtsp is not None:
            stream_capture = cv2.VideoCapture(rtsp)
            _, frame = stream_capture.read()
            stream_capture.release()
            if not _:
                return None
            return frame
        return None


    def run(self):
        while not self.stop_flag:
            if self.is_pause:
                time.sleep(1)
                continue
            
            if self.stream_capture.isOpened():
                ret, frame = self.stream_capture.read()
                if not ret:
                    # self.stop()
                    self.stream_capture = cv2.VideoCapture(self.source)
                    time.sleep(0.06)
                    frame = self.stopstream_image
                
            else:
                frame = self.stopstream_image
            self.queue_camera.append(frame)
            time.sleep(0.06)

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