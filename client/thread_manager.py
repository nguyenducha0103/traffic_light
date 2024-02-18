from client.base.recognize_thread import ExtractionThread
from client.base.process_thread import ProcessThread
from client.restreaming.camera_read import CameraReading
from client.restreaming.ffmpeg_restream import FfmpegRestream
from client.base.push_event_thread import PustEvent
import time

class ThreadManager():
    def __init__(self,host, port, queue_camera, queue_frame, queue_vehicle, queue_restream, queue_event):
        self.cameraread = CameraReading(queue_camera)
        self.process = ProcessThread(queue_camera=queue_camera, queue_frame=queue_frame, queue_vehicle = queue_vehicle, queue_restream= queue_restream, queue_event=queue_event)
        self.extract = ExtractionThread(queue_vehicle)
        self.restream = FfmpegRestream(queue_frame)
        self.pustevent = PustEvent(queue_event=queue_event, host=host, port=port)

        self.extract.daemon = True
        self.process.daemon = True
        self.cameraread.daemon = True
        self.restream.daemon = True
        self.pustevent.daemon =True

    def add_camera(self, config):
        rtsp = config['rtsp']
        red_roi = config['red_roi']
        process_roi = config['process_roi']
        traffic_light_roi = config['traffic_light_roi']
        self.cameraread.load_rtsp(rtsp=rtsp)
        self.process.camera.load_roi(red_roi, process_roi, traffic_light_roi)

    def remove_camera(self):
        pass

    def update_roi(self, config_roi):
        red_roi = config_roi['red_roi']
        process_roi = config_roi['process_roi']
        traffic_light_roi = config_roi['traffic_light_roi']
        self.process.camera.load_roi(red_roi, process_roi, traffic_light_roi)

    def update_rtsp(self, config_rtsp):
        rtsp = config_rtsp['rtsp']
        self.cameraread.load_rtsp(rtsp)

    def start(self):
        self.cameraread.start()
        self.process.start()
        self.extract.start()
        self.pustevent.start()
        # self.restream.start()

    def stop(self):
        self.process.is_pause = True
        self.cameraread.is_pause = True

    def restart(self):
        self.cameraread.is_pause = False
        self.process.is_pause = False