from client.vehicle_models.red_light_check import TraficLightClassifier
import numpy as np 
import cv2
import time
from tools.images_tools import auto_sort

class Camera(object):
    def __init__(self):
        self.red_roi = np.array(auto_sort([(10,360),(1200,360),(10,710), (1200,710)]))
        self.process_roi = np.array(auto_sort([(1,150),(1279,150),(1279,710), (1,710)]))
        self.traffic_light_roi = np.array([1023, 60, 1066,166])

        self.TrafficLightClassifer = TraficLightClassifier()
        self.red_light = False
        self.red_time = 0
    
    def load_roi(self, red_roi = None, process_roi = None, traffic_light_roi = None):
        if red_roi is not None and len(red_roi):
            self.red_roi = np.array(auto_sort(red_roi))
        
        if process_roi is not None and len(process_roi):
            self.process_roi = np.array(auto_sort(process_roi))
        
        if traffic_light_roi is not None and len(traffic_light_roi):
            self.traffic_light_roi = np.array(traffic_light_roi)

    def update_roi(self, process_roi):
        self.process_roi = np.array(auto_sort(process_roi))

    def red_light_check(self, frame):
        x1,y1,x2,y2 = self.traffic_light_roi
        traffic_color = self.TrafficLightClassifer.predict(frame[y1:y2, x1:x2])
        return traffic_color

    def update_stat(self, frame):
        # color = {'red':(0, 15, 153), 'green':(20,20,255), 'yellow':(0,255,100)}
        traffic_color = self.red_light_check(frame)
        cv2.polylines(frame, [self.process_roi.reshape((-1, 1, 2))], True, (60, 163, 255), 1)
        x1,y1,x2,y2 = self.traffic_light_roi

        if traffic_color == 'red':
            # frame = vehicle_manager.putText_utf8(frame, f'Stop!', (x1,y1-8), (0, 15, 153), background = True)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255),2)
            # Update Traffic light
            self.red_light = True
            self.red_time = time.time()
            cv2.polylines(frame, [self.red_roi.reshape((-1, 1, 2))], True, (20,20,255), 1)
        elif traffic_color == 'green':
            # Update Traffic light
            self.red_light = False
            cv2.polylines(frame, [self.red_roi.reshape((-1, 1, 2))], True, (0,255,100), 1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,100),2)
        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255),2)
            cv2.polylines(frame, [self.red_roi.reshape((-1, 1, 2))], True, (20,20,255), 1)
        return frame
    
class CameraManager(object):
    def __init__(self) -> None:
        pass