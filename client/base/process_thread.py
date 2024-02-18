import os
import cv2
import time
import uuid
import ctypes
import json
import threading
import numpy as np
import pycuda.driver as cuda
import datetime

from client.vehicle_models.vehicle_detector import VehicleDetector
from client.tracking.bytetrack.byte_tracker import BYTETracker
from client.base.object_manager import VehicleManager
from client.vehicle_models.red_light_check import TraficLightClassifier
from tools.images_tools import image_resize
from client.base.camera import Camera

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class ProcessThread(threading.Thread):
    def __init__(self, queue_frame, queue_camera, queue_vehicle, queue_restream, queue_event):
        threading.Thread.__init__(self)

        self.queue_frame = queue_frame
        self.queue_camera = queue_camera
        self.queue_vehicle = queue_vehicle
        self.queue_restream = queue_restream
        self.queue_event = queue_event

        self.device = cuda.Device(0)
        self.cuda_ctx = self.device.make_context()
        self.name = 'ProcessThread'
        self.stop_flag = False
        self.push = True

        self.vehicle_manager = VehicleManager()
        self.detector = VehicleDetector(engine_path='/traffic_light/client/vehicle_models/weights/VehicleDetector.trt',ctx=self.cuda_ctx, n_classes=6, input_shape=(640, 640))
        self.stopstream_image = cv2.imread("images/endstream.jpg")
        self.tracker = BYTETracker()
        self.color_classifer = TraficLightClassifier()

        self.camera = Camera()
        self.is_pause = False

    def draw_box(self, frame, bbox, vc_ind):
        color = {2:(120,180,85), 3:(200,130,0), 4:(63,12,144), 5:(155,224,252),0:(120,180,85), 1:(120,180,85)}
        frame = cv2.rectangle( frame,(bbox[0], bbox[1]), (bbox[2], bbox[3]), color[int(vc_ind)], 2)
            # frame = self.putText_utf8(frame, f'{vehicle.lp} !!!', (vehicle.bbox[0],vehicle.bbox[1]), (0, 15, 153))
        # else:
        #     frame = cv2.rectangle(frame,(vehicle.bbox[0], vehicle.bbox[1]), (vehicle.bbox[2], vehicle.bbox[3]), color[vehicle.type],2)
        #     frame = self.putText_utf8(frame, f'{vehicle.lp}', (vehicle.bbox[0],vehicle.bbox[1]), color[vehicle.type])
        return frame
    
    def in_roi(self, point, roi_type):
        point = Point(point)
        polygon = Polygon(roi_type)
        return polygon.contains(point)
    
    def save_event(self, frame, vehicle):
        p1 = vehicle.track_point[0]
        p2 = vehicle.track_point[int(len(vehicle.track_point)/2)]
        p3 = vehicle.track_point[-1]
        cv2.circle(frame, p1, 2, (0,0,255), -1)
        cv2.circle(frame, p2, 2, (0,0,255), -1)
        cv2.circle(frame, p3, 2, (0,0,255), -1)
        cv2.line(frame, p1, p2, (0,0,255),1)
        cv2.line(frame, p2, p3, (0,0,255),1)
        frame = cv2.rectangle(frame,(vehicle.bbox[0], vehicle.bbox[1]), (vehicle.bbox[2], vehicle.bbox[3]), (0, 0, 2550),1)
        return frame
    
    def run(self):
        while not self.stop_flag:
            if self.is_pause:
                self.queue_frame.append(self.stopstream_image)
                time.sleep(0.1)
                continue
            
            if len(self.queue_camera):
                frame = self.queue_camera.popleft()
                if frame is not None:
                    t1 = time.time()
                    frame = image_resize(frame, width=1280)
                    if self.push:
                        violated_frame = np.copy(frame)

                    # Do inference
                    detections = self.detector.detect(frame, conf_thres=0.5)
                    if len(detections):
                        bboxes, scores, classes = detections[:,:4].astype(np.int32), detections[:,4], detections[:,5].astype(np.int32)
                        
                        for b, c in zip(bboxes, classes):
                            if self.in_roi(self.vehicle_manager.cal_center(b), self.camera.process_roi):
                                self.draw_box(frame, b, c)

                        bcl = np.concatenate((bboxes, np.expand_dims(classes, 1)), axis=1)
                        
                        # Do track
                        track_result = np.array(self.tracker.update(bboxes,scores, bcl), dtype = 'O')
                        new_tracks = []
                        
                        for track in track_result:
                            track_box = track[5][:4]
                            track_id = np.expand_dims(track[4],0)
                            track_cls = np.expand_dims(track[5][4],0)
                            new_track = np.concatenate([track_box, track_id, track_cls], axis = 0)
                            new_tracks.append(new_track)
                        track_result = new_tracks

                        self.vehicle_manager.update_tracking(track_result, frame, self.queue_vehicle)
                    
                    # Update status for camera such as ROI, red time, red line, ...
                    frame = self.camera.update_stat(frame)
                    
                    # n = len(self.temporary_delete.keys())
                    # while n > 0:
                    #     lst_key = list(self.temporary_delete.keys())
                    #     per_id = lst_key[n-1]
                    #     per = self.temporary_delete[per_id]
                    #     if time.time() - per.delete_moment > 20:
                    #         print(f'[Post Event:{per.name} : {strftime("%Y-%m-%d %H:%M:%S", localtime(per.last_time))}]')
                    #         self.temporary_delete.pop(per_id)
                    #     n -= 1
                    
                    for vehicle in self.vehicle_manager.list_vehicle:
                        if time.time() - vehicle.last_time > 2:
                            # if vehicle.identied:
                            #     vehicle.delete_moment = time.time()
                            #     self.temporary_delete.update({vehicle.id:vehicle})

                            self.vehicle_manager.dict_vehicle.pop(str(vehicle.track_id))
                            self.vehicle_manager.list_vehicle.remove(vehicle)
                            del vehicle
                            continue
                        
                        if self.in_roi(vehicle.center_box, self.camera.process_roi):
                            if not vehicle.identied:
                                if vehicle.type == 'person':
                                    vehicle.identied = True
                                else:
                                    self.queue_vehicle.append(vehicle)

                            # if vehicle.id in self.temporary_delete.keys():
                            #     old_vehicle = self.temporary_delete[vehicle.id]
                            #     self.retracking(vehicle, old_vehicle)
                            #     self.temporary_delete.pop(vehicle.id)
                            #     vehicle.delete_moment = 0
                            
                            # Logic detect red light violation
                            if not vehicle.violated:
                                if self.in_roi(vehicle.center_box, self.camera.red_roi):
                                    vehicle.in_roi_before = True
                                else:
                                    if vehicle.time_out_roi == 0:
                                        vehicle.time_out_roi = time.time()
                                    if vehicle.in_roi_before and self.camera.red_light:
                                        if self.camera.red_time - vehicle.time_out_roi < 0.5: 
                                            vehicle.violated = True
                                            if self.push:
                                                violated_frame = self.save_event(violated_frame, vehicle)
                                                self.queue_event.append([vehicle, violated_frame])
                                                
                                
                            if time.time() - vehicle.last_time < 0.1:
                                # draw text only
                                frame = self.vehicle_manager.draw(frame, vehicle, vehicle.violated)    
                                if vehicle.violated:
                                    frame = cv2.rectangle(frame, (vehicle.bbox[0], vehicle.bbox[1]), (vehicle.bbox[2], vehicle.bbox[3]), (0, 20, 153),2)
                    #Calc time for display FPS
                    t2 = time.time()
                    frame = self.vehicle_manager.putText_utf8(frame, f'FPS: {int(1/(t2 - t1))}',(10, 50), ((102,104,177)))
                    
                    self.queue_frame.append(frame)
                    self.queue_restream.append(frame)
            else:
                time.sleep(0.02)

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