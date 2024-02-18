import threading
import cv2
import socket
import numpy as np
import ctypes
import socketio
import time
import datetime
import os
import uuid


class PustEvent(threading.Thread):
    def __init__(self, queue_event, host='10.70.39.40', port=5690, topic = 'event'):
        threading.Thread.__init__(self)

        self.queue_event = queue_event
        self.stop_flag = False
        self.sio_client = socketio.Client()
        self.host = host
        self.port = port

        self.url = f'http://{self.host}:{self.port}'
        
        self.stop_flag = False
        self.topic = topic
    
    def send(self, message):
        self.sio_client.emit(self.topic, message)
    
    def connect(self):
        self.sio_client.connect(self.url, socketio_path='ws/sockets')

    
    def create_event(self, vehicle, violation_frame):
        event = {'time': datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
        event.update({'type':vehicle.type})
        event.update({'LP':''})

        lp = None
        if len(vehicle.lp):
            name = vehicle.lp + '-' + uuid.uuid4().hex
            lp = vehicle.lp_image
            event.update({'LP':vehicle.lp})
        else:
            name = uuid.uuid4().hex
        vehicle_folder = os.path.join('/traffic_light/storage/', name)
        os.makedirs(vehicle_folder, exist_ok=True)

        path_lp = os.path.join(vehicle_folder, 'lp.jpg')
        path_frame = os.path.join(vehicle_folder, 'frame.jpg')

        event.update({'frame_path': f'http://{self.host}:{self.port}'+path_frame})
        event.update({'lp_path': ''})
        
        if lp is not None:
            cv2.imwrite(path_lp, lp)
            event.update({'lp_path': f'http://{self.host}:{self.port}'+path_lp})

        cv2.imwrite(path_frame, violation_frame)
        return event

    def run(self):
        self.connect()
        while not self.stop_flag:
            if len(self.queue_event):
                vehicle, violation_frame = self.queue_event.popleft()
                event = self.create_event(vehicle, violation_frame)
                msg = {'event': event}
                self.send(message=msg)
            time.sleep(0.01)

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