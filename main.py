import io
import cv2
import time
import uvicorn
import numpy as np
from PIL import Image
from collections import deque

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from client.base.server import sio_app
from client.thread_manager import ThreadManager
from tools.images_tools import image_decode, image_encode, draw_box, auto_sort, image_resize
import argparse

args = argparse.ArgumentParser()
args.add_argument('-ip', '--host', default='0.0.0.0', help='Host server', type=str)
args.add_argument('-p', '--port', default=5690, help='Port server', type = int)

args = vars(args.parse_args())

host = args['host']
port = args['port']

def respones(code, msg, data):
    res = {
        "code": code,
        "msg": msg,
        "data": data
    }
    return res

# Initiation API server with FastAPI
app = FastAPI()
app.mount('/ws', app=sio_app)
app.mount("/traffic_light", StaticFiles(directory="/traffic_light"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

queue_frame = deque(maxlen=10)
queue_vehicle = deque(maxlen=10)
queue_restream = deque(maxlen=10)
queue_camera = deque(maxlen=2)
queue_event = deque(maxlen=5)

stopstream_image = cv2.imread('/traffic_light/images/endstream.jpg')
_, jpeg_stopimage = cv2.imencode('.jpg', stopstream_image)

# source = 'rtsp://admin1:admin123456_@113.162.227.49:554'

thread_manager = ThreadManager(host, port, queue_camera, queue_frame, queue_vehicle, queue_restream, queue_event)
thread_manager.pustevent.host = host
thread_manager.pustevent.port = port

@app.get('/')
async def hello():
    return 'Application'

@app.get("/stream/mjpeg")
async def video_feed():
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get('/camera/start')
async def start_camera():
    if thread_manager.process.is_alive():
        thread_manager.restart()
    else:
        thread_manager.start()
    return respones(1, 'Started Camera', None)

@app.get('/camera/stop')
async def stop_camera():
    thread_manager.stop()
    return respones(1, 'Stopped Camera', None)

@app.put('/camera/add')
async def add_camera(config : dict):
    thread_manager.add_camera(config)
    if thread_manager.process.is_alive():
        thread_manager.restart()
    else:
        thread_manager.start()
    return respones(code=1, msg='Add camera Successfuly', data=None)

@app.get('/camera/remove')
async def add_camera():
    thread_manager.remove_camera()
    thread_manager.stop()
    return respones(code=1, msg='Add camera Successfuly', data=None)

@app.post('/camera/config/update')
async def update_roi(config_roi : dict):
    thread_manager.update_roi(config_roi = config_roi)
    return respones(code=1, msg='Update Roi Successfuly', data=None)

@app.post('/camera/rtsp/update')
async def update_roi(config_rtsp : dict):
    thread_manager.update_rtsp(config_rtsp = config_rtsp)
    return respones(code=1, msg='Update RTSP Successfuly', data=None)

@app.get('/camera/capture')
async def capture(rtsp:dict):
    frame = thread_manager.cameraread.capture(rtsp['rtsp'])
    if frame is None:
        return respones(code=-1, msg='Faild to capture', data=None)
        
    frame_b64 = image_encode(cv2.cvtColor(image_resize(frame, 1280), cv2.COLOR_BGR2RGB))

    return respones(code=1, msg=None, data={'frame': frame_b64})


@app.post('/lp/detect')
async def recognizer(vehicle_info : dict):
    # print(vehicle_info)
    vehicle_image = image_decode(vehicle_info['vehicle_image'])
    vehicle_image = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)
    vc_type = vehicle_info['type']
    result = thread_manager.extract.recognize_vehicle(vehicle_image, vc_type)
    if result is None:
        return respones(code = -1, msg="Cant find Liscene Plate", data=None)
    lp, lp_image = result
    return respones(code=1, msg=None, data={'lp_image':image_encode(lp_image), 'lp':lp})

@app.post('/vehicle/detect')
async def detect(frame: dict):
    frame = image_decode(frame['frame'])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = thread_manager.process.detector.detect(frame)
    if len(dets):
        for d in dets:
            img = draw_box(frame, d[:4].astype(np.int32), d[-1])
            img = thread_manager.process.vehicle_manager.putText_utf8(img, thread_manager.process.vehicle_manager.type[d[-1]], (int(d[0]), int(d[1])), (255,255,255))
        return respones(code = 1, msg = None, data={'frame': image_encode(img)})

def gen():
    """Video streaming generator function."""
    while True:
        # t1 = time.time()
        if len(queue_frame):
            frame = queue_frame.popleft()
            if frame.shape == ():
                jpeg = jpeg_stopimage
            else:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG")
                jpeg = buffer.getvalue()
        else:
            time.sleep(0.001)
            continue
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


if __name__ == '__main__':
    
    uvicorn.run(app, host=host, port=port, access_log=False)
    
