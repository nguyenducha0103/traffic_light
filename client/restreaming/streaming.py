
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
import uvicorn
import argparse
# from camera_service.cameraip.v1_0.thread_manager import ThreadControler
# from camera_service.webcam.v1_0.thread_manager import ThreadControler

import cv2
import time 

app = FastAPI()
from camera_read import CameraReading
# import cv2
import sys
# appname = "live"
# restream_url = {
#     "rtmp": f"rtmp://10.70.39.204:1935/{appname}/webcam",
#     "flv": f"http://10.70.39.204:7001/{appname}/webcam.flv",
#     "hls": f"http://10.70.39.204:7002/{appname}/webcam.m3u8",
# }
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", nargs="+",
    help="camera source", required=True)
ap.add_argument("-i", "--ip", type=str, default="0.0.0.0",
    help="ip address of the device")
ap.add_argument("-o", "--port", type=int, default=55555,
    help="ephemeral port number of the server (1024 to 65535)")

args = vars(ap.parse_args())

print(args["source"])

source = args['source']
stopstream_image = cv2.imread("endstream.jpg")
_, jpeg_stopimage = cv2.imencode('.jpg',stopstream_image)
cameras = {}
if len(source):
    for i,s in enumerate(source):

        queue_camera = {
            'frame': None,
            'source': int(s) if s.isnumeric() else s
        }

        camera = CameraReading(queue_camera)
        camera.daemon = True
        cameras[i] = camera
        cameras[i].start()



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

@app.get("/{stream_id}")
async def video_feed(stream_id: int):
    if stream_id not in cameras:
        return "Not exist!"
    
    return StreamingResponse(gen(cameras[stream_id]), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get('/stop/{stream_id}')
async def stop_stream(stream_id: int):
    if stream_id in cameras:
        cameras[stream_id].stop()
        cameras[stream_id] = None
        cameras.pop(stream_id)

        return "stopped"
    
    return "Not exist!"

def gen(camera):
    """Video streaming generator function."""
    print(camera)
    while True:
        time.sleep(0.02)
        frame = camera.queue_camera['frame']
        frame = image_resize(frame,width=1280,height=720)
        if frame.shape == ():
            jpeg = jpeg_stopimage
        else:
            _, jpeg = cv2.imencode('.jpg', frame)
            
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
host_ip = str("http://") + str(s.getsockname()[0])+":"+str(args["port"])+"/"
s.close()

print(f'Uvicorn running on {host_ip} (Press CTRL+C to quit)')
if __name__ == '__main__':
    
	# start a thread that will perform motion detection
    uvicorn.run(app, host=args["ip"], port=args["port"], access_log=False)
    

    
