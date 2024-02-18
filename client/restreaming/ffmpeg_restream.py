import threading
import ctypes
import time
import subprocess
import requests
import os
import signal
import json
import cv2

def run_commandline(popenargs):
    process = subprocess.Popen(popenargs, shell=True, stdout=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(input)
        retcode = process.poll()
        
        process.kill()
        process.wait()
    except:
        process.kill()
        process.wait()
        raise subprocess.CalledProcessError(
            retcode, process.args, output=stdout, stderr=stderr)
    return retcode, stdout.decode("utf-8"), stderr


class FfmpegRestream(threading.Thread):
    def __init__(self, queue_frame):
        threading.Thread.__init__(self)
        self.name = 'FfmpegRestreamThread'
        self.fps = 20
        # self.width = int(3072*thread_handler.output_scale_ratio)
        # self.height = int(1728*thread_handler.output_scale_ratio)
        
        self.width = int(1280)
        self.height = int(720)
        
        self.queue_frame = queue_frame

        self.command = []
        self.process = None
        
        if os.name == "nt":
            self.os_type = "windows"
        else:
            self.os_type = "linux"
            
        self.stop_flag = False
        self.appname = 'live'
        self.id = 1
        self.hls_restream = {
            "rtmp": f"rtmp://10.70.39.204:1935/{self.appname}/{self.id}",
            "flv": f"http://10.70.39.204:7001/{self.appname}/{self.id}.flv",
            "hls": f"http://10.70.39.204:7002/{self.appname}/{self.id}.m3u8"}
        
    def register_room(self):
        camera_code = self.id
        register_status = False
        print('restream room registering....')
        try:
            register_url = "http://10.70.39.204:8090/control/get?room="+str(camera_code)  
            req = requests.get(url=register_url)
            req_text = json.loads(req.text)
            register_status = req_text.get('status')
            print('register successful!')
            print(self.hls_restream)
        except Exception as e:
            print("register error:", e)
            self.process = None
            
        if register_status:
            code_room = req_text['data']
            rtmp_url = "rtmp://10.70.39.204:1936/live/" + str(code_room)
            self.command = ['ffmpeg',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(self.fps),
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-f', 'flv',
                    '-loglevel', 'panic',
                    rtmp_url]
            
            # command = ['ffmpeg',
            #         '-y',
            #         '-f', 'rawvideo',
            #         '-vcodec', 'rawvideo',
            #         '-pix_fmt', 'bgr24',
            #         '-s', "{}x{}".format(width, height),
            #         '-r', str(60),
            #         '-i', '-',
            #         '-c:v', 'libx264',
            #         '-pix_fmt', 'yuv420p',
            #         '-preset', 'ultrafast',
            #         '-f', 'flv',
            #         rtmp_url]
            # rtmp_restream = rtmp://localhost:1935/{appname}/movie
            
            return True
        else:
            self.command = []
            return False
        # rtmp_url = "rtmp://10.70.39.204:1936/live/L17LTlsVqMNTZyLKMIFSD2x28MlgPJ0SDZVHnHJPxMKi0tWx"
        
    def run(self):
        # self.p = subprocess.Popen(self.command, stdin=subprocess.PIPE,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while not (self.stop_flag):
            # print('is_start_camera:',thread_handler.is_start_camera)
            if self.process is None:
                # if thread_handler.is_start_camera:
                is_reg_room = self.register_room()
                if is_reg_room == True:
                    if self.os_type == "windows":
                        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, shell=True)
                    else:
                        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
                # time.sleep(1)
            else:
                if len(self.queue_frame):
                    frame = self.queue_frame.popleft()
                    # print('1')
                    if frame.shape != (): 
                        # print('2')
                        # print('frame.shape[0]:',frame.shape[0])
                        # print('self.height:',self.height)
                        resized_frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_AREA)

                        # if frame.shape[0] == self.height:
                            # print('3')
                        try:
                            # print('4')
                            self.process.stdin.write(resized_frame.tobytes())
                            # print('sended')
                        except Exception as e:
                            print('restream process error:', e)
                            # stdout, stderr = self.process.communicate(input)
                            # retcode = self.process.poll()
                            self.process.kill()
                            self.process.wait()
                            self.process = None
                            # if not (self.stop_flag):
                            #     if self.os_type == "windows":
                            #         self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, shell=True)
                            #     else:
                            #         self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
                else:
                    time.sleep(0.016)
            time.sleep(0.05)

        # self.process.kill()
        # self.process.wait()
            # print('time:',time.time()-sttime)
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
        # os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        if self.process != "":
            # os.kill(self.process.pid, signal.CTRL_C_EVENT)
            self.process = ""
        # self.process.kill()
        # self.process.wait()

