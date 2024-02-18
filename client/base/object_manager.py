import time
from PIL import ImageFont, ImageDraw, Image
# import taichi as ti
import numpy as np


class Person():
    def __init__(self, id):
        self.id = -1
        
        self.track_id = id
        self.vehicle = None
        self.bbox = []
        self.first_time = 0
        self.last_time = 0
        self.delete_moment = 0

        self.identied = False


class Vehicle():
    def __init__(self, id):
        self.id = -1
        
        self.track_id = id
        self.vehicle = None
        self.bbox = []
        self.center_box = ()

        self.first_time = 0
        self.last_time = 0
        self.delete_moment = 0
        
        self.lp_image = None
        self.identied = False
        self.type = None
        self.lp = ''
        self.score = 0

        self.in_roi_before = False

        self.violated = False
        self.time_out_roi = 0
        self.track_point = []


class VehicleManager():

    def font(self,font_size=20):
        return ImageFont.truetype("font/RobotoSlab-Regular.ttf", font_size)
    
    def __init__(self):
        self.list_vehicle = []
        self.dict_vehicle = {}

        self.current_time = time.time()
        self.font = self.font

        self.temporary_delete = {}
        self.type = {0:'person', 1:'bicycle', 2:'motorbike', 3:'car',4:'truck', 5:'bus'}

    def cal_center(self, box):
        return (int((box[2]-box[0])/2 + box[0]), int((box[3]-box[1])/2 + box[1]))

    def draw(self, frame, vehicle, violation = False):
        color = {'motorbike':(120,180,85), 'car':(167,120,0), 'truck':(63,12,144), 'bus':(155,224,252),'person':(120,180,85), 'bicycle':(120,180,85)}
        if violation:
            # frame = cv2.rectangle(frame,(vehicle.bbox[0], vehicle.bbox[1]), (vehicle.bbox[2], vehicle.bbox[3]), (0, 20, 153),2 )
            frame = self.putText_utf8(frame, f'{vehicle.lp} !!!', (vehicle.bbox[0],vehicle.bbox[1]-2), (0, 15, 153))
        else:
            # frame = cv2.rectangle(frame,(vehicle.bbox[0], vehicle.bbox[1]), (vehicle.bbox[2], vehicle.bbox[3]), color[vehicle.type],2)
            frame = self.putText_utf8(frame, f'{vehicle.lp}', (vehicle.bbox[0],vehicle.bbox[1]), color[vehicle.type])
        return frame

    def add_vehicle(self, vehicle):
        self.list_vehicle.append(vehicle)
        self.dict_vehicle.update({str(vehicle.track_id): vehicle})

    def update_tracking(self, tracking_results, frame, queue_vehicle):

        self.current_time = time.time()
        # self.current_datetime = datetime.datetime.now()
        for track in tracking_results:
            # track_bbox = track[0]
            track_bbox = np.array(track[:4]).astype(np.int32)
            x1,y1,x2,y2 = track_bbox
            track_id = track[4]
            vehicle_id = str(track_id)

            if int(track[5]) in list(self.type.keys()):
                type = track[5]
                # update vehicle info if vehicle in queue
                if vehicle_id in self.dict_vehicle:
                    vehicle = self.dict_vehicle[vehicle_id]
                    vehicle.bbox = track_bbox
                    vehicle.center_box = self.cal_center(track_bbox)

                    vehicle.last_time = time.time()
                    vehicle.vehicle_image = frame[y1:y2,x1:x2]
                    vehicle.type = self.type[int(type)]
                    vehicle.track_point.append(self.cal_center(track_bbox))
                # create new vehicle info if tracking not in queue
                else:
                    new_vehicle_id = vehicle_id
                    vehicle = Vehicle(new_vehicle_id)
                    vehicle.bbox = track_bbox
                    vehicle.center_box = self.cal_center(track_bbox)
                    vehicle.type = self.type[int(type)]

                    vehicle.first_time = time.time()
                    vehicle.last_time = time.time()
                    
                    try:
                        vehicle.vehicle_image = frame[y1:y2,x1:x2]
                    except:
                        print('Cant label for vehicle')

                    self.add_vehicle(vehicle)
      

    def putText_utf8(self, img, text, pos, color, background = True):
        x, y = pos
        pil_image = Image.fromarray(img)

        draw = ImageDraw.Draw(pil_image)

        if background:
            bbox = draw.textbbox((x,y-13), text, font=self.font(13))
            draw.rectangle(bbox, fill=color)

        bbox = draw.text((x,y-13), text, font=self.font(13), fill=(255,255,255))

        image = np.asarray(pil_image)
        return image

    
    def cvt_color_putText_BGR2RGB(self, color):
        b,g,r = color
        return (r, g, b)
    
    def retracking(self, vehicle_new, vehicle_old):
        vehicle_new.vehicle_image = vehicle_old.vehicle_image
        vehicle_new.first_time = vehicle_old.first_time

    