import numpy as np
import math
import base64
import cv2


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

def draw_box(frame, bbox, vc_ind):
    color = {2:(120,180,85), 3:(200,130,0), 4:(63,12,144), 5:(155,224,252),0:(120,180,85), 1:(120,180,85)}
    frame = cv2.rectangle( frame,(bbox[0], bbox[1]), (bbox[2], bbox[3]), color[int(vc_ind)], 2)
        # frame = self.putText_utf8(frame, f'{vehicle.lp} !!!', (vehicle.bbox[0],vehicle.bbox[1]), (0, 15, 153))
    # else:
    #     frame = cv2.rectangle(frame,(vehicle.bbox[0], vehicle.bbox[1]), (vehicle.bbox[2], vehicle.bbox[3]), color[vehicle.type],2)
    #     frame = self.putText_utf8(frame, f'{vehicle.lp}', (vehicle.bbox[0],vehicle.bbox[1]), color[vehicle.type])
    return frame

def get_center_bbox(bbox):
    x1,y1,x2,y2 = [int (i) for i in bbox]
    center = (x1+(x2-x1)//2, y1+(y2-y1)//2)
    return center

def auto_sort(pp):
    if not len(pp):
        return pp
    cent=(sum([p[0] for p in pp])/len(pp),sum([p[1] for p in pp])/len(pp))
    pp.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
    return pp

def distance_two_points(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

# import numpy as np
    
def unit_vector(x1, y1, x2, y2):
    # Vector v or differenceh
    v_x = x2 - x1
    v_y = y2 - y1
    # Add inaccuracy to v
    # lenght of v
    v_len = (v_x**2 + v_y**2)**0.5
    # unit vector u of v
    u_x = v_x / v_len
    u_y = v_y / v_len
    return (u_x, u_y)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    # print(f'{v1[0][0]},{v1[0][1]},{v1[1][0]},{v1[1][1]}')
    # print(f'{v2[0][0]},{v2[0][1]},{v2[1][0]},{v2[1][1]}')
    v1_u = unit_vector(v1[0][0],v1[0][1],v1[1][0],v1[1][1])
    v2_u = unit_vector(v2[0][0],v2[0][1],v2[1][0],v2[1][1])
    # print(f'{v1_u}:{v2_u}')
    angle = np.arccos(np.dot(v1_u, v2_u))

    if v2[1][1] - v2[0][1] > 0:
        return -angle*180/np.pi
    else:
        return angle*180/np.pi


def image_decode(base64_img):
    jpg_original = base64.b64decode(base64_img)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image = cv2.imdecode(jpg_as_np, flags=1)
    return image

def image_encode(img):
    _, buffer = cv2.imencode('.jpg', img)
    base64_img = base64.b64encode(buffer)
    return base64_img

def processing_bbox_outside(bbox,img_shape):
    processed_bbox = []
    for box in bbox:
        x1,y1,x2,y2 = [int (i) for i in box]
        
        x1,y1,x2,y2 = x1 if x1 >= 0 else 0,\
                y1 if y1 >= 0 else 0,\
                x2 if x2 <= img_shape[1] else img_shape[1],\
                y2 if y2 <= img_shape[0] else img_shape[0]
        processed_bbox.append([x1,y1,x2,y2])
        
    return np.array(processed_bbox)

