import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import sys

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
LEN_ALL_RESULT = 6001
LEN_ONE_RESULT = 6

def preprocess(raw_bgr_image, input_shape):
    """
    description: Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    input_w, input_h = input_shape
    image_raw = raw_bgr_image
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image

def xywh2xyxy(input_shape, origin_h, origin_w, x):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
    """
    input_w, input_h = input_shape
    y = np.zeros_like(x)
    r_w = input_w / origin_w
    r_h = input_h / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h

    return y

def convert_points(boxes, old_size, new_size):
    points = boxes.astype(float)

    old_h, old_w = old_size
    new_h, new_w = new_size

    r_w, r_h = 1.0 / old_w * new_w, 1.0 / old_h * new_h
    points[:, [0, 2]] *= r_w
    points[:, [1, 3]] *= r_h
    
    return points.astype(int)

def postprocess(output, origin_h, origin_w, input_shape, score_thr=0.5, nms_thr = 0.45):
    """
    description: postprocess the prediction
    param:
        output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
        origin_h:   height of original image
        origin_w:   width of original image
    return:
        result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
        result_scores: finally scores, a numpy, each element is the score correspoing to box
        result_classid: finally classid, a numpy, each element is the classid correspoing to box
    """
    # Get the num of boxes detected
    num = int(output[0])
    # Reshape to a two dimentional ndarray
    pred = np.reshape(output[1:], (-1, LEN_ONE_RESULT))[:num, :]
    pred = pred[:, :6]
    # Do nms
    boxes = non_max_suppression(pred, origin_h, origin_w, conf_thres=score_thr, nms_thres=nms_thr)
    result_boxes = boxes[:, :4] if len(boxes) else np.array([])
    # result_boxes = convert_points(result_boxes, (origin_w, origin_h), input_shape)
    result_scores = boxes[:, 4] if len(boxes) else np.array([])
    result_classid = boxes[:, 5] if len(boxes) else np.array([])

    result = np.concatenate([result_boxes,np.expand_dims(result_scores,0),np.expand_dims(result_classid,0)], axis=1)
    return result

def bbox_iou( box1, box2, x1y1x2y2=True):
    """
    description: compute the IoU of two bounding boxes
    param:
        box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
        x1y1x2y2: select the coordinate format
    return:
        iou: computed iou
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                    np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(prediction, origin_h, origin_w,input_shape = (640,640), conf_thres=0.5, nms_thres=0.4):
    """
    description: Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    param:
        prediction: detections, (x1, y1, x2, y2, conf, cls_id)
        origin_h: original image height
        origin_w: original image width
        conf_thres: a confidence threshold to filter detections
        nms_thres: a iou threshold to filter detections
    return:
        boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
    """
    # Get the boxes that score > CONF_THRESH
    boxes = prediction[prediction[:, 4] >= conf_thres]
    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    boxes[:, :4] = xywh2xyxy(input_shape, origin_h, origin_w, boxes[:, :4])
    # clip the coordinates
    boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
    # Object confidence
    confs = boxes[:, 4]
    # Sort by the confs
    boxes = boxes[np.argsort(-confs)]
    # Perform non-maximum suppression
    keep_boxes = []
    while boxes.shape[0]:
        large_overlap = bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
        label_match = boxes[0, -1] == boxes[:, -1]
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        keep_boxes += [boxes[0]]
        boxes = boxes[~invalid]
    boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
    return boxes


class TritonKhoi():
    def __init__(self,
            model = 'fire_smoke',
            input_width = 640,
            input_height = 640,
            mode = "FP32",
            url = '10.70.39.40:8011',
            verbose = False,
            ssl = None,
            root_certificates = None,
            private_key = None,
            certificate_chain = None,
            client_timeout = None):
            
        self.model = model
        self.input_width = input_width
        self.input_height = input_height
        self.mode = mode
        self.client_timeout = client_timeout

        # Create server context
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=verbose,
                ssl=ssl,
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        # Health check
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)
        
        if not self.triton_client.is_model_ready(model):
            print("FAILED : is_model_ready")
            sys.exit(1)
    
    def detect(self, input_image, batch_size = 1):
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('input', [batch_size, 3, self.input_height, self.input_width], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('output'))
        
        input_image_buffer = preprocess(input_image, (self.input_width, self.input_height))
        #
        # input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        
        inputs[0].set_data_from_numpy(input_image_buffer)
        
        results = self.triton_client.infer(model_name=self.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=self.client_timeout)
        
        result = results.as_numpy('output')[0]
        
        res = postprocess(result, origin_w=input_image.shape[1], origin_h= input_image.shape[0], input_shape=(640,640), score_thr=0.5)
        # if res:
        #     print('detections', res[0].confidence)
        return res
    
if __name__ =='__main__':
    import time
    t = TritonKhoi()
    
    img = cv2.imread('./ok.jpg')
    # img = cv2.resize(img, (640, 640))
    # img = cv2.resize(img, (512,512))
    for i in range(1):
        t1 = time.time()
        res = t.detect(img)
        t2 = time.time()
        print(t2 - t1)
    # img_to_draw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img = cv2.resize(img, (640,640))
    print(len(res))

    for r in res:
        cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255,255,0), 1)
    cv2.imwrite('img.jpg', img)