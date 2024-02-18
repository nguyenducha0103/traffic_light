import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import sys

import cv2
import numpy as np


class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = int(classID)
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height
    
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)
        
    def width(self):
        return self.x2 - self.x1
    
    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))
    
    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))
    
    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)
    

import cv2
import numpy as np

  
def preprocess(img, input_shape, letter_box=True):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = np.float32(img)
    img = np.ascontiguousarray(img)
    img /= 255.0
    
    return img

def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.
    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep

def convert_points(boxes, old_size, new_size):
    points = boxes.astype(float)

    old_h, old_w = old_size
    new_h, new_w = new_size

    r_w, r_h = 1.0 / old_w * new_w, 1.0 / old_h * new_h
    points[:, [0, 2]] *= r_w
    points[:, [1, 3]] *= r_h
    
    return points.astype(int)

def postprocess(predictions, img_w, img_h, input_shape, nms_thr = 0.45, score_thr = 0.45):
    """Postprocess for output from YOLOv7"""
    # print(predictions.shape)
    predictions = np.reshape(predictions, (1, -1, int(5+2)))[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    
    # score
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
    # predictions = predictions.reshape((7,-1))
    # print(predictions[0].astype(np.float32))
    # dets = _nms_boxes(predictions, nms_threshold=nms_thr)
    if dets is None:
        return []
    # print(len(dets))
    boxes = convert_points(dets[:,:4], input_shape, (img_h, img_w))
    scores = dets[:,4]
    labels = dets[:,5]

    detected_objects = []
    for box, score, label in zip(boxes, scores, labels):
        detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], img_w, img_h))
    return detected_objects

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    
    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

class TritonKhoi():
    def __init__(self,
            model = 'khoichay',
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
        inputs.append(grpcclient.InferInput('images', [batch_size, 3, self.input_height, self.input_width], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('output0'))
        
        input_image_buffer = preprocess(input_image, (self.input_width, self.input_height), letter_box=True)
        #
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        
        inputs[0].set_data_from_numpy(input_image_buffer)
        
        results = self.triton_client.infer(model_name=self.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=self.client_timeout)
        
        result = results.as_numpy('output0')
        # result = torch.tensor(result)
        # print(result.shape)
        # res = non_max_suppression(result)
        # print(res)
        res = postprocess(result[0], img_w=input_image.shape[1], img_h= input_image.shape[0], input_shape=(640,640), score_thr=0.5)
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
    
    for bb in res:
        # if int(bb[5])==1:
        #     cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,0), 1)
        # else:
        #     cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,255), 1)
        print((int(bb.x2), int(bb.y2)), bb.confidence)
        cv2.rectangle(img, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), (255,0,0), 2)
    cv2.imwrite('test_drawed.jpg', img)