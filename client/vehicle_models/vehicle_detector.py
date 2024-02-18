import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

def convert_points(boxes, old_size, new_size):
    points = boxes.astype(float)

    old_h, old_w = old_size
    new_h, new_w = new_size

    r_w, r_h = 1.0 / old_w * new_w, 1.0 / old_h * new_h
    points[:, [0, 2]] *= r_w
    points[:, [1, 3]] *= r_h
    
    return points


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

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class MyLogger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        pass


class TrtModel:
    def __init__(self, engine_path, ctx, dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = MyLogger()
        self.runtime = trt.Runtime(self.logger)
        
        self.engine = self.load_engine(self.runtime, self.engine_path)

        
        self.max_batch_size = 1
        self.cuda_ctx = ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()
        try:
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
            self.context = self.engine.create_execution_context()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

        self.cuda_ctx = ctx
        # self.v8 = v8
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def run(self, x:np.ndarray, batch_size = 1):
        if self.cuda_ctx:
            self.cuda_ctx.push()
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        self.stream.synchronize()
        
        output = [out.host.reshape(batch_size,-1) for out in self.outputs]

        if self.cuda_ctx:
            self.cuda_ctx.pop()
        return output
    
    def detect(self, x:np.ndarray, batch_size=1):
        return None
    

class VehicleDetector(TrtModel):
    def __init__(self, engine_path, ctx, n_classes, input_shape):
        super().__init__(engine_path, ctx)
        self.input_shape = input_shape
        self.n_classes = n_classes
        
    def preprocess(self, im):
        new_shape = self.input_shape
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        # Compute padding [width, height]
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

        dw /= 2 
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im,
                                top,
                                bottom,
                                left,
                                right,
                                cv2.BORDER_CONSTANT,
                                value=(127,127,127))  
        # im = cv2.rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, 0)
        return img, r, (dw,dh)
    
    def postprocess(self, output, r, dwdh, nms_thr, score_thr):
        """Postprocess for output from YOLOv7"""
        
        boxes = output[:, :4]
        scores = output[:, 4:5] * output[:, 5:]
        
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
        if dets is None:
            return []
        else:
            dwdh = np.array(dwdh * 2, dtype=np.float32)

            dets[:,:4] -= dwdh
            dets[:,:4] /= r

            return dets

    def detect(self, x: np.ndarray, nms_thres = 0.65, conf_thres = 0.25):
        img_procesed, r, dwdh = self.preprocess(x)
        
        
        output = self.run(img_procesed)
        
        # inference with batch size = 1
        output = np.reshape(output, (1, -1 , 5 + self.n_classes))[0]
        
        output = self.postprocess(output, r, dwdh, nms_thres, conf_thres)
        
        return output
    
if __name__ == "__main__":
    import cv2
    import numpy as np
    import time
    import pycuda.driver as cuda
    import numpy as np

    device = cuda.Device(0)
    cuda_ctx = device.make_context()
    m = VehicleDetector(engine_path='./weights/VehicleDetector.trt',ctx =cuda_ctx,  n_classes=6,input_shape=(640,640))

    img = cv2.imread('/traffic_light/images/endstream.jpg')
    for i in range(10):
        t1 = time.time()
        pred = m.detect(img)
        print(time.time() - t1)
    for p in pred:
        cv2.rectangle(img, (p[0], p[1]),(p[2], p[3]), (255,0,0), 1)
        
    cv2.imwrite('vehicle_check.jpg', img)