import sys
# from common.client.detector import TritonClient
import cv2
import os
import time
import numpy as np

import pycuda.autoinit 


def get_model_size(model):
    return (model.input_shape[::-1])

def prepare_img(frame, size_target):
    if frame.shape[0]*frame.shape[1] < size_target[0]*size_target[1]:
        inter = cv2.INTER_LINEAR
    else: inter = cv2.INTER_AREA
    frame_resized = cv2.resize(frame, (size_target), interpolation = inter)
    
    return frame_resized

def convert_points(boxes, old_size, new_size):
    points = boxes[:, 2:].astype(float)

    old_h, old_w = old_size
    new_h, new_w = new_size

    r_w, r_h = 1.0 / old_w * new_w, 1.0 / old_h * new_h
    points[:, [0, 2]] *= r_w
    points[:, [1, 3]] *= r_h
    
    return points.astype(int)

def devide_overlap_obj(detections, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: a dictionary of object overlap >= thresh with key is the selected object with highest confidence along with select objects
    """
    x1 = detections[:, -4]
    y1 = detections[:, -3]
    x2 = detections[:, -2]
    y2 = detections[:, -1]
    scores = detections[:, -5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.arange(scores.shape[0]-1, -1, -1)

    keep = []
    overlapping_objects = {}
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

        overlapping_mask = np.append(False, ovr >= thresh)
        if overlapping_mask.sum():
            overlapping_objects[detections[order[0]].tobytes(
            )] = detections[order[overlapping_mask]]

        overlapping_mask[0] = True
        order = order[~overlapping_mask]

    return detections[keep], overlapping_objects

def crop_img_by_x_y(img, xmin, ymin, xmax, ymax):
    h, w, _ = img.shape
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, w)
    ymax = min(ymax, h)
    return img[ymin:ymax, xmin:xmax, :]

class LP_RECOGNIZER(object):
    def __init__(self, lp_detector, lp_recognizer):
        self.lp_detector = lp_detector
        self.lp_recognizer = lp_recognizer
        self.aug = False
        self.lp_class_names = ['2l', '1l']
        self.char_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
                        'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

    def process_detections_triton(self, detections, nms=.5, return_overlap=False, type_det = 'lp'):
        def return_result(detections=np.empty(0), overlapping_objs={}):
            if return_overlap:
                return detections, overlapping_objs
            return detections

        if len(detections) == 0:
            return return_result()
        
        if type_det == 'lp':
            names = self.lp_class_names
        else:
            names = self.char_class_names
        # convert bbox to points
        detections = np.array([np.concatenate((np.array([names[int(d[5])], str(round(d[4]*100,2))]),np.array(d[:4])), dtype='O') for d in detections])

        detections, overlapping_objs = devide_overlap_obj(detections, nms)
        return return_result(detections, overlapping_objs)

    def detect_lp_from_vc_img_triton(self, vc_img, return_croped_lp=False, return_converted_points=False):
        result = {}
        size_target = get_model_size(self.lp_detector)
        vc_resized = prepare_img(vc_img, size_target)
        
        detections = self.lp_detector.detect(vc_resized, conf_th=0.3)
        result['ll_detections'] = self.process_detections_triton(detections)
        
        if not len(result['ll_detections']):
            return False

        xmin, ymin, xmax, ymax = convert_points(
            result['ll_detections'], vc_resized.shape[:2], (vc_img.shape[:2]))[0]

        if return_converted_points:
            result['converted_points'] = xmin, ymin, xmax, ymax
        if return_croped_lp:
            result['ll_img'] = crop_img_by_x_y(vc_img, xmin, ymin, xmax, ymax)

        return result
    
    def process_detection_trt(self, detections, return_overlap = False, nms = 0.5, type_det ='lp'):
        def return_result(detections=np.empty(0), overlapping_objs={}):
            if return_overlap:
                return detections, overlapping_objs
            return detections
    
        if len(detections[0]) == 0:
            return return_result()
        dets = []
        for i in range(len(detections[0])):
            box = detections[0][i]
            x1, y1, x2, y2 = box
            if type_det == 'lp':
                cls_names = self.lp_class_names[int(detections[2][i])]
            else:
                cls_names = self.char_class_names[int(detections[2][i])]
            conf = detections[1][i]*100
            dets.append([cls_names, conf, x1, y1, x2, y2])
            
        dets = np.array(dets, dtype = 'O')
        detections, overlapping_objs = devide_overlap_obj(dets, nms)
        
        return return_result(detections, overlapping_objs)
    
    def detect_lp_from_vc_img_trt(self, vc_img, return_croped_lp=False, return_converted_points=False):
        result = {}
        size_target = get_model_size(self.lp_detector)
        vc_resized = prepare_img(vc_img, size_target)
        detections = self.lp_detector.detect(vc_resized, conf_th=0.5)
        result['ll_detections'] = self.process_detection_trt(detections)
        
        if not len(result['ll_detections']):
            return False
        
        if len(result['ll_detections']) > 1:
            result['ll_detections'] = result['ll_detections'][result['ll_detections'][:,1] == result['ll_detections'][:,1].max()]
        
        xmin, ymin, xmax, ymax = convert_points(
            result['ll_detections'], vc_resized.shape[:2], (vc_img.shape[:2]))[0]

        if return_converted_points:
            result['converted_points'] = xmin, ymin, xmax, ymax
        if return_croped_lp:
            result['ll_img'] = crop_img_by_x_y(vc_img, xmin, ymin, xmax, ymax)

        return result

    def lp_recognition_trt(self, lp_img, return_overlap = False):
        size_target = get_model_size(self.lp_recognizer)

        lp_img = prepare_img(lp_img, size_target)
        dets = self.lp_recognizer.detect(lp_img, conf_th = 0.3)
        
        if return_overlap:
            detections, overlap = self.process_detection_trt(dets,return_overlap = return_overlap, type_det='c')
            return detections, overlap
        else:
            detections = self.process_detection_trt(dets,return_overlap = return_overlap,type_det='c')
            return detections

    def rotate_and_detect_lp(self, vc_img):
        lp_detection_result = self.detect_lp_from_vc_img_trt(
            vc_img, return_croped_lp=True, return_converted_points=True)
        
        if not lp_detection_result:
            return False
            
        ll_detections = lp_detection_result['ll_detections']
        ll_img = lp_detection_result['ll_img']

        xmin, ymin, xmax, ymax = lp_detection_result['converted_points']

        lp_resized = prepare_img(ll_img, get_model_size(self.lp_recognizer))

        c_detections = self.lp_recognition_trt(lp_resized)
        
        if len(c_detections) < 6:
            return None
        
        h = lp_resized.shape[0]/2
        c_detections = [(int(c[2]), int(c[3]))
                        for c in c_detections if c[3] >= h]  
        if len(c_detections) < 2: 
            lp_detection_result['ll_img_rotated'] = ll_img
            return lp_detection_result
            
        c_detections.sort()

        # find Affine matrix
        pt1 = np.array(c_detections)
        pt2 = pt1.copy()
        pt2[:, 1] = pt1[-1, 1]
        M = cv2.estimateAffinePartial2D(pt1, pt2)
        if M[0] is None:
            lp_detection_result['ll_img_rotated'] = ll_img
            return lp_detection_result
        # calculate new position of lp after applying transformation.
        lp_pos = np.array([[xmin, xmax], [ymin, ymax]])
        new_lp_pos = (M[0][:, :2] @ lp_pos + M[0][:, [2]]).astype(int)
        
        xmin_, xmax_ = new_lp_pos[0]
        ymin_, ymax_ = new_lp_pos[1]
        h, w, _ = vc_img.shape
        # restruct the matrix transformation so it doesn't move lp out of the view.
        if ymax_ > .9*h:
            M[0][1][-1] -= ymax_ - .9*h
        else:
            if ymin_ < .1*h:
                M[0][1][-1] += .1*h - ymin_

        if xmax_ > .9*w:
            M[0][0][-1] -= xmax_ - .9*w
        else:
            if xmin_ < .1*w:
                M[0][0][-1] += .1*w - xmin_
        
        new_lp_pos = (M[0][:, :2] @ lp_pos + M[0][:, [2]]).astype(int)
        
        xmin_, xmax_ = new_lp_pos[0]
        ymin_, ymax_ = new_lp_pos[1]
        # apply affine transformation to vc_img
        vc_img_rotated = cv2.warpAffine(
            vc_img, M[0], (vc_img.shape[1], vc_img.shape[0]))
        
        if not len(vc_img_rotated.copy()[ymin_:ymax_, xmin_: xmax_]):
            lp_detection_result['ll_img_rotated'] = ll_img
            return lp_detection_result
        
        lp_detection_result['ll_img_rotated'] = vc_img_rotated.copy()[ymin_:ymax_, xmin_: xmax_]
        return lp_detection_result
        
    def replace_with_overlap(self, detection: np.ndarray, overlap_container: dict, is_digit=False):
        key = detection.tobytes()
        if key in overlap_container:
            while len(overlap_container[key]):
                r, overlap_container[key] = overlap_container[key][0], overlap_container[key][1:]
                if is_digit:
                    if r[0].isdigit():
                        return r
                else:
                    if r[0].isalpha():
                        return r
        return detection

    def augment_lp_img_from_vehicle_img(self, ll_img, ll_detections, lp_resized_img_shape, vehicle_img_shape):
        model_size = (ll_img.shape[0], ll_img.shape[1])
        ll_converted_points = convert_points(
            ll_detections, lp_resized_img_shape, (vehicle_img_shape))

        # reduce height, increase width
        ll_detections_rh_iw = ll_detections.copy()
        rh_iw_h_offset = (
            (ll_detections[:, -1] - ll_detections[:, -3]) * 0.025).astype(float).round()
        rh_iw_w_offset = (
            (ll_detections[:, -2] - ll_detections[:, -4]) * 0.05).astype(float).round()
        
        # increase height, increase width
        ll_detections_ih_iw = ll_detections.copy()
        ih_iw_h_offset = (
            (ll_detections[:, -1] - ll_detections[:, -3]) * 0.03).astype(float).round()
        ih_iw_w_offset = (
            (ll_detections[:, -2] - ll_detections[:, -4]) * 0.3).astype(float).round()
        
        ll_imgs = []
        # only process the highest confidence lp
        i = 0
        # no padding
        if not self.aug:
            if ll_detections[i][0] =='2l':
                xmin, ymin, xmax, ymax = ll_converted_points[i]
                ll_imgs.append((ll_img, ll_detections[i][0]))
            else:
                ll_imgs.append((cv2.copyMakeBorder(ll_img, 3, 3,
                                            0, 0, cv2.BORDER_CONSTANT, 0), ll_detections[i][0]))
        else:
            xmin, ymin, xmax, ymax = ll_converted_points[i]
            ll_imgs.append((ll_img, ll_detections[i][0]))
            
            rh_iw_offset = int(
                rh_iw_w_offset[i] / model_size[0] * model_size[1] + rh_iw_h_offset[i])
            ll_imgs.append((cv2.copyMakeBorder(ll_img, rh_iw_offset, rh_iw_offset,
                                            0, 0, cv2.BORDER_CONSTANT, 0), ll_detections[i][0]))

            ih_iw_offset = int(
                ih_iw_w_offset[i] / model_size[0] * model_size[1] + ih_iw_h_offset[i])
            ll_imgs.append((cv2.copyMakeBorder(ll_img, ih_iw_offset, ih_iw_offset,
                                            0, 0, cv2.BORDER_CONSTANT, 0), ll_detections[i][0]))
        return ll_imgs

    def post_process_lp_detections(self, lp_detections, layer, vc_type, overlap_container={}, ratio_filter=False):
        # ratio filter
        lp_image = get_model_size(self.lp_recognizer)
       
        
        # re-order
        lp_detections = lp_detections[lp_detections[:, 2].argsort()]
        
        # # detect layer
        two_layers = ((lp_detections[:, -3] >= lp_detections[:, -1].min()).sum() * 2) >= len(lp_detections) * 0.4
        if two_layers or layer =='2l':
            two_layers = True

        # seperate lp numbers
        if two_layers:
            # mask
            upper_mask = lp_detections[:,3] < lp_detections[:, 3].astype(float).mean()
            lower_mask = lp_detections[:, -1] > lp_detections[:, -1].astype(float).mean()
            upper_pos = lp_detections[upper_mask, 3].astype(float).mean()
            lower_pos = lp_detections[lower_mask, 3].astype(float).mean()
            
            confused_idx = np.where(upper_mask == lower_mask)[0]
            ids = np.where((upper_pos - lp_detections[confused_idx, 3]) < (
                lp_detections[confused_idx, -1] - lower_pos))[0]
            upper_mask[confused_idx[ids]] = False

            header = lp_detections[upper_mask]
            digits = lp_detections[~upper_mask]
        else:
            # find biggest gap
            sep_id = (lp_detections[1:, 2] - lp_detections[:-1, -2]).argmax() + 1
            header = lp_detections[:sep_id]
            digits = lp_detections[sep_id:]

        # correction rules
        alphabet_dict = {'A': '4', 'B': '8', 'C': '0', 'D': '0', 'G': '6', 'H': '1',
                        'N': '4', 'S': '5', 'T': '7', 'U': '0', 'X': '3', 'Z': '2'}
        digits_dict = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A',
                    '5': 'S', '6': 'G', '7': 'Z', '8': 'B', '9': 'O'}
        series_1_dict = {'0': 'D', '1': 'T', '2': 'Z', '3': 'S', '4': 'A', '5': 'S',
                        '6': 'G', '7': 'Z', '8': 'B', '9': 'B', 'I': 'T', 'J': 'T', 'O': 'D', 'Q': 'D'}
        series_2_dict = {'G': '6'}

        # correct digits
        if digits.shape[0]:
            digits = np.apply_along_axis(
                self.replace_with_overlap, axis=1, arr=digits, overlap_container=overlap_container, is_digit=True)
        digits[:, 0] = [alphabet_dict[d]
                        if d in alphabet_dict else d for d in digits[:, 0]]

        # seperate province id and series
        if vc_type == "motorbike":
            province_id_mask = header[:, 2] < header[:, 2].astype(float).mean()
            province_id = header[province_id_mask]
            series = header[~province_id_mask]
        else:
            province_id = header[:2]
            series = header[2:]

        # correct province id
        if province_id.shape[0]:
            province_id = np.apply_along_axis(
                self.replace_with_overlap, axis=1, arr=province_id, overlap_container=overlap_container, is_digit=True)
        if len(province_id) > 0 and province_id[0][0] in alphabet_dict:
            province_id[0][0] = alphabet_dict[province_id[0][0]]
    
        if len(province_id) > 1 and province_id[1][0] in alphabet_dict:
            province_id[1][0] = alphabet_dict[province_id[1][0]]
            

        if len(series) > 0:
            series[0] = self.replace_with_overlap(series[0], overlap_container)
            if series[0][0] in series_1_dict:
                series[0][0] = series_1_dict[series[0][0]]

        if len(series) == 2 and series[1][0] in series_2_dict:
            series[1][0] = series_2_dict[series[-1][0]]
        
        lp = ''.join(province_id[:,0]) + ''.join(series[:,0]) + '-' + ''.join(digits[:,0])

        return lp, ( province_id[:, 0], series[:, 0], digits[:, 0]), np.vstack((province_id, series, digits))

    def process_lp(self, ll_imgs_and_layout, vc_type, lp_class_colors = False, return_cropped_lp=False):
        result = {}
        size_target = get_model_size(self.lp_recognizer)
        lp_img, best_lp, best_lp_detections, best_lp_score = None, None, None, 0
        for i in range(len(ll_imgs_and_layout)):
            
            lp_resized = prepare_img(ll_imgs_and_layout[i][0], size_target)
            
            lp_detections, overlap_container = self.lp_recognition_trt(lp_resized, return_overlap = True)
            
            if len(lp_detections) < 7:
                continue
            try:
                lp, _lp, lp_pos_detections = self.post_process_lp_detections(
                    lp_detections, ll_imgs_and_layout[i][1], vc_type, overlap_container)
                
            except Exception as e:
                print('Failed in post process! check it!!!')
            
            lp_score = (len(_lp[0]) == 2) + (len(_lp[0]) > 0 and _lp[0][0].isdigit()) \
                        + (len(_lp[0]) > 1 and _lp[0][1].isdigit())\
                        + (len(_lp[1]) in (1, 2))\
                        + (len(_lp[1]) > 0 and _lp[1][0].isalpha())\
                        + (len(_lp[2]) in (4, 5))\
                        + sum([len(_lp[2]) > i and _lp[2][i].isdigit() for i in (0, 1, 2, 3)])\
                        + lp_pos_detections[:, 1].astype(float).sum()/10

            if lp_score > best_lp_score:
                the_best_i = i
                lp_img, best_lp, best_lp_detections, best_lp_score = lp_resized.copy(), lp, lp_pos_detections, lp_score
        
        if best_lp:
            result['score'] = best_lp_score
            result['img'] = cv2.cvtColor(lp_img.copy(), cv2.COLOR_RGB2BGR)
            result['lp'] = lp
            result['detections'] = best_lp_detections
            # outImg = lp_img.copy()
            # # draw detections in lp image
            # for a in best_lp_detections:
            #     cv2.rectangle(outImg,(a[2],a[3]),(a[4],a[5]),(255,0,255),1)
            #     cv2.putText(outImg, a[0], (a[2], a[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            # cv2.imwrite('LP_img.jpg', outImg)
            return result
        else:
            return None

    def recognize(self, vc_img, vc_type):
        """
        """
        vc_img = cv2.cvtColor(vc_img, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        # 1/ Rotate input vehicle images
        lp_detections = self.rotate_and_detect_lp(vc_img)
        
        if not lp_detections:
            return None
        
        img_rotated = lp_detections['ll_img_rotated']
        
        if not all(img_rotated.shape) or type(img_rotated) != np.ndarray:
            return None
        
        # 2/ Augment
        lp_images = self.augment_lp_img_from_vehicle_img(img_rotated, lp_detections['ll_detections'], get_model_size(self.lp_detector) , vc_img.shape[:2])
        
        # 3/ Process detections, detect line, correct rules
        final_lp = self.process_lp(lp_images, vc_type=vc_type)
    
        return final_lp
    