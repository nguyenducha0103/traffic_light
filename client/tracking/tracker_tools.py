import numpy as np

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


def normalizing_coord(coord, shape):
    x1,y1,x2,y2 = [int(i) for i in coord]
    x1,y1,x2,y2 = x1 if x1 >= 0 else 0,\
            y1 if y1 >= 0 else 0,\
            x2 if x2 <= shape[1] else shape[1],\
            y2 if y2 <= shape[0] else shape[0]
    
    normalized_coord = [x1,y1,x2,y2]            
    return normalized_coord

def tracker_inference(trackers,detection_results,detect_frame,is_reset=False):
    # 'tracking_results' is a list of each vehicle's tracking info:
    # tracking_results[id] = list[list[x1, y1, x2, y2], int(ID), int(class ID)]
    # tracking_results = trackers.update(np.array(detection_results),is_reset=is_reset)
    track_bboxes = []
    track_scores = []
    track_classes = []
    for detection_result in detection_results:
        track_bboxes.append([detection_result[0], detection_result[1], detection_result[2], detection_result[3]])
        track_scores.append(detection_result[5])
        track_classes.append(detection_result[4])
    
    tracking_results = trackers.update(np.array(track_bboxes),np.array(track_scores),np.array(track_classes))
    # print(tracking_results)
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@',len(detection_results),len(tracking_results))
    # print(tracking_results)
    processed_tracking_results = []
    # for x1,y1,x2,y2, vehicle_ID, classID, conf in tracking_results:
    # print('tracking_results:',tracking_results)
    for x1,y1,x2,y2, vehicle_ID, classID in tracking_results:
        # print("{},{},{},{}, vehicle_ID-{}, classID-{}, conf-{}".format(x1,y1,x2,y2, vehicle_ID, classID, conf))
        try:
            bbox = normalizing_coord([int(x1),int(y1),int(x2),int(y2)],detect_frame.shape)
            vehicle_ID = int(vehicle_ID%10000)
            classID = int(classID)
            # classID = int(0)
            # conf = float(conf)
            conf = float(1)
            # bbox = [detect_roi[i%2] + round(bbox[i]/processing_scale_ratio) for i in range(len(bbox))]
            processed_tracking_results.append([bbox, vehicle_ID, classID, conf])
        except:
            pass

    return processed_tracking_results