import cv2
import time

n_devices = 3

device_id = 0
while n_devices:
    vc = cv2.VideoCapture(0)
    
    ret, frame = vc.read()
    if ret:
        print(device_id)
        print(frame.shape)
        cv2.imshow("show",frame)

        cv2.waitKey(1)
        device_id += 1
        # n_devices -= 1

    time.sleep(1)

cv2.destroyAllWindows()