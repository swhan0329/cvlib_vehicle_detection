import os
import cv2
import argparse
import cvlib as cv
from cvlib.object_detection import draw_bbox

import numpy as np
import imutils

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, dest="input")

args = parser.parse_args()

input_name = args.input
print(input_name)
frame = cv2.imread(input_name, cv2.IMREAD_COLOR)

bbox, label, conf = cv.detect_common_objects(frame, model='yolov4', enable_gpu=True)

# for i, cla in enumerate(label):
#     if cla == 'car' or cla == 'bus' or cla == 'truck' or cla == 'train':
#         blur_h = int((bbox[i][3]-bbox[i][1])*1/3)
#         if bbox[i][0] > 0 and bbox[i][1] > 0:
#             blur_area = frame[bbox[i][3]-blur_h:bbox[i][3]+int(blur_h/3),bbox[i][0]:bbox[i][2]]
#             blur_img = cv2.blur(blur_area, (7,7))
#             frame[bbox[i][3]-blur_h:bbox[i][3]+int(blur_h/3),bbox[i][0]:bbox[i][2]] = blur_img
# draw bounding box over detected objects
frame = draw_bbox(frame, bbox, label, conf, write_conf=True)

# display output
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame',720,360)
cv2.imshow("frame", frame)
# cv2.imwrite("output/"+input_name[6:-4]+"_result.jpg",frame)
cv2.waitKey(0)


cv2.destroyAllWindows()