import os
import cv2
import argparse
import cvlib as cv
from cvlib.object_detection import draw_bbox
from shapely.geometry import Polygon


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

import numpy as np
import imutils

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, dest="input")

args = parser.parse_args()

input_name = args.input

rec_boxes = [[16, 561, 23, 582,122, 544, 112, 528], [216, 491, 226, 504,306, 474, 290, 462], [355, 440, 372, 450,456, 421, 429, 408], [475, 397, 492, 405,546, 384, 526, 376], [570, 364, 589, 371,635, 354, 609, 346], [655, 335, 671, 340,696, 329, 680, 323], [718, 313, 730, 316,755, 307, 741, 302], [774, 292, 785, 297,808, 290, 792, 285], [825, 276, 835, 278,848, 273, 839, 270], [865, 262, 874, 265,888, 261, 878, 257], [901, 249, 910, 251,918, 247, 911, 244]]
space_list = [1]*len(rec_boxes)
print("initialize free space list",space_list)
frame = cv2.imread(input_name, cv2.IMREAD_COLOR)

bbox, label, conf = cv.detect_common_objects(frame, model='yolov4', enable_gpu=True)

rec_num = 0
for rec_box in rec_boxes:
    print("rec num",rec_num)
    for abbox in bbox:
        # print([[rec_box[0],rec_box[1]],[rec_box[2],rec_box[3]],[rec_box[4],rec_box[5]],[rec_box[6],rec_box[7]]])
        # print([[abbox[0],abbox[1]],[abbox[2],abbox[1]],[abbox[2],abbox[3]],[abbox[0],abbox[3]]])
        iou = calculate_iou([[rec_box[0],rec_box[1]],[rec_box[2],rec_box[3]],[rec_box[4],rec_box[5]],[rec_box[6],rec_box[7]]],[[abbox[0],abbox[1]],[abbox[2],abbox[1]],[abbox[2],abbox[3]],[abbox[0],abbox[3]]])*100
        if iou != 0.0:
            print("iou",iou,"%")
        if iou > 10.0:
            space_list[rec_num] = 0
    rec_num += 1

print("current free space list",space_list)
for idx, space in enumerate(space_list):
    pts = np.array([[rec_boxes[idx][0],rec_boxes[idx][1]],[rec_boxes[idx][2],rec_boxes[idx][3]],[rec_boxes[idx][4],rec_boxes[idx][5]],[rec_boxes[idx][6],rec_boxes[idx][7]]],np.int32)
    pts = pts.reshape((-1, 1, 2))
    if space:
        frame = cv2.polylines(frame, [pts], True, (255,0,0),3)
    else:
        frame = cv2.polylines(frame, [pts], True, (0,0,255),3)
    
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
