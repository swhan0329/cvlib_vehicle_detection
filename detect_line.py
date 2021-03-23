import os
import cv2
import argparse
from scipy import ndimage

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, dest="input")

args = parser.parse_args()

input_name = args.input
frame = cv2.imread(input_name, cv2.IMREAD_COLOR)

for i in range(0,10):
    print("rotated",i,"degree")
    #rotation angle in degree
    img = ndimage.rotate(frame, i,reshape=True)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),3)

    low_threshold = 100
    high_threshold = 300
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imshow("edges", edges)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 400  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 500  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    print(len(lines))
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient = -10*(y2-y1)/(x2-x1)
            if gradient > 5:
                pass
            print(x1,x2,y1,y2)
            print("gradient", -10*(y2-y1)/(x2-x1))
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)


    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 1, line_image, 1, 0) 

    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img',1080,720)
    cv2.imshow("line", line_image)
    cv2.imshow("img", lines_edges)
    # cv2.imwrite("output/"+input_name[6:-4]+"_result.jpg",frame)
    cv2.waitKey(0)


    cv2.destroyAllWindows()