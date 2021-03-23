import cv2
import numpy as np
cap = cv2.VideoCapture('input/test.mp4')

if cap.isOpened():
    print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))

idx = 0
while True:
    ret, frame = cap.read()
    if ret:
        if idx == 0:
            save_frame = frame.copy()
            cv2.imshow('copy',save_frame)
        print("frame number",idx)
        cv2.imshow('input',frame)

        grayA = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(save_frame,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kpA, desA = sift.detectAndCompute(grayA, None)
        kpB, desB = sift.detectAndCompute(grayB, None)
        
        bf = cv2.BFMatcher()
        matches = bf.match(desA, desB)
        
        sorted_matches = sorted(matches, key = lambda x : x.distance)
        res = cv2.drawMatches(frame, kpA, save_frame, kpB, sorted_matches[:30], None, flags = 2)
        
        src = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape((-1, 1, 2))
        dst = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape((-1, 1, 2))
        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        print(src,"\n", dst)
        

        print("\nhomography matrix\n",H)
        cv2.waitKey(0)
        idx += 1


    else:
        break
cap.release()
cv2.destroyAllWindows()