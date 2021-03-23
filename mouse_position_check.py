import cv2

point_list = []

def position_check(event, x, y, flags, param):
    # if event == cv2.EVENT_MOUSEMOVE:
    #     # print(f'x: {x}, y: {y}')
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f'x: {x}, y: {y}')
        
        point_list.append((x, y))
        print(x, y)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)
        # img = np.zeros((512,512,3), np.uint8)           

#img = cv2.imread('cctv2_preset2_multi.bmp')
cap = cv2.VideoCapture('input/1.jpg')
ret, img = cap.read()
copy_img = img.copy()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1080, 1920)
cv2.setMouseCallback('image', position_check)

cv2.waitKey(0)
cv2.destroyAllWindows()


for i,v in enumerate(point_list):
    point_list[i] = list(point_list[i])
print(point_list)               