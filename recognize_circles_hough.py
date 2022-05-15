import cv2 as cv
import numpy as np


img = cv.imread('targets/sample4.jpg')
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2*(y1-y2)**2


img_gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


circles = cv.HoughCircles(img_gr, cv.HOUGH_GRADIENT, 1.2, 20, param1=500, param2=100, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    chosen = None
    for i in circles[0, :]:
        if chosen is None: chosen = i
        if prevCircle is not None:
            if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                chosen = i
    cv.circle(img, (chosen[0], chosen[1]), 1, (0, 100,100), 3)
    cv.circle(img, (chosen[0], chosen[1]), chosen[2], (255, 0 , 255), 3)

    prevCircle = chosen
cv.imshow("circles", img)
# cv.imshow("img", img_gr)
cv.waitKey(0)
cv.destroyAllWindows()