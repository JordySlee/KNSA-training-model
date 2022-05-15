import cv2
import numpy as np
import math

def get_center(contour):
    M = cv2.moments(contour)

    if M['m00'] != 0:
        cx = int(round(M['m10']/M['m00']))
        cy = int(round(M['m01']/M['m00']))
    else:
        cx,cy = 0, 0
    center = (cx, cy)
    return center
for i in range(5):
    i += 1
    img = cv2.imread(f"targets/sample{i}.jpg")


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_mask = cv2.inRange(v, 0, 155)

    contours= cv2.findContours(v_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    outer_circle = None
    biggest_area = 0
    marked = img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_area = area
            outer_circle = contour

    cv2.drawContours(marked, [outer_circle], -1, (0, 255, 0), 3)

    biggest_radius = math.sqrt(biggest_area/math.pi)

    center_v_mask = cv2.inRange(v, 215, 255)
    contours = cv2.findContours(center_v_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    center = get_center(contours[0])

    area = cv2.contourArea(contours[0])
    smallest_radius = math.sqrt(area / math.pi)

    marked = cv2.circle(marked, center, 2, (155,155,0), -1)

    hole_mask = cv2.inRange(h, 0, 30)
    hole_mask = cv2.medianBlur(hole_mask, 11)

    contours= cv2.findContours(hole_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    holes = []
    for contour in contours:
        c = get_center(contour[0])
        holes.append(c)
        marked = cv2.circle(marked, c, 2, (0,0,155), -1)

    remaining_radius = biggest_radius - smallest_radius
    slices = remaining_radius / 9

    scores = []

    for hole in holes:
        dx = hole[0] - center[0]
        dy = hole[1] - center[1]
        dist = math.sqrt(dx*dy + dy*dy)

        dist -= smallest_radius
        if dist < 0:
            scores.append(10)
        else:
            scores.append(9 - int(dist/slices))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for a in range(len(holes)):
        tup = (holes[a][0], holes[a][1])
        marked = cv2.putText(marked, str(scores[a]), tup, font, 1, (0,0,155), 2, cv2.LINE_AA)

    cv2.imshow(f"sample: {i}", marked)
    cv2.waitKey(0)