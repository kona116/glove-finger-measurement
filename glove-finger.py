import numpy as np
import cv2
import imutils
import math

# circle_radius = 150

def red(img,hand):

    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_red = np.array([0, 80, 80], dtype="uint8")
    high_red = np.array([179, 255, 255], dtype="uint8")
    red_mask = cv2.inRange(hsvim, low_red, high_red)
    red_mask = cv2.erode(red_mask, None, iterations=2)
    red_mask = cv2.dilate(red_mask, None, iterations=2)
    # red = cv2.bitwise_and(img, img, mask=red_mask)

    cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # find the center from the moments 0.000001 is added to the denominator so that divide by
        # zero exception doesn't occur
        center = (int(M["m10"] / (M["m00"] + 0.000001)), int(M["m01"] / (M["m00"] + 0.000001)))
        # print("center_left",center_left)
        # only proceed if the radius meets a minimum size
        if radius > circle_radius:
            # draw the circle and centroid on the frame,
            cv2.circle(img, (int(x), int(y)), int(radius),
                       (0, 0, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            # 5 fingers detected
            if hand == True:
                cv2.putText(img, 'Top of the Glove', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                orientation = True
                # print(str(orientation))
                extPoint(orientation, img, c)

    return

def black(img,hand):
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_black = np.array([0, 0, 0], dtype="uint8")
    high_black = np.array([180, 130, 130], dtype="uint8")
    black_mask = cv2.inRange(hsvim, low_black, high_black)
    black_mask = cv2.erode(black_mask, None, iterations=2)
    black_mask = cv2.dilate(black_mask, None, iterations=2)

    cnts = cv2.findContours(black_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        # covers the object with minimum area.
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # find the center from the moments 0.000001 is added to the denominator so that divide by
        # zero exception doesn't occur
        center = (int(M["m10"] / (M["m00"] + 0.000001)), int(M["m01"] / (M["m00"] + 0.000001)))
        # only proceed if the radius meets a minimum size
        if radius > circle_radius:
            # draw the circle and centroid on the frame,
            cv2.circle(img, (int(x), int(y)), int(radius),
                       (0, 0, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            # 5 fingers detected
            if hand == True:
                cv2.putText(img, 'Bottom of the Glove', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                orientation = False
                # print(str(orientation))
                extPoint(orientation, img, c)

    return

def skinmask(img):

    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([180, 130, 130], dtype="uint8")
    # lower = np.array([0, 48, 80], dtype = "uint8")
    # upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))

    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(blurred, None, iterations=2)
    thresh = cv2.dilate(blurred, None, iterations=2)
    return thresh

def getcnthull(mask_img):
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(contours)

    return contours, hull

def extPoint(orientation, img, c):
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # x,y = int(extLeft[0]),int(extLeft[1])
    # print(x,y)
    # print(extLeft," | ",extRight)
    # Top of the glove
    if orientation == True:
        if extLeft[1] > extRight[1]:
            cv2.putText(img, 'Left hand', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # print("Left hand")
        else:
            cv2.putText(img, 'Right hand', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # print("Right hand")
    else:
        if extLeft[1] < extRight[1]:
            cv2.putText(img, 'Left hand', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # print("Left hand")
        else:
            cv2.putText(img, 'Right hand', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # print("Right hand")

    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is yellow, right-most
    # is green, top-most is blue, and bottom-most is light blue
    cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
    cv2.circle(img, extLeft, 8, (0, 255, 255), -1)
    cv2.circle(img, extRight, 8, (0, 255, 0), -1)
    cv2.circle(img, extTop, 8, (255, 0, 0), -1)
    cv2.circle(img, extBot, 8, (255, 255, 0), -1)

def getdefects(contours):
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)
    return defects

cap = cv2.VideoCapture("http://192.168.1.103:8080/video") # '0' for webcam
# cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, img = cap.read()
    img = cv2.resize(img,(500,500))
    # Flipped the frame so that left hand appears on the left side and right hand appears on the right side
    img = cv2.flip(img,1)
    try:
        circle_radius = 150
        # orientation = ''
        mask_img = skinmask(img)
        contours, hull = getcnthull(mask_img)
        cv2.drawContours(img, [contours], -1, (255,255,0), 2)
        cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)

        defects = getdefects(contours)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                # distance between point and convex hull
                d = (2 * ar) / a

                # cosine theorem
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # angle less than 90 degree, treat as fingers
                # distance between point and convex hull more than 30, treat as fingers
                if angle <= 90 and d > 30:
                    cnt += 1
                    cv2.circle(img, far, 3, [0, 0, 255], -1)
            # cnt+=1
                # if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                #     cnt += 1
                #     cv2.circle(img, far, 4, [0, 0, 255], -1)
            if cnt > 0:
                cnt = cnt+1
            if cnt >= 5:
                hand = True
                red(img,hand)
                black(img,hand)
            else:
                hand = False
                red(img,hand)
                black(img, hand)

            cv2.putText(img, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        cv2.imshow("img", img)

    except:
        pass
    # mask off the last 8bits of the sequence and the ord() of any english keyboard character will not be greater than 255
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
