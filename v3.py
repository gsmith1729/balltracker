import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawCircle(x, y, img):
    cv2.circle(img, (x, y), 25, (0, 0, 255), 2)

def mouseCallback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        drawCircle(x, y)

cap = cv2.VideoCapture('slomostumphit.mp4')
movementFilter = cv2.createBackgroundSubtractorKNN(dist2Threshold=3000, detectShadows=False)
smoother = lambda x: np.ones((x,x),np.float32)/(9*x)

points = []

while(1):
    # Take each frame
    _, original = cap.read()
    (height, width) = original.shape[:2]
    original = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)

    #original = cv2.filter2D(original,-1,smoother(10))

    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # define range of HSV cutoff
    lower_cutoff = np.array([20,0,100])
    upper_cutoff = np.array([50,255,255])

    # Mask the HSV image to aim to get ball related colors
    mask = cv2.inRange(hsv, lower_cutoff, upper_cutoff)
    res = cv2.bitwise_and(original, original, mask=mask)

    # Background subtraction/motion detection mask
    fgmask = movementFilter.apply(res)

    # Smooth the mask
    # Smoothing between frames?
    smoothed = cv2.filter2D(fgmask,-1,smoother(40))
 
    # Apply a threshold to the smoothed mask
    ret,thresh = cv2.threshold(smoothed,240,255,0)
    
    # Find the weighted average / moment
    M = cv2.moments(thresh)

    # Calculate x,y coordinate of center
    cX = -1
    cY = -1
    if M["m00"] != 0:
        cX = int(M["m10"] / (M["m00"]))
        cY = int(M["m01"] / (M["m00"]))

    # Choose a layer to display
    #display = original
    display = cv2.add(original, cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
    #display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Ball highlight
    if thresh.mean() > 0.1 and thresh.mean() < 255:
        points.append([cX, cY])
        drawCircle(cX,cY,display)

    # Trail
    for point in points:
        drawCircle(point[0],point[1],display)

    # Show the result
    cv2.imshow('Video',display)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

