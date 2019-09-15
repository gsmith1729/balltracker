import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawCircle(x, y):
    cv2.circle(img, (x, y), 25, (0, 0, 255), 5)

def mouseCallback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        drawCircle(x, y)


img = cv2.imread(r"picture.jpg")
cv2.namedWindow('window')
cv2.setMouseCallback('window', mouseCallback)

while(1):
    cv2.imshow('window', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
