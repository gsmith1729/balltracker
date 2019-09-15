import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawCircle(x, y, img):
    cv2.circle(img, (x, y), 25, (0, 0, 255), 3)

def drawTrailCircle(x, y, img):
    cv2.circle(img, (x, y), 25, (0, 255, 100), 2)

def drawPolynomial(a, b, c, img, color):
    height, width = img.shape[:2]
    for x in range (0, width):
        cv2.circle(img, (x, int(a*x*x+b*x+c)), 1, color, 2)

def mouseCallback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        stumppos[0]=x
        stumppos[1]=y
        print(stumppos)
stumppos=[0,0]
cv2.namedWindow('Video')
cv2.setMouseCallback('Video',mouseCallback)
cap = cv2.VideoCapture('normalSpeedMiss.mp4')
movementFilter = cv2.createBackgroundSubtractorKNN(dist2Threshold=3000, detectShadows=False)
smoother = lambda x: np.ones((x,x),np.float32)/(9*x)

points = []
frameNumber = 0

while(1):
    print("Frame {}".format(frameNumber))
    
    # Take each frame
    _, original = cap.read()
    (width, height) = original.shape[:2]
    original = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    original = cv2.resize(original, (int(width/2),int(height/2)))
    #original = cv2.filter2D(original,-1,smoother(10))

    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # define range of HSV cutoff
    lower_cutoff = np.array([20,0,150])
    upper_cutoff = np.array([50,255,255])

    # Mask the HSV image to aim to get ball related colors
    mask = cv2.inRange(hsv, lower_cutoff, upper_cutoff)
    res = cv2.bitwise_and(original, original, mask=mask)

    # Background subtraction/motion detection mask
    fgmask = movementFilter.apply(res)

    # Smooth the mask
    # Smoothing between frames?
    smoothed = cv2.filter2D(fgmask,-1,smoother(20))
 
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
    #display=mask


    # Trail
    for point in points:
        drawTrailCircle(point[0],point[1],display)
    # Ball highlight

    if thresh.mean() > 0.05 and thresh.mean() < 255:
        points.append([cX, cY])
        drawCircle(cX,cY,display)

    if len(points) > 2:
        x = []
        y = []
        for i in range(1, min(8,len(points))):
            x.append(points[i][0])
            y.append(points[i][1])

        a, b, c = np.polyfit(np.array(x),np.array(y),2)
        dt = 1/(cap.get(cv2.CAP_PROP_FPS)) # s
        dx = (points[len(points) - 1][0] - points[1][0])/((len(points) - 2))
        dx_dt = dx/dt
        dy = points[len(points) - 1][1] - points[len(points)-2][1]
        dy_dt = dy/dt
        g = 2*a*dx_dt*dx_dt
        # a :: pixel^-1
        # dx_dt :: pixel s^-1
        # g :: pixel s^-2
        sf = 9.8 / g
        # sf :: meter / pixel
        print("Y speed: {} m/s".format(-dy_dt*sf))
        print("X speed: {} m/s".format(dx_dt*sf))
        
        if len(points) > 7:
            drawPolynomial(a, b, c, display, (255, 0, 255))
        else:
            drawPolynomial(a, b, c, display, (255, 0, 0))
        sx=stumppos[0]
        if a*sx*sx+b*sx+c>stumppos[1]:
            print("Wickets Hitting")
        drawCircle(stumppos[0],stumppos[1],display)

        

    # Show the result
    cv2.imshow('Video',display)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    if k == 32:
        cv2.waitKey(0)

    frameNumber = frameNumber + 1

#import csv
#with open('points.csv', 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    for point in points:
#        spamwriter.writerow(point)

cap.release()
cv2.destroyAllWindows()


