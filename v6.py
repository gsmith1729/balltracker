import cv2
import numpy as np

VIDEO_FILENAME = "slomostumphit.mp4"
ROTATION_AMOUNT = None  # cv2.ROTATE_90_CLOCKWISE

def drawPolynomial(a, b, c, img, color):
    height, width = img.shape[:2]
    for x in range(0, width):
        cv2.circle(img, (x, int(a * x * x + b * x + c)), 1, color, 2)


def mouseCallback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global stumpPosition
        stumpPosition = (x, y)
        print("Stump position set to {}".format(stumpPosition))


stumpPosition = None
windowTitle = "Video"
cv2.namedWindow(windowTitle, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowTitle, 1800, 3000)
cv2.setMouseCallback(windowTitle, mouseCallback)

movementFilter = cv2.createBackgroundSubtractorKNN(dist2Threshold=3000, detectShadows=False)
smoother = lambda x: np.ones((x, x), np.float32) / (6 * x)

points = []
frameNumber = 0

cap = cv2.VideoCapture(VIDEO_FILENAME)
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # :: pixel
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # :: pixel
fps = cap.get(cv2.CAP_PROP_FPS) # :: s^-1
dt = 1 / fps  # :: s
scaleFactor = np.sqrt(360 * 640 / (width * height))

while 1:
    frameNumber = frameNumber + 1
    if frameNumber > totalFrames:
        print("End of video")
        break
    print("Frame {}/{}".format(frameNumber, totalFrames))

    # Take each frame
    _, original = cap.read()

    if ROTATION_AMOUNT is not None:
        original = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    
    original = cv2.resize(
        original, (int(scaleFactor * width), int(scaleFactor * height))
    )
    # original = cv2.filter2D(original,-1,smoother(10))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # define range of HSV cutoff
    lower_cutoff = np.array([20, 0, 100])
    upper_cutoff = np.array([50, 255, 255])

    # Mask the HSV image to aim to get ball related colors
    mask = cv2.inRange(hsv, lower_cutoff, upper_cutoff)
    res = cv2.bitwise_and(original, original, mask=mask)

    # Background subtraction/motion detection mask
    fgmask = movementFilter.apply(res)

    # Smooth the mask
    # Smoothing between frames?
    smoothed = cv2.filter2D(fgmask, -1, smoother(15))

    # Apply a threshold to the smoothed mask
    ret, thresh = cv2.threshold(smoothed, 240, 255, 0)

    # Find the weighted average / moment
    M = cv2.moments(thresh)

    # Calculate x,y coordinate of ball position
    ballPosition = None
    if M["m00"] != 0:
        ballPosition = (int(M["m10"] / (M["m00"])), int(M["m01"] / (M["m00"])))

    # Choose a layer to display
    # display = original
    display = cv2.add(original, cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
    # display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # display=mask

    # Trail
    for p in points:
        cv2.circle(display, (p[0], p[1]), 25, (0, 255, 100), 2)

    # Ball highlight
    if ballPosition is not None and thresh.mean() > 0.05 and thresh.mean() < 255:
        points.append(ballPosition)
        cv2.circle(display, ballPosition, 25, (0, 0, 255), 3)

    if len(points) > 1:
        x = list(map(lambda p: p[0], points[:15]))
        y = list(map(lambda p: p[1], points[:15]))

        a, b, c = np.polyfit(x, y, 2)
        # a :: pixel^-1, b :: unitless, c :: pixel
        dx = (points[len(points) - 1][0] - points[0][0]) / ((len(points) - 1)) # :: pixel
        dx_dt = dx * fps # :: pixel s^-1
        dy = points[len(points) - 1][1] - points[len(points) - 2][1] # :: pixel
        dy_dt = dy * fps # :: pixel s^-1
        g = 2 * a * dx_dt * dx_dt # :: pixel s^-2
        sf = 9.8 / g # :: m pixel^-1
        print("Y speed: {} m/s".format(-dy_dt * sf))
        print("X speed: {} m/s".format(dx_dt * sf))

        px = points[len(points) - 1][0]
        if abs(a * px * px + b * px + c - points[len(points) - 1][1]) > 10:
            points = []

        if len(points) > 15:
            drawPolynomial(a, b, c, display, (255, 0, 255))
        else:
            drawPolynomial(a, b, c, display, (255, 0, 0))

        if stumpPosition is not None:
            sx = stumpPosition[0]
            if a * sx * sx + b * sx + c > stumpPosition[1]:
                print("Wickets hitting")
            else:
                print("Wickets missing")

    # Show the stump position
    if stumpPosition is not None:
        cv2.circle(display, (stumpPosition[0], stumpPosition[1]), 25, (0, 255, 255), 3)

    # Show the result
    cv2.imshow(windowTitle, display)

    # Listen for key presses
    k = cv2.waitKey(15) & 0xFF
    # Quit if the 'Esc' or 'q' key is hit
    if k == 27 or k == ord("q"):
        break
    # Pause if the spacebar is hit
    if k == 32:
        cv2.waitKey(0)

# import csv
# with open('points.csv', 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=',',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    for point in points:
#        spamwriter.writerow(point)

cap.release()
cv2.destroyAllWindows()
