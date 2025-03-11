

import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_V4L2)

# cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # normalize the frame
    frame_new = cv.normalize(
        frame, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1
    )
    # Display the resulting frame
    cv.imshow("frame", frame)
    cv.imshow("framev2", frame)
    # press q to quit
    if cv.waitKey(1) & 0xFF == ord("q"):
        break