import cv2
import numpy as np

# capturing the video
capturedVideo = cv2.VideoCapture('leapord.mp4')

# reading two frames
count, frame1 = capturedVideo.read()
count, frame2 = capturedVideo.read()

while capturedVideo.isOpened():

    # find the difference between first fame and second frame
    differenceOfFrames = cv2.absdiff(frame1, frame2)

    # converting frames from BGR to GRAY , Easy to find contours in the gray scale mode
    grayFrame = cv2.cvtColor(differenceOfFrames, cv2.COLOR_BGR2GRAY)

    # blurring(frame name, k_size, sigmaX value )
    blur = cv2.GaussianBlur(grayFrame, (5, 5), 0)

    # _ we don't need first variable(src, threshold_value, max , type)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=3)

    # going to find the contours in the dilated image(img, mode, method )
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # iterate through all contours
    for contour in contours:

        # save all coordinates of saved contours
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 400:
            continue
        elif cv2.contourArea(contour) > 400:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame1, "Status: {}".format('Warning'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame1, "Status: {}".format('Animal Movement Detected'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)

    # drawing the contours in the original frame (frame, contours, contourID, color, thickness)
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    # displaying frame
    cv2.imshow("feed", frame1)

    frame1 = frame2
    count, frame2 = capturedVideo.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
capturedVideo.release()