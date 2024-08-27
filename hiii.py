import cv2
import controll as cnt 
from cvzone.HandTrackingModule import HandDetector

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize video capture
video = cv2.VideoCapture(0)

# Desired size for the frame
frame_width = 1000
frame_height = 600

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Resize the frame
    frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Find hands in the frame
    hands, img = detector.findHands(frame)
    
    if hands:
        lmList = hands[0]
        fingerUp = detector.fingersUp(lmList)
        
        print(fingerUp)
        cnt.led(fingerUp)
        
        # Display the finger count
        if fingerUp == [0,0,0,0,0]:
            cv2.putText(frame, 'Finger count:0', (20, frame_height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        elif fingerUp == [0,1,0,0,0]:
            cv2.putText(frame, 'Finger count:1', (20, frame_height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)    
        elif fingerUp == [0,1,1,0,0]:
            cv2.putText(frame, 'Finger count:2', (20, frame_height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        elif fingerUp == [0,1,1,1,0]:
            cv2.putText(frame, 'Finger count:3', (20, frame_height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        elif fingerUp == [0,1,1,1,1]:
            cv2.putText(frame, 'Finger count:4', (20, frame_height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        elif fingerUp == [1,1,1,1,1]:
            cv2.putText(frame, 'Finger count:5', (20, frame_height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA) 

    # Display the frame
    cv2.imshow("frame", frame)
    
    # Exit loop on pressing 'k'
    k = cv2.waitKey(1)
    if k == ord("k"):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
