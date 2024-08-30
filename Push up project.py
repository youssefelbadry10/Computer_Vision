import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Push-Up Counter Variables
counter = 0
stage = None  # None, "down", "up"

# Start Video Capture
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize Frame to Make It Bigger
    frame = cv2.resize(frame, (1200, 1000))  # Adjust dimensions for a larger frame

    # Recolor Image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make Pose Detection
    results = pose.process(image)

    # Recolor Back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract Landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Get Coordinates for Right Side (Adjust for Left Side if needed)
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        # Calculate Angles for Proper Form Detection
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        hip_angle = calculate_angle(shoulder, hip, [hip[0], 1.0])  # Assume a vertical line to calculate hip angle

        # Push-Up Counter Logic
        if elbow_angle > 160 and hip_angle > 160:  # Upper position (arms straight and hips aligned)
            stage = "up"
        if elbow_angle < 90 and stage == "up":  # Lower position (elbows bent)
            stage = "down"
            counter += 1
            print(f"Push-Up Count: {counter}")

        # Visualize Angles
        cv2.putText(image, str(int(elbow_angle)),
                    tuple(np.multiply(elbow, [1600, 1200]).astype(int)),  # Adjust for resized frame
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(hip_angle)),
                    tuple(np.multiply(hip, [1600, 1200]).astype(int)),  # Adjust for resized frame
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    except:
        pass

    # Render Pose Landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    # Render Push-Up Counter
    cv2.rectangle(image, (0, 0), (400, 80), (245, 117, 16), -1)  # Adjusted size
    cv2.putText(image, 'PUSH-UP COUNT', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str(counter), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the Frame
    cv2.imshow('Push-Up Counter', image)

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

