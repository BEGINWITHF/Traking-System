import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

MODEL_PATH = "hand_landmarker.task"
MIN_DETECTION_CONFIDENCE = 0.15
MIN_PRESENCE_CONFIDENCE = 0.15
MIN_TRACKING_CONFIDENCE = 0.15
NUM_HANDS = 2
STABLE_TIME = 0.8
SMOOTHING_FACTOR = 0.65
POINT_RADIUS = 6
POINT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 255, 255)
LINE_THICKNESS = 4
WINDOW_NAME = "MediaPipe Hands (Press Q to quit)"
FLIP_FRAME = True

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17)
]

cap = cv2.VideoCapture(0)
hand_cache = None
last_detection_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    current_hands = None

    if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
        current_hands = detection_result.hand_landmarks
        hand_cache = current_hands
        last_detection_time = current_time
    elif hand_cache is not None and (current_time - last_detection_time) < STABLE_TIME:
        current_hands = hand_cache

    if current_hands:
        for hand_landmarks in current_hands:
            for lm in hand_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), POINT_RADIUS, POINT_COLOR, -1)

            for start_idx, end_idx in HAND_CONNECTIONS:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                x1 = int(start.x * frame.shape[1])
                y1 = int(start.y * frame.shape[0])
                x2 = int(end.x * frame.shape[1])
                y2 = int(end.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)

    status = "Tracking..." if current_hands else "No hand detected"
    status_color = (0, 255, 0) if current_hands else (0, 0, 255)
    cv2.putText(frame, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    display_frame = cv2.flip(frame, 1) if FLIP_FRAME else frame
    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
