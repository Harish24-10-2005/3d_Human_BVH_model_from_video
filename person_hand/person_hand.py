import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.drawing_utils = mp_drawing

    def __call__(self, image):
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        hand_boxes = []
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box of the hand
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                # Convert normalized coordinates to pixel values
                h, w, _ = image.shape
                x_min, x_max = int(x_min * w), int(x_max * w)
                y_min, y_max = int(y_min * h), int(y_max * h)
                hand_boxes.append([x_min, y_min, x_max, y_max])

                # Convert landmarks to a list of coordinates
                landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]
                landmarks_list.append(landmarks)

                # Draw hand landmarks
                self.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return hand_boxes, landmarks_list