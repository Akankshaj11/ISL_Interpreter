import cv2
import mediapipe as mp
import numpy as np
from utils.inference_cnn import predict_static
from utils.overlays import draw_overlay_text
from utils.hotkeys import handle_keypress

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
running = False
recognized_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    running, action = handle_keypress(key, running)

    # Perform recognition when running
    if running:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert landmarks to array for model input
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks)

                recognized_text = predict_static(landmarks)

        draw_overlay_text(frame, recognized_text, running)

    cv2.imshow("ISL Interpreter - Real Time", frame)

    if action == "quit":
        break

cap.release()
cv2.destroyAllWindows()
