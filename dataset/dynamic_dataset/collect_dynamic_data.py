import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Configuration (Adjusted for 50 sequences) ---
DATA_PATH = os.path.join('dataset', 'dynamic_dataset') 
actions = ['hello', 'thank_you', 'yes', 'no', 'food', 'water'] 
NO_SEQUENCES = 50     # ⭐️ 50 video sequences per word
SEQUENCE_LENGTH = 30  # 30 frames in each sequence (~1 second)

# --- Feature Extraction Function ---
def extract_keypoints(results):
    # Flatten Pose (33*4), Face (468*3), Left Hand (21*3), Right Hand (21*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Create the folder structure
for action in actions:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


# --- Data Collection Loop ---
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
quit_flag = False

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        for sequence in range(NO_SEQUENCES):
            
            sequence_dir = os.path.join(DATA_PATH, action, str(sequence))
            # Skip if the sequence is already complete
            if os.path.exists(os.path.join(sequence_dir, f'{SEQUENCE_LENGTH-1}.npy')):
                print(f"Skipping existing sequence: {action} - {sequence}")
                continue

            for frame_num in range(SEQUENCE_LENGTH):

                ret, frame = cap.read()
                if not ret:
                    quit_flag = True; break
                
                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Drawing landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Visualization logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, f'COLLECTING: {action} | Seq: {sequence}/{NO_SEQUENCES} | Frame: {frame_num}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Save Keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(sequence_dir, str(frame_num))
                np.save(npy_path, keypoints)

                cv2.imshow('OpenCV Feed', image)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    quit_flag = True; break
            
            if quit_flag: break
        if quit_flag: break
            
    cap.release()
    cv2.destroyAllWindows()