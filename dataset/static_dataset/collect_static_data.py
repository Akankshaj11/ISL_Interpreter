# import cv2
# import mediapipe as mp
# import numpy as np
# import csv
# import os

# # --- 1. Configuration and Paths ---
# # Target file path (relative to where you run the script, e.g., ISL_Interpreter/)
# # Adjust this path based on where you run the script. If you run it from the root:
# DATA_FILE = os.path.join('dataset', 'static_dataset', 'alphabets_numbers.csv') 

# # The signs you want to collect data for
# signs_to_collect = [
#     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
#     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
#     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
# ] 
# # NOTE: 'J' and 'Z' often require dynamic motion. You might move them to the dynamic dataset later.

# SAMPLES_PER_SIGN = 50 # Number of samples to collect for each sign

# # Initialize MediaPipe Hands and Drawing Utils
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # --- 2. CSV Header Creation ---
# def create_csv_header(data_path):
#     """Generates the header row for the CSV file."""
#     header = ['label']
#     for i in range(21): # 21 landmarks
#         header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
    
#     # Ensure the directory exists before writing the file
#     os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
#     with open(data_path, mode='w', newline='') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow(header)
#     print(f"Created new CSV file with header at: {data_path}")

# # --- 3. Main Data Collection Loop ---
# if not os.path.exists(DATA_FILE):
#     create_csv_header(DATA_FILE)

# cap = cv2.VideoCapture(0)
# print(f"Starting data collection. Target file: {DATA_FILE}")

# # Use mp_hands.Hands for robust hand detection
# with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    
#     for sign in signs_to_collect:
#         print(f"\n--- Starting collection for SIGN: {sign} ---")
#         samples_collected = 0
        
#         while samples_collected < SAMPLES_PER_SIGN: 
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # 1. Flip the image for natural view
#             frame = cv2.flip(frame, 1)
            
#             # 2. Process the frame with MediaPipe Hands
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for display

#             # 3. Draw Landmarks and Display Instructions
#             hand_detected = False
#             if results.multi_hand_landmarks:
#                 hand_detected = True
#                 # Draw landmarks on the image
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
#             # Display text overlays
#             cv2.putText(image, f'SIGN: {sign}', (10, 30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#             cv2.putText(image, f'Samples collected: {samples_collected}/{SAMPLES_PER_SIGN}', (10, 70), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
#             cv2.putText(image, 'Press "S" to capture | "Q" to Quit.', (10, 470), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
#             # Show the webcam feed
#             cv2.imshow('ISL Static Data Collector', image)

#             # 4. Capture Logic (Key Press)
#             key = cv2.waitKey(10) & 0xFF
            
#             if key == ord('s'):
#                 if hand_detected:
#                     # Extract keypoints from the first detected hand
#                     first_hand = results.multi_hand_landmarks[0]
#                     keypoints_raw = np.array([[res.x, res.y, res.z] for res in first_hand.landmark]).flatten()
                    
#                     # Create the row data: [label, feature1, feature2, ...]
#                     data_row = [sign] + keypoints_raw.tolist()
                    
#                     # Save to CSV in append mode
#                     with open(DATA_FILE, mode='a', newline='') as f:
#                         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                         csv_writer.writerow(data_row)
                    
#                     samples_collected += 1
#                     print(f"Captured Sample {samples_collected}/{SAMPLES_PER_SIGN} for Sign: {sign}")
#                 else:
#                     cv2.putText(image, 'NO HAND DETECTED!', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
#                     cv2.imshow('ISL Static Data Collector', image)
#                     cv2.waitKey(500) # Wait a moment to show the error
                

#             if key == ord('q'):
#                 print("Exiting data collection...")
#                 break
        
#         # If 'q' was pressed inside the inner loop, break the outer loop too
#         if key == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()
# print(f"Data collection process finished. Data saved to: {DATA_FILE}")




# import cv2
# import mediapipe as mp
# import numpy as np
# import csv
# import os
# import pandas as pd # ⭐️ NEW: Import pandas for reading/counting existing data

# # --- 1. Configuration and Paths ---
# # Target file path (relative to where you run the script, e.g., ISL_Interpreter/)
# DATA_FILE = os.path.join('dataset', 'static_dataset', 'alphabets_numbers.csv') 

# # The signs you want to collect data for
# signs_to_collect = [
#     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
#     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
#     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
# ] 

# SAMPLES_PER_SIGN = 50 # Number of samples to collect for each sign

# # Initialize MediaPipe Hands and Drawing Utils
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # --- 2. Utility Function: Check Existing Samples ---
# def count_existing_samples(data_path, signs_list):
#     """Counts how many samples have already been collected for each sign in the CSV."""
#     if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
#         return {sign: 0 for sign in signs_list}
    
#     try:
#         # Load only the 'label' column to count occurrences
#         df = pd.read_csv(data_path, usecols=['label'])
        
#         # Count occurrences of each sign label
#         counts = df['label'].value_counts().to_dict()
        
#         # Ensure every sign in the list has an entry (even if 0)
#         full_counts = {sign: counts.get(sign, 0) for sign in signs_list}
#         return full_counts
#     except pd.errors.EmptyDataError:
#         return {sign: 0 for sign in signs_list}
#     except Exception as e:
#         print(f"Error reading existing CSV data: {e}")
#         return {sign: 0 for sign in signs_list}

# # --- 3. CSV Header Creation ---
# def create_csv_header(data_path):
#     """Generates the header row for the CSV file."""
#     header = ['label']
#     for i in range(21): # 21 landmarks
#         header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
    
#     # Ensure the directory exists before writing the file
#     os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
#     with open(data_path, mode='w', newline='') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow(header)
#     print(f"Created new CSV file with header at: {data_path}")

# # --- 4. Main Data Collection Loop (Revised) ---

# # Check if CSV file exists; if not, create it with the header.
# if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
#     create_csv_header(DATA_FILE)

# # ⭐️ NEW: Get counts of already collected samples
# existing_counts = count_existing_samples(DATA_FILE, signs_to_collect)
# print("\nExisting samples found:")
# for sign, count in existing_counts.items():
#     if count > 0:
#         print(f"  {sign}: {count} samples")
# print("--------------------------")


# cap = cv2.VideoCapture(0)
# print(f"Starting data collection. Target file: {DATA_FILE}")
# quit_flag = False

# # Use mp_hands.Hands for robust hand detection
# with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    
#     for sign in signs_to_collect:
        
#         # ⭐️ CORE CHANGE: Determine how many samples are still needed
#         samples_collected = existing_counts.get(sign, 0)
#         samples_needed = SAMPLES_PER_SIGN - samples_collected
        
#         # Skip this sign if the quota is met
#         if samples_needed <= 0:
#             print(f"--- Quota for SIGN: {sign} is met ({SAMPLES_PER_SIGN}). Skipping. ---")
#             continue
            
#         print(f"\n--- Resuming collection for SIGN: {sign}. Need {samples_needed} more samples. ---")
        
#         # Reset current frame count for the while loop
#         current_sign_samples = samples_collected
        
#         while current_sign_samples < SAMPLES_PER_SIGN: 
#             ret, frame = cap.read()
#             if not ret:
#                 quit_flag = True
#                 break
            
#             # 1. Flip the image for natural view
#             frame = cv2.flip(frame, 1)
            
#             # 2. Process the frame with MediaPipe Hands
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(image)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for display

#             # 3. Draw Landmarks and Display Instructions
#             hand_detected = False
#             if results.multi_hand_landmarks:
#                 hand_detected = True
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
#             # Display text overlays (Use current_sign_samples)
#             cv2.putText(image, f'SIGN: {sign}', (10, 30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#             cv2.putText(image, f'Samples collected: {current_sign_samples}/{SAMPLES_PER_SIGN}', (10, 70), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
#             cv2.putText(image, 'Press "S" to capture | "Q" to Quit.', (10, 470), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
#             cv2.imshow('ISL Static Data Collector', image)

#             # 4. Capture Logic (Key Press)
#             key = cv2.waitKey(10) & 0xFF
            
#             if key == ord('s'):
#                 if hand_detected:
#                     # Extract keypoints from the first detected hand
#                     first_hand = results.multi_hand_landmarks[0]
#                     keypoints_raw = np.array([[res.x, res.y, res.z] for res in first_hand.landmark]).flatten()
                    
#                     # Save to CSV
#                     data_row = [sign] + keypoints_raw.tolist()
#                     with open(DATA_FILE, mode='a', newline='') as f:
#                         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                         csv_writer.writerow(data_row)
                    
#                     current_sign_samples += 1 # Increment the local counter
#                     print(f"Captured Sample {current_sign_samples}/{SAMPLES_PER_SIGN} for Sign: {sign}")
#                 else:
#                     cv2.putText(image, 'NO HAND DETECTED!', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
#                     cv2.imshow('ISL Static Data Collector', image)
#                     cv2.waitKey(500) 
                
#             if key == ord('q'):
#                 print("Exiting data collection...")
#                 quit_flag = True
#                 break
        
#         # Break the outer loop if the quit flag was set
#         if quit_flag:
#             break

# cap.release()
# cv2.destroyAllWindows()
# print(f"Data collection process finished. Data saved to: {DATA_FILE}")




import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pandas as pd 

# --- 1. Configuration and Paths ---
DATA_FILE = os.path.join('dataset', 'static_dataset', 'alphabets_numbers_multilingual.csv') 

# ⭐️ MODIFIED: The signs cover the English alphabet (used for fingerspelling 
# Hindi and Marathi words) and numbers (0-9).
# Note: 'J' and 'Z' are dynamic motions, but we keep them for completeness; 
# you should pose them briefly as a final static posture.
signs_to_collect = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
] 

SAMPLES_PER_SIGN = 50 # Number of samples to collect for each sign

# Initialize MediaPipe Hands and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- 2. Utility Function: Check Existing Samples (Requires Pandas) ---
def count_existing_samples(data_path, signs_list):
    """Counts how many samples have already been collected for each sign in the CSV."""
    if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
        return {sign: 0 for sign in signs_list}
    
    try:
        df = pd.read_csv(data_path, usecols=['label'])
        counts = df['label'].value_counts().to_dict()
        full_counts = {sign: counts.get(sign, 0) for sign in signs_list}
        return full_counts
    except pd.errors.EmptyDataError:
        return {sign: 0 for sign in signs_list}
    except Exception as e:
        print(f"Error reading existing CSV data: {e}")
        return {sign: 0 for sign in signs_list}

# --- 3. CSV Header Creation ---
def create_csv_header(data_path):
    """Generates the header row for the CSV file."""
    header = ['label']
    for i in range(21): # 21 landmarks
        header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
    
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    with open(data_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
    print(f"Created new CSV file with header at: {data_path}")

# --- 4. Main Data Collection Loop ---

if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
    create_csv_header(DATA_FILE)

existing_counts = count_existing_samples(DATA_FILE, signs_to_collect)
print("\nExisting samples found:")
for sign, count in existing_counts.items():
    if count > 0:
        print(f"  {sign}: {count} samples")
print("--------------------------")


cap = cv2.VideoCapture(0)
print(f"Starting data collection. Target file: {DATA_FILE}")
quit_flag = False

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    
    for sign in signs_to_collect:
        
        # Determine how many samples are still needed
        samples_collected = existing_counts.get(sign, 0)
        samples_needed = SAMPLES_PER_SIGN - samples_collected
        
        if samples_needed <= 0:
            print(f"--- Quota for SIGN: {sign} is met ({SAMPLES_PER_SIGN}). Skipping. ---")
            continue
            
        print(f"\n--- Resuming collection for SIGN: {sign}. Need {samples_needed} more samples. ---")
        
        current_sign_samples = samples_collected
        
        while current_sign_samples < SAMPLES_PER_SIGN: 
            ret, frame = cap.read()
            if not ret:
                quit_flag = True
                break
            
            # 1. Flip the image for natural view
            frame = cv2.flip(frame, 1)
            
            # 2. Process the frame with MediaPipe Hands
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

            # 3. Draw Landmarks and Display Instructions
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display text overlays 
            cv2.putText(image, f'SIGN: {sign}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Samples collected: {current_sign_samples}/{SAMPLES_PER_SIGN}', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, 'Press "S" to capture | "Q" to Quit.', (10, 470), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.imshow('ISL Static Data Collector', image)

            # 4. Capture Logic (Key Press)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('s'):
                if hand_detected:
                    # Extract keypoints from the first detected hand
                    first_hand = results.multi_hand_landmarks[0]
                    keypoints_raw = np.array([[res.x, res.y, res.z] for res in first_hand.landmark]).flatten()
                    
                    # Save to CSV
                    data_row = [sign] + keypoints_raw.tolist()
                    with open(DATA_FILE, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(data_row)
                    
                    current_sign_samples += 1 
                    print(f"Captured Sample {current_sign_samples}/{SAMPLES_PER_SIGN} for Sign: {sign}")
                else:
                    cv2.putText(image, 'NO HAND DETECTED!', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.imshow('ISL Static Data Collector', image)
                    cv2.waitKey(500) 
                
            if key == ord('q'):
                print("Exiting data collection...")
                quit_flag = True
                break
        
        if quit_flag:
            break

cap.release()
cv2.destroyAllWindows()
print(f"Data collection process finished. Data saved to: {DATA_FILE}")