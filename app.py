import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time
import pygame  # pygame for playing sound
from app3 import main_813

# Initialize pygame mixer for sound outside of the loop
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()

# Load sound file
sound_file = "camera.mp3"  # Replace with your sound file path
sound = pygame.mixer.Sound(sound_file)

# Mediapipe settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)  # Adjusted for better accuracy
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

def detect_finger_frame(landmarks, width, height):
    """
    Detects the frame formed by fingers and returns the coordinates of the rectangle.
    """
    if len(landmarks) != 2:
        return None
    
    # Determine right and left hands based on thumb positions
    right_hand = landmarks[0] if landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[1].landmark[mp_hands.HandLandmark.THUMB_TIP].x else landmarks[1]
    left_hand = landmarks[1] if right_hand == landmarks[0] else landmarks[0]
    
    # Get coordinates of thumb and index finger tips for both hands
    right_thumb_tip = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    right_index_tip = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    left_thumb_tip = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    left_index_tip = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Calculate the vertices of the rectangle (finger frame)
    top_left = (int(left_thumb_tip.x * width), int(left_thumb_tip.y * height))
    top_right = (int(right_index_tip.x * width), int(right_index_tip.y * height))
    bottom_right = (int(right_thumb_tip.x * width), int(right_thumb_tip.y * height))
    bottom_left = (int(left_index_tip.x * width), int(left_index_tip.y * height))

    return [top_left, top_right, bottom_right, bottom_left]

def save_full_image(image):
    """
    Saves the full original screen image rotated 90 degrees counterclockwise.
    """
    # Rotate the image 90 degrees counterclockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Create a timestamped filename using the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_path = f"/run/user/1000/gvfs/google-drive:host=gmail.com,user=nakata.keitai.05/0AFMYJEkYbZEKUk9PVA/FingerFrame/finger_frame_{current_time}.jpg"

    save_path = f"finger_frame_{current_time}.jpg"

    # Save the rotated image
    cv2.imwrite(save_path, rotated_image)
    main_813()
    
    print(f"save at {save_path}")

# Variables to track hand detection
last_finger_detection_time = None
finger_disappeared = False
sound_played = False  # To ensure sound is only played once per detection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Check if finger frame is detected
    if result.multi_hand_landmarks:
        finger_frame = detect_finger_frame(result.multi_hand_landmarks, width, height)
        
        if finger_frame:
            # Draw the detected finger frame on the displayed frame
            cv2.polylines(frame, [np.array(finger_frame)], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Update the last time the finger was detected
            last_finger_detection_time = time.time()
            finger_disappeared = False  # Reset flag since finger is detected

            # Play sound only once per detection
            if not sound_played:
                sound.play()  # Play sound using pygame mixer
                sound_played = True  # Prevent multiple sound plays during continuous detection

    else:
        # If the finger has disappeared
        if last_finger_detection_time and not finger_disappeared:
            # Check if it's been 1 second since the finger disappeared
            if time.time() - last_finger_detection_time >= 1:
                # Save the full image only after the finger disappears for 1 second
                save_full_image(frame)
                finger_disappeared = True  # Set flag to avoid multiple saves
                sound_played = False  # Reset sound flag after disappearance

    # Rotate the frame 90 degrees counterclockwise for preview
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Show the rotated frame with the green finger frame (if detected)
    # cv2.imshow('Finger Frame Detection', rotated_frame)

    #if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC' key
    #     break

# Release resources
cap.release()
cv2.destroyAllWindows()