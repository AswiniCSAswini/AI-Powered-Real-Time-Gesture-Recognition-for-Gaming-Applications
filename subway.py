import cv2
import mediapipe as mp
import time
from pynput.keyboard import Controller, Key

# Initialize MediaPipe Hands and keyboard controller
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
keyboard = Controller()

# Initialize variables
last_action = None
last_action_time = 0
gesture_cooldown = 0.5  # Cooldown time in seconds between gestures

# Variables to track index finger position
previous_index_position = None

def detect_swipe(previous_pos, current_pos):
    """
    Detects swipe gestures based on the movement of the index finger tip.
    Args:
        previous_pos: Tuple (x, y) of the previous position of the index finger tip.
        current_pos: Tuple (x, y) of the current position of the index finger tip.
    Returns:
        Gesture direction as "up", "down", "left", or "right", or None if no significant swipe is detected.
    """
    if previous_pos is None:
        return None

    dx = current_pos[0] - previous_pos[0]  # Change in x-axis
    dy = current_pos[1] - previous_pos[1]  # Change in y-axis

    # Detect significant swipes
    if abs(dy) > 0.1 and abs(dy) > abs(dx):  # Vertical swipe
        return "up" if dy < 0 else "down"
    elif abs(dx) > 0.1 and abs(dx) > abs(dy):  # Horizontal swipe
        return "left" if dx < 0 else "right"

    return None

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the current position of the index finger tip
            landmarks = [lm for lm in hand_landmarks.landmark]
            index_tip = landmarks[8]
            current_index_position = (index_tip.x, index_tip.y)

            # Detect swipe gesture
            gesture = detect_swipe(previous_index_position, current_index_position)
            current_time = time.time()

            if gesture and (gesture != last_action or (current_time - last_action_time > gesture_cooldown)):
                last_action = gesture
                last_action_time = current_time

                # Print the gesture
                print(f"Gesture Detected: {gesture}")
                
                # Display the gesture on the video feed
                cv2.putText(
                    frame, 
                    f"Gesture: {gesture}", 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2, 
                    cv2.LINE_AA
                )

                # Perform action based on gesture
                if gesture == "up":
                    keyboard.press(Key.up)
                    keyboard.release(Key.up)
                elif gesture == "down":
                    keyboard.press(Key.down)
                    keyboard.release(Key.down)
                elif gesture == "left":
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                elif gesture == "right":
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)

            # Update the previous index finger position
            previous_index_position = current_index_position

    # Show the frame
    cv2.imshow('Subway Surfers Hand Control', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
