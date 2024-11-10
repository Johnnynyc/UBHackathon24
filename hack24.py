import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import math

# Initialize mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils  # For drawing hand landmarks on the video frame

# Initialize drawing variables
drawing = False  # Tracks if drawing is active
draw_color = (0, 255, 0)  # Color for the drawing
draw_radius = 5  # Radius for the drawing circles
last_x, last_y = None, None  # Last point coordinates for smooth drawing

# Function to get the tip of the index finger
def get_index_fingertip_position(hand_landmarks, image_shape):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    h, w, _ = image_shape
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)
    return x, y

# Function to get the tip of the thumb finger
def get_thumb_fingertip_position(hand_landmarks, image_shape):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    h, w, _ = image_shape
    x = int(thumb_tip.x * w)
    y = int(thumb_tip.y * h)
    return x, y

# Streamlit setup
st.set_page_config(layout="wide")
st.title("Hand Drawing with Webcam Feed")

# Streamlit UI Controls
clear_canvas = st.button("Clear Canvas")

col1, col2 = st.columns([1, 1])  # Create two columns for split layout

# Start webcam feed and hand tracking
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Unable to access the webcam. Please check your camera settings.")

# Create placeholders for webcam feed and canvas
frame_placeholder = col1.empty()
canvas_placeholder = col2.empty()

# Initialize canvas for drawing (in col2)
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White canvas for drawing

# Main loop for processing the webcam feed
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
   
    # Flip and process image
    image = cv2.flip(image, 1)  # Flip the image horizontally for mirror effect
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB for mediapipe
    results = hands.process(image_rgb)  # Get hand landmarks

    # Display the webcam feed in RGB format in col1
    frame_placeholder.image(image_rgb, use_container_width=True)

    # Process drawing on the canvas in col2
    if results.multi_hand_landmarks:  # If hands are detected
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw landmarks

            # Get the index fingertip position
            x, y = get_index_fingertip_position(hand_landmarks, image.shape)
            thumb_x, thumb_y = get_thumb_fingertip_position(hand_landmarks, image.shape)

            # Calculate distance between index and thumb fingertips
            distance = math.sqrt((x - thumb_x) ** 2 + (y - thumb_y) ** 2)

            # Activate drawing mode when fingers are close together
            if distance < 30:  # Adjust threshold distance as needed
                drawing_mode = True
            else:
                drawing_mode = False

            # Draw if drawing mode is active
            if drawing_mode:
                if last_x is not None and last_y is not None:
                    # Draw a line from the last point to the current point for smooth drawing
                    cv2.line(canvas, (last_x, last_y), (x, y), draw_color, draw_radius)
                # Update last point coordinates
                last_x, last_y = x, y
            else:
                last_x, last_y = None, None  # Reset last point when not drawing

    # Convert the canvas to RGB before displaying in col2
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas_placeholder.image(canvas_rgb, caption="Drawing Canvas", use_container_width=True)

    # Check if Clear Canvas button is clicked
    if clear_canvas:
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Reset the canvas

cap.release()
cv2.destroyAllWindows()
