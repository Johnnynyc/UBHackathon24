import cv2
import streamlit as st
import numpy as np
import mediapipe as mp

# Set up Mediapipe hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

# Set up Streamlit
st.title('Welcome to our AnyDraw')
frame_window = st.image([])  # Placeholder for video frame

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera



while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
        # Process the frame
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  # Flip and convert to RGB
    results = hands.process(frame)
    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame in Streamlit
    frame_window.image(frame)  # Update the image in the Streamlit window

        # Limit frame rate to avoid excessive CPU usage
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
