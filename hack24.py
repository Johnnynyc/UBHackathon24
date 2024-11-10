import cv2
import streamlit as st
import numpy as np
import mediapipe as mp

# Set up Mediapipe hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

st.title('Streamlit OpenCV Camera Feed')

frame_window = st.image([])
camera_running = False

def stop_camera(camera):
    camera.release()                            #free up resource after using camera
    cv2.destroyAllWindows()                     #closes all camera
    
if(st.button('Start Game', key="start_button")) and not camera_running:
    camera_running = True
    camera = cv2.VideoCapture(1) 
    
    if(camera.isOpened()):
        if(st.button('Stop Game', key="stop_button")):
            camera_running = False
            stop_camera(camera)
            st.write("Game has been stopped") 

    if not camera.isOpened():
        st.error("Unable to access the camera. PLEASE CHECK PERMISSIONS!!")
    
    else:
        while camera_running:
            
            ret, frame = camera.read()      #ret checks if frame was successfully captured 

            if not ret:
                st.error("Failed to grab frame.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      #convert BGR to standard RGB
            frame_rgb = cv2.flip(frame_rgb, 1)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            frame_window.image(frame_rgb)                           #release the video


