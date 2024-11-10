import cv2
import streamlit as st
import numpy as np

st.title('Streamlit OpenCV Camera Feed')

frame_window = st.image([])

camera = cv2.VideoCapture(2)  

if not camera.isOpened():
    st.error("Unable to access the camera. PLEASE CHECK PERMISSIONS!!")
else:
    while True:
        
        ret, frame = camera.read()      #ret checks if frame was successfully captured 

        if not ret:
            st.error("Failed to grab frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      #convert BGR to standard RGB

        frame_window.image(frame_rgb)                           #release the video

camera.release()                                                #free up resource after using camera
camera.destroyAllWindows()                                      #closes all camera
