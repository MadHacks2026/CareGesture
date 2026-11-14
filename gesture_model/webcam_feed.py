import cv2
import sys
import csv
import numpy as np
import mediapipe as mp
import urllib
from pathlib import Path

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)
mp_drawing = mp.tasks.vision.drawing_utils

with HandLandmarker.create_from_options(options) as landmarker:
    pass

def run_webcam(capture=False):
    with HandLandmarker.create_from_options(options) as landmarker:
        # Open the default camera
        cam = cv2.VideoCapture(1) # if this causes issues, use cv2.VideoCapture(0)

        # Get the default frame width and height
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        writer = None
        if capture:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('webcam_testing_output.mp4', fourcc, 12.0, (frame_width, frame_height))
            file = open("results.csv", "a")
            writer = csv.writer(file)

        
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # annotate frame using model
            annotated_frame = frame

            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            annotated_frame = cv2.flip(annotated_frame, 1)
            annotated_frame.flags.writeable = False
            results = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, annotated_frame))

            annotated_frame.flags.writeable = True
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
                
            if results.hand_landmarks:
                for num, hand in enumerate(results.hand_landmarks):
                    # to change the colors you use this mp.drawing
                    mp_drawing.draw_landmarks(annotated_frame, hand, mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=4), )
            
            
            if capture:
                out.write(annotated_frame)
            cv2.imshow('Camera', annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and writer objects
        cam.release()
        if capture:
            out.release()
            file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # run python3 webcam.py --capture
    # or python3 webcam.py to not write to file
    if ("--capture" in sys.argv):
        run_webcam(capture=True)
    else:
        run_webcam(capture=False)
        