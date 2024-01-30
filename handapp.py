pip install opencv-python
pip install streamlit
pip install mediapipe
pip install numpy
pip install tabulate


import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tabulate import tabulate

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

joint_list = [[4,3,2, "Thumb IP jt"],[3,2,1, "Thump MC jt"],[8, 7, 6, "Index finger PIP jt"], [7, 6, 5, "Index finger DIP jt"], [6, 5, 0, "Index finge MCP jt"],
              [12, 11, 10, "Middle Finger DIP jt"], [11, 10, 9, "Middle finger PIP jt"], [10, 9, 0, "Middle finge MCP jt"],
              [16, 15, 14, "Ring Finger DIP jt"], [15, 14, 13, "Ring finger PIP jt"], [14, 13, 0, "Ring finge MCP jt"],
              [20, 19, 18, "Little Finger DIP jt"], [19, 18, 17, "Little finger PIP jt"], [18, 17, 0, "Little finge MCP jt"]
             ]

def draw_finger_angles(image, results, joint_list):
    angle_results = []

    for hand_landmarks in results.multi_hand_landmarks:
        for joint in joint_list:
            a = np.array([hand_landmarks.landmark[joint[0]].x, hand_landmarks.landmark[joint[0]].y])
            b = np.array([hand_landmarks.landmark[joint[1]].x, hand_landmarks.landmark[joint[1]].y])
            c = np.array([hand_landmarks.landmark[joint[2]].x, hand_landmarks.landmark[joint[2]].y])

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            # Convert normalized coordinates to pixel coordinates
            b = tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int))

            # Add the results to the list
            angle_results.append([joint[3], angle])

            # Highlighting the added label name in the print statement
            cv2.putText(image, f"{joint[3]}: {angle:.2f} deg", b,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return image, angle_results

# Streamlit app header
st.title("Hand Tracking App")

# File upload widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Check if the user uploaded a file
if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)

    # Hand tracking using MediaPipe
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                if num == 0:  # assuming you are tracking only one hand
                    # Render left or right detection
                    text, coord = "Left Hand", (50, 50)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            image, angle_results = draw_finger_angles(image, results, joint_list)

            # Print the results in a table format using Streamlit
            st.text(tabulate(angle_results, headers=["Joint", "Angle"], tablefmt="grid"))

        # Display the image with annotations using Streamlit
        st.image(image, caption='Hand Tracking', channels="BGR", use_column_width=True)

# Display message if no file is selected
else:
    st.text("No file selected.")
