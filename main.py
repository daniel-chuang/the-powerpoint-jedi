"""
The Presentation Force:

An image recognition project that allows users to control their slideshow
presentations through hand gestures.
"""

"""
Hand landmark reference:
0 : Wrist
1-4 : Thumb
5-8 : Second Finger
9-12 : Middle Finger
13-16 : Ring Finger
17-20 : Pinky Finger

For finger landmarks, it is indexed from base to tip

Coordinate system reference:
X : x-position on image
Y : y-position on image
Z : estimated depth
"""

"""
Gesture for switching slides : Two finger swipe with right hand (for now)

Check for this by identifying if hand is in state of two fingers pointing
left or right at any point.

If the hand had two fingers pointing at either side for more than 3 frames,
and then the hand switches for more than 3 frames, then trigger an action.

Detect if the hand is a "two finger point" by checking if the distance between
the second and middle finger nodes is greater than the distance between the
ring and pinky finger by a certain percentage.

Then, get the direction by comparing the two finger node positions compared
to the position of the wrist (node 0).
"""

### Parameters ###
which_hand = "Right"

### Imports ###
from shutil import which
from sre_parse import expand_template
import cv2 # OpenCV
import mediapipe as mp # Google Mediapipe
import pyautogui as gui # PySimpleGUI
from google.protobuf.json_format import MessageToDict # Converting class
from sklearn.linear_model import LinearRegression
import numpy as np
from math import dist

from pprint import pprint # For development: to print the landmark values nicely

# Setting up Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands # The main model

### Functions ###
## Webcam input and output
def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read() # Reads the video capture from OpenCV
        image = cv2.flip(image, 1) # Flipped for selfie view
        
        # Checks if the capture doesn't work so it doesn't crash Mediapipe
        if not success:
            print("Empty camera frame -- Ignoring")
            continue

        # Runs the main algorithm otherwise
        results = find_hands(image)
        image = draw_hands(image, results)

        # Gets the landmarks of the chosen hand
        hand_landmarks = chosen_hand_landmarks(results)
        if hand_landmarks is not None:
            extended_list = check_raised_fingers(hand_landmarks)
            pprint(check_gesture(extended_list, hand_landmarks, which_hand))


        # Shows the resulting image (for development)
        cv2.imshow('MediaPipe Hands', image) 
        
        # Allows the user to end the capture
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release() # Ends the OpenCV capture

## Finding hand landmarks
def find_hands(image):
    # Sets up the model
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        # Applies the model to find the hands
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

    return results

## Drawing the hand landmarks and connections
def draw_hands(image, results):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style() # Draws the landmarks on the image
            )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

## Assigns left or right, converts data to a list
def handedness(results):
    if results.multi_handedness == None:
        return []

    # Setting up a list to be returned
    hand_list = []

    # Converting Google's odd class type into a dictionary
    for idx, hand_handedness in enumerate(results.multi_handedness):
        handedness_dict = MessageToDict(hand_handedness)["classification"][0]
        hand_list.append(handedness_dict["label"])
    
    return hand_list

## Gets the hand landmarks of the chosen hand
def chosen_hand_landmarks(results):
    # Gets the index of the correct hand
    hand_list = handedness(results)
    hand_index = None
    if which_hand in hand_list:
        hand_index = hand_list.index(which_hand)
    
    # Getting the hand landmarks of the correct hand
    if hand_index != None:
        hand_landmarks_raw = results.multi_hand_landmarks[hand_index]
        hand_landmarks_raw = MessageToDict(hand_landmarks_raw)["landmark"]
        
        # Currently a list of dicts, converting to a matrix
        hand_landmarks = np.zeros(shape=[1,3]) # Creating an empty array
        for landmark in hand_landmarks_raw:
            hand_landmarks = np.vstack([hand_landmarks, list(landmark.values())])
        hand_landmarks = hand_landmarks[1:22]
        return hand_landmarks

## Returns an array of raised fingers
# Either None, Right-Point, or Left-Point
# - Uses hand_landmarks instead of results for only 1 hand
def check_raised_fingers(hand_landmarks): # Numpy array type
    if hand_landmarks is None:
        return None

    model = LinearRegression()

    # Performing a linear regression to check which fingers are extended
    finger_list = [] # thumb, second, middle, ring, pinky
    for iter in range(0, 5): # Iterating through fingers
        landmark_start = (iter * 4) + 1 # Starts the index at the first landmark

        # Have to flatten x and y for SciPy
        x = hand_landmarks[landmark_start:landmark_start + 4, 0].reshape(-1, 1)
        y = hand_landmarks[landmark_start:landmark_start + 4, 1].reshape(-1, 1)
        model.fit(x, y)
        finger_list.append({"slope": model.coef_, "intercept": model.intercept_, "r_sq" : model.score(x, y)})

    # Making a list for extended and not extended fingers
    extended_list = [True if finger["r_sq"] > 0.85 else False for finger in finger_list]
    
    # Verifying that extended fingers are extended by checking if the tip is the farthest
    for iter in range(len(extended_list)):
        if extended_list[iter] == True:
            landmark_start = (iter * 4) + 1 # Starts the index at the first landmark
            distance_base = dist(hand_landmarks[landmark_start + 1, 0:2], hand_landmarks[0, 0:2])
            distance_tip = dist(hand_landmarks[landmark_start + 3, 0:2], hand_landmarks[0, 0:2])
            if distance_tip <= distance_base:
                extended_list[iter] = False

    return extended_list

## Either returns a gesture or none using a list of extended fingers
def check_gesture(extended_list, hand_lankmarks, which_hand):
    # Two fingers pointing
    if False not in extended_list[1:3] and (True not in extended_list[3:5]):
        # Checking for palm facing (outward or inward)
        if which_hand == "Right":
            # Checking x position of second finger compared to thumb
            if hand_lankmarks[8, 0] > hand_lankmarks[4, 0]:
                return "Facing"
            else:
                return "Away"
        elif which_hand == "Left": # For left hand
            if hand_lankmarks[8, 0] > hand_lankmarks[4, 0]:
                return "Away"
            else:
                return "Facing"
    return "None"


### Running the script ###
if __name__ == "__main__":
    main()