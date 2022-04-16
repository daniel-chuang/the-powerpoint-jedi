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
import cv2 # OpenCV
import mediapipe as mp # Google Mediapipe
import pyautogui as gui # PySimpleGUI
from google.protobuf.json_format import MessageToDict # Converting class

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
        print(hand_landmarks)

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

## Assigns left or right
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
        hand_landmarks = results.multi_hand_landmarks[hand_index]
        return hand_landmarks

## Checking hand gesture type
# Either None, Right-Point, or Left-Point
def check_gesture():

    pass

### Running the script ###
if __name__ == "__main__":
    main()