import shutil
import cv2
import os
import mediapipe as mp
import numpy as np

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 
IMAGE_PATH = os.path.join('IMG_DATA')

words = []

with open('Words.txt', 'r') as file:
    # Read each line and store it in the list
    for line in file:
        words.append(line.strip())  # Remove leading and trailing whitespaces if needed

actions = np.array(words)   #words list
sequence_count = 30     #50 trials
sequence_length = 30    #30 frames / trial 

mp_pose = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils #drawing helpers
mp_holistic = mp.solutions.holistic     #holistic model

def folder_init():
    #Create 30 folders per action
    for action in actions: 
        for sequence in range(sequence_count):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                os.makedirs(os.path.join(IMAGE_PATH, action, str(sequence)))
            except:
                pass

def folder_restart():
    try:
        # Delete the entire directory tree
        shutil.rmtree(DATA_PATH)
        shutil.rmtree(IMAGE_PATH)
        print(f"Folder '{DATA_PATH}, {IMAGE_PATH}' and its subfolders deleted successfully.")
    except FileNotFoundError:
        pass

    folder_init()

#recolors image and make detections
def recolor(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=2))  #change display

    
def extract_keypoints(results): #convert landmakrs to nparray
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


folder_restart()


# start webcam capture
cap = cv2.VideoCapture(0)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

directory = actions[0]

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through sequences aka videos
    for sequence in range(30):
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = recolor(frame, holistic)

            img_path = os.path.join(IMAGE_PATH, directory, str(sequence), f"frame{frame_num}.jpg")
            cv2.imwrite(img_path, image)

            # Draw landmarks
            draw_landmarks(image, results)
            
            # NEW Apply wait logic
            if frame_num == 0: 
                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(directory, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(500)
            else: 
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(directory, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(20)
            # NEW Export keypoints
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, directory, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

cap.release()
cv2.destroyAllWindows()