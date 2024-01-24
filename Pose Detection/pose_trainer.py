import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv



BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_pose = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils #drawing helpers
mp_holistic = mp.solutions.holistic     #holistic model


cap = cv2.VideoCapture(0)
frame_rate = cap.get(cv2.CAP_PROP_FPS)


#Initialize coords.csv file
# num_coords = 501

# landmarks = ['class']
# for val in range(1, num_coords+1):
#     landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

# with open('coords.csv', mode='w', newline='') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerow(landmarks)

#33 pose landmarks, 468 face landmarks, and 21 hand landmarks per hand
class_name = "Lets Go Baby"


#recolors image and make detections
def recolor():
    #recolor to rgb
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    #make detection
    results = holistic.process(image)

    #recolor back to bgr
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def detect_landmarks():
    #face landmark
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    
    #left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    
    #right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    
    #holistic detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=2))  #change display
    


#Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    #while webcam is on
    while cap.isOpened():
        
        #success = return value; frame = data returned
        success, frame = cap.read()
        if not success:
            continue
        
        #render frame
        image, results = recolor()
        #detect face, hand, pose
        detect_landmarks()

        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate rows
            row = pose_row+face_row
            
            # Append class name 
            row.insert(0, class_name)
            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass

               #show feed
        cv2.imshow('Webcam Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):  #break loop if we hit 'q'
            break

cap.release()
cv2.destroyAllWindows()
