"""

<<MOST RECENT-Tuesday>>


Function to track blinking
"""

import math
import cv2
import mediapipe as mp
import numpy as np


class BlinkTracker:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.FONTS = cv2.FONT_HERSHEY_SIMPLEX
        
        self.blink_POS = None # 0 for center, -1 for left and 1 for right
        self.blink_count = 0  # Initialize the blink counter
        self.command_tracker = 0


    def euclidean_distance(self, point, point1):
        x, y = point.x, point.y
        x1, y1 = point1.x, point1.y
        return math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    def blink_ratio(self, landmarks):
        # Calculate blink ratio for left eye from eye border regions
        vertical_distance = self.euclidean_distance(landmarks[386], landmarks[374])
        horizontal_distance = self.euclidean_distance(landmarks[362], landmarks[263])
        L_ratio = vertical_distance / horizontal_distance
        #print("L:", L_ratio, end=" ")


        # Calculate blink ratio for right eye from eye border regions
        vertical_distance = self.euclidean_distance(landmarks[159], landmarks[145])
        horizontal_distance = self.euclidean_distance(landmarks[33], landmarks[133])
        R_ratio = vertical_distance / horizontal_distance
        #print("R:",R_ratio)

        return L_ratio,R_ratio



    def blink_detection_run(self):

        detected_pos=None
        detected_pos_temp=None
        freshly_open=False
   
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate the blink ratio
                    L_ratio, R_ratio = self.blink_ratio(face_landmarks.landmark)
                    
                    #Flip frame
                    frame = cv2.flip(frame, 1)
                    
                    if L_ratio < 0.25 and R_ratio < 0.25:
                        cv2.putText(frame, "BOTH EYE BLINKED", (50, 50), self.FONTS, 1, (0, 0, 255), 5)
                        detected_pos_temp = 0
                        #BLINK NOT AS SENSITIVE WHEN BOTH ARE CLOSED
                        self.blink_count += 2  #manually give it a boost

                    elif L_ratio < 0.25:
                        cv2.putText(frame, "ONLY Left Detected", (50, 50), self.FONTS, 1, (0, 0, 255), 3)
                        detected_pos_temp = -1

                    elif R_ratio < 0.25:
                        cv2.putText(frame, "ONLY Right Detected", (50, 50), self.FONTS, 1, (0, 0, 255), 3)
                        detected_pos_temp = 1

                    else:                      
                        if self.blink_count>3:
                            freshly_open=True
                        self.blink_count = 0  #Reset the counter if no blink is detected
                        detected_pos_temp=None

                    
                    # Increment counter if same blink detected
                    if detected_pos is not None and detected_pos == detected_pos_temp:
                        self.blink_count += 1
                    else:
                        self.blink_count = 1  # Reset counter to one if new position is detected
                        detected_pos = detected_pos_temp


                    # Update blink_POS after 7 consecutive frames
                    if self.blink_count >= 7:
                        possible_blink=detected_pos
                       
                    #wait till eye are open after a blink to update variable
                    if freshly_open is True:
                        
                        self.blink_POS = possible_blink
                        
                        #Reset variables
                        detected_pos=None
                        possible_blink=None
                        self.blink_count = 0  # Reset counter after updating 
                        freshly_open=False     
                        self.command_tracker=0
            
            #print("Count", self.blink_count) 
            if self.blink_POS is not None:
                print("Blink at", self.blink_POS) 

                #Now updated in parent class                
                self.blink_POS=None

           
            
            cv2.imshow('Eye Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    eye_tracker = BlinkTracker()
    eye_tracker.blink_detection_run()
