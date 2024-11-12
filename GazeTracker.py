import numpy as np
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import time
from scipy.io import loadmat
import dlib

from utils_webcam import find_landmarks, get_normalized, estimate_gaze, show_proj_gaze

class GazeNet(nn.Module):  # Define neural network model class
    def __init__(self):
        super(GazeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)  # Output: 32x56x20
        self.conv2 = nn.Conv2d(20, 50, 5)  # Output: 12x24x50
        self.fc1 = nn.Linear(6 * 12 * 50, 128)  # Fully connected layer
        self.reg = nn.Linear(128, 2)  # Output: 2 (gaze direction)

    def forward(self, x, y):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # Max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # Max pooling
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.reg(x)  # Append head pose
        return x

"""class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        # Input: 1 channel grayscale image (36x60), 20 filters with 5x5 kernel
        self.layer1 = nn.Conv2d(1, 15, kernel_size=5)  # Output: 32x56x20
        # Second conv layer, 20 input channels to 50 output channels
        self.layer2 = nn.Conv2d(15, 60, kernel_size=5)  # Output: 12x24x50
        # Fully connected layer taking flattened features from second conv layer
        self.fc_hidden = nn.Linear(6*12*60, 64)  # Computed from maxpool2 dimensions
        # Regression layer for predicting 2 outputs
        self.output_layer = nn.Linear(64, 2)
        self.float()

    def forward(self, x,y):
        # First conv layer + ReLU + 2x2 max pooling, output size 16x28x20
        x = F.max_pool2d(F.relu(self.layer1(x)), kernel_size=2)
        # Second conv layer + ReLU + 2x2 max pooling, output size 6x12x50
        x = F.max_pool2d(F.relu(self.layer2(x)), kernel_size=2)
        # Flatten for fully connected layer
        x = torch.flatten(x, start_dim=1)
        # Hidden fully connected layer + ReLU
        x = F.relu(self.fc_hidden(x))
        # Final regression output layer
        x = self.output_layer(x)
        return x"""


"""class GazeNet_II(nn.Module):
    def __init__(self):
        super(GazeNet_II, self).__init__()
        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(1, 32, 5)  # Input: 1x36x60, Output: 32x32x56
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1

        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(32, 64, 5)  # Input: 32x32x56, Output: 64x28x52
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2

        # Fully connected layer with dropout
        self.fc1 = nn.Linear(64 * 6 * 12, 128)  # Adjusted dimensions after pooling
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability

        # Regression layer for final output
        self.reg = nn.Linear(128, 2)

    def forward(self, x,y): 
        # First conv block: Conv -> ReLU -> Max Pool -> Batch Norm
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))  # Output: 32x16x28

        # Second conv block: Conv -> ReLU -> Max Pool -> Batch Norm
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))  # Output: 64x6x12

        # Flatten
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch dimension

        # Fully connected layer with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout

        # Regression layer for gaze prediction
        x = self.reg(x)
        return x"""

    
class TorchGazeTracker:
    def __init__(self, screen_width=1920, screen_height=1080, model_path="", camera_matrix="", dist_coef_path=""):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Load pre-trained GazeNet model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gazenet = GazeNet().to(self.device)
        self.gazenet = torch.load(model_path, map_location=self.device)  # Load the model

        # Load Dlib face detector and landmark predictor
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor("Deep_L/shape_predictor_68_face_landmarks.dat")

        # Load face model and camera matrix
        self.facemodel = loadmat("Deep_L/6 points-based face model.mat")["model"]
        self.camera_mat = np.load(camera_matrix)
        self.dist_coef = np.load(dist_coef_path) if dist_coef_path else np.zeros((1, 5))

        # Initialize transformations
        self.tfm = transforms.Compose([transforms.ToTensor()])
        self.screen_width = screen_width
        self.screen_height = screen_height
    

        # Deque for storing gaze points over time (2 seconds worth of frames)
        self.gaze_points = deque(maxlen=30)  # With a 30 FPS camera, stores 30 frames (1 seconds)
        self.landmarks = None

        self.gaze_direction=None # 0 for left, 1 for centre and 2 for right

    

    def convert_polar_vector(self, angles):
        y = -1 * torch.sin(angles[:, 0])
        x = -1 * torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
        z = -1 * torch.cos(angles[:, 0]) * torch.cos(angles[:, 1])
        mag_v = torch.sqrt(x * x + y * y + z * z)
        x /= mag_v
        y /= mag_v
        z /= mag_v
        return x, y, z

    def process_frame(self, img):
        # Get image dimensions and detect landmarks
        img_h, img_w = img.shape[:2]
        self.landmarks = find_landmarks(img, self.dlib_detector, self.dlib_predictor)

        if self.landmarks.shape[0] == 0:
            return None, None

        # Get normalized eye images
        norm_img_l, hr_vec_norm_l, ht_vec_l = get_normalized(img, self.facemodel, self.landmarks, self.camera_mat, self.dist_coef, "l")
        norm_img_r, hr_vec_norm_r, ht_vec_r = get_normalized(img, self.facemodel, self.landmarks, self.camera_mat, self.dist_coef, "r")

        # Estimate gaze vectors using the pre-trained model
        g_vec_l = estimate_gaze(norm_img_l, hr_vec_norm_l, self.gazenet, self.device)
        g_vec_r = estimate_gaze(norm_img_r, hr_vec_norm_r, self.gazenet, self.device)

        return g_vec_l, g_vec_r

    def classify_gaze_direction(self, avg_gaze):
        """
        Classifies the gaze as 'left', 'right', or 'center' based on the average gaze vector.
        Using the horizontal component (x) of the gaze vector for classification.
        """
        x_avg = avg_gaze[0]  # Use x-component of the average gaze vector
        
    
        if x_avg < -0.05:  # Threshold for left # -0.07
            return "left"
        elif x_avg > 0.02:  # Threshold for right #0.04 
            return "right"
        else:
            return "center"
    

    def compute_average_gaze(self):
        """
        Computes the average gaze direction from the deque containing gaze points.
        Returns a vector representing the average gaze.
        """
        if len(self.gaze_points) == 0:
            return None
        
        avg_gaze = np.mean(self.gaze_points, axis=0)
        return avg_gaze

    def run(self):
        start_time = time.time()
        while True:
            ret, img = self.cap.read()
            if not ret:
                break

            # Process the frame to get gaze vectors
            g_vec_l, g_vec_r = self.process_frame(img)

            if g_vec_l is not None and g_vec_r is not None:
                # Average the left and right gaze vectors
                avg_gaze_vec = (g_vec_l + g_vec_r) / 2

                # Convert to NumPy only if it's still a tensor
                if isinstance(avg_gaze_vec, torch.Tensor):
                    avg_gaze_vec = avg_gaze_vec.detach().cpu().numpy()

                # Append the gaze vector to the deque
                self.gaze_points.append(avg_gaze_vec)

                # Every 2 seconds, compute and classify the average gaze
                current_time = time.time()
                if current_time - start_time >= 1.0:
                    avg_gaze = self.compute_average_gaze()
                    if avg_gaze is not None:
                        """gaze_direction = self.classify_gaze_direction(avg_gaze)
                        """
                        print(f"Average Gaze Direction: {avg_gaze[0]}")

                        if avg_gaze[0] < -0.03:  # Threshold for left #0.07
                            self.gaze_direction=0
                            print("Average Gaze Direction: Left")
                        elif avg_gaze[0] > 0.04:  # Threshold for right
                            self.gaze_direction=2
                            print(f"Average Gaze Direction: Right")
                        else:
                            self.gaze_direction=1
                            print(f"Average Gaze Direction: Centre") # ",avg_gaze[0]
                    # Reset start time for the next 2-second window
                    start_time = current_time

                # Show the projection of the gaze on the webcam feed
                show_proj_gaze(g_vec_l, g_vec_r, self.landmarks, img,False)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close all windows
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    model_path = "Deep_L/base.pt" #model2.pt" 
    camera_matrix_path = "Deep_L/camera_mat.npy"
    dist_coef_path = None  

    # Create instance and run the gaze tracker
    gaze_tracker = TorchGazeTracker(
        screen_width=1920,
        screen_height=1080,
        model_path=model_path,
        camera_matrix=camera_matrix_path,
        dist_coef_path=dist_coef_path
    )
    gaze_tracker.run()
