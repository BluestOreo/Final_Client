"""
Manual Control


CODE TO RUN AS CLIENT:

This script implements a client interface for controlling a remote wheelchair via a graphical user interface (GUI).

It establishes a TCP connection to communicate with a server, providing functionalities such as:

1. Video streaming: Receives and displays live video from the server.
2. Motor control: Commands to move the wheelchair forward, left, or right, or stop it, based on eye gaze direction detected via a gaze tracker.
3. Servo control: Adjusts servo motor positions for specific directions using gaze tracking data.
4. Ultrasonic feedback: Receives and processes ultrasonic sensor data from the server.
5. Connection management: Allows the user to connect, disconnect, and toggle video streaming, all within a full-screen GUI.

The program uses multithreading to manage separate tasks: video streaming, receiving messages, and eye tracking. 

Library dependencies include OpenCV, Tkinter, PIL, and custom modules for thread and video handling.
"""


#import libraries
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

import numpy as np
import cv2
"""import socket
import os
import io
import time
import sys"""
#from threading import Timer
from threading import Thread


from multiprocessing import Process


#custom files
from GazeTracker import *
from Video import *
from Thread import *
from Command import COMMAND as cmd 


#===========================Client Window============================================
#define interface window class and methods
class ClientWindow:
    def __init__(self, root):

        global timer


        #==========initialize connection variables==============
        self.IP='172.20.10.3' # IP address for rapsberry pi 4 when connected to 'Iphone' Hotspot
        self.gaze_tracker = None
        self.video_running = False
        self.TCP = VideoStreaming() #class from video file
        


        #======intiialize sensor and actuator variables===========
        self.servo1 = 90
        self.servo2 = 90
        self.Ultrasonic=0


        #===========Interface window  intialization==============
        self.root = root
        self.root.title("Client interface window")
        # Set window to full screen
        self.root.attributes("-fullscreen", True)
        
        # Get screen width and height
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Calculate the video frame height (about 90% of the screen height)
        self.video_height = int(self.screen_height * 0.9)
        self.button_height = int(self.screen_height * 0.1)

        # Create a frame for the video (90% of the window)
        self.video_frame = tk.Frame(self.root, width=self.screen_width, height=self.video_height)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas to display the video
        self.video_canvas = tk.Canvas(self.video_frame, width=self.screen_width, height=self.video_height)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Create a frame for the buttons (10% of the window)
        self.button_frame = tk.Frame(self.root, height=self.button_height)
        self.button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Add Connect and Disconnect buttons to the button frame
        self.connect_button = ttk.Button(self.button_frame, text="Connect", command=self.connect_btn)
        self.connect_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.disconnect_button = ttk.Button(self.button_frame, text="Disconnect", command=self.disconnect_btn)
        self.disconnect_button.pack(side=tk.RIGHT, padx=10, pady=10)


        # Add stop command button to debug functions
        self.connect_button = ttk.Button(self.button_frame, text="Stop", command=self.stop_btn)
        self.connect_button.pack(side=tk.LEFT, padx=0, pady=10)

        # Add Video button to Initiate streaming
        self.connect_button = ttk.Button(self.button_frame, text="Video", command=self.vid_stream_btn)
        self.connect_button.pack(side=tk.LEFT, padx=5, pady=10)

        #========Message Handling predefined commands=======
        #encode commands to reduce amount of data sent  
        self.Buzzer_ON="BO"

        self.Ultrasonic_Request="UR"
        self.Ultrasonic_Update="UU"
        
        self.Gyro_Request="GR"
        self.Gyro_Update="GU"

        self.Servo_H="SH"
        self.Servo_V="SV"


        self.Motor_FORWARD="MF"
        self.Motor_STOP="MS"
        self.Motor_LEFT="ML"
        self.Motor_RIGHT="MR"
 
        self.stop_Char = '\n' # new line for new commands
        self.break_Char = '%' #delimiter variable

        


       
     
    #===========Connect Button Callback Function=============
    def connect_btn(self):
        print("Connect has been pressed!")

        #===========Local Camera===================
        try: 
            # Instantiate the TorchGazeTracker object to track eye gaze locally
            self.gaze_tracker = TorchGazeTracker(
                screen_width=1920,
                screen_height=1080,
                model_path = "Deep_L/gazenet_g_lenet.pt",
                camera_matrix = "Deep_L/camera_mat.npy",
                dist_coef_path = None,
            )

            # Start gaze tracking in a separate thread
            self.eyetracking = Thread(target=self.gaze_tracker.run)
            self.eyetracking.start()

        except Exception as e:
            print('Local video error:', e)

        #==========STREAMING CODE========================
        # Start the TCP client connection
        try:
            self.TCP.StartTcpClient(self.IP)  # Establish TCP connection
        except Exception as e:
            print('TCP client connection error:', e)


        

        #===========Message receiving thread=============
        try:
            self.message_handler = Thread(target=self.receive_command)
            self.message_handler.start()
        except Exception as e:
            print('Receive message thread error:', e)

        print('Client IP Address:' + str(self.IP) + '\n')

    
    #===========Supporting Method to validate image captured on screen before display=============
    def validate_jpg(self, file_path):
        try:
            # Confirm the file has a .jpg extension
            if file_path.lower().endswith('.jpg'):
                with open(file_path, 'rb') as file:
                    content = file.read()

                    # Check for JPEG start markers
                    if not content.startswith(b'\xff\xd8'):
                        return False #invalid file

                    # Check for JFIF or Exif headers and verify JPEG end markers
                    if content[6:10] in (b'JFIF', b'Exif'):
                        if not content.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                            return False
                    else:
                        # Attempt to verify image using PIL for further validation
                        try:
                            Image.open(file).verify()
                        except Exception:
                            return False

                return True
            else:
                return False
        except Exception:
            return False



    
    #===========Method to update video on screen from file saved=============
    def video_update(self):
        
        # Get the screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate 90% of the screen dimensions
        target_width = int(screen_width)
        target_height = int(screen_height * 0.9)

        #check if a new image has been saved
        if self.TCP.video_Flag:
                
            self.TCP.video_Flag = False
            try:
                # Load the image and check if it's a valid JPEG
                if self.validate_jpg('video.jpg'):
                    # Load the image using Pillow (PIL)
                    img = Image.open('video.jpg')

                    # Flip the image horizontally
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    #Resize the image to fit 90% of the screen
                    img_resized = img.resize((target_width, target_height), Image.LANCZOS)

                    # Convert the image to a format Tkinter can display
                    img_tk = ImageTk.PhotoImage(img_resized)

                    # Update the canvas with the new frame
                    self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    self.video_canvas.image = img_tk

            except Exception as e:
                pass
                #print("Video Update Error:", e)

        self.root.after(20, self.video_update) # refresh every 20mS 
        self.TCP.video_Flag = True
        

    #===========Disconnect Button Callback Function=============
    def disconnect_btn(self):
        print("Disconnected!")

        #==========STREAMING CODE==========
        self.TCP.StopTcpcClient()

        try:
            stop_thread(self.recv)
            stop_thread(self.streaming)
        except:
            pass

        #========LOCAL CAMERA CODE========
        try:
           stop_thread(self.eyetracking)

        except:
            pass

        #========Main window==============
        self.video_canvas.delete("all")  # Clear the video feed
        self.root.quit()

   
    #===========Video Button Callback Function=============
    def vid_stream_btn(self):

        #=============Video Stream================
        try:
            self.streaming = Thread(target=self.TCP.streaming, args=(self.IP,))
            self.streaming.start()

            print("Connection succesful!")
            
            # If succesful intialize  Video update logic
            try:
                self.video_update() #contains an inbuilt self refresh call hence no need for extra threads
            except Exception as e:
                print('Video Update error:', e)
        
        except Exception as e:
            print('Stream video error:', e)

    
        #===========Actuator Control=============
        try:
             
            #Also contains an inbuilt self refresh call hence no need for extra threads
            #self.update_servo() # design was changed to use caster wheels hence no need for servo
            self.update_motor()

        except Exception as e:
            print('Servo update method error:', e)
       
    
    
    #===========Stop Command Button Callback Function=============
    def stop_btn(self):
        
        print("Stop button pressed!")
        print("current eye gaze is:", self.gaze_tracker.gaze_direction)

        method_list = [
            #self.WC_BackWard,
            #self.WC_ForWard,
            #self.WC_Moveleft,
            self.WC_Stop,
            #self.WC_Moveright,
            
        ]
        
        # Loop through the methods and call each one
        for method in method_list:
            try:
                method()  # Call each method in the list
            except:
                pass


     #===========Update Servo Function=============
    def update_servo(self):
        "Function to control the servo motor based on the gaze direction flag"

        print("Received:", self.gaze_tracker.gaze_direction)
        flag_variable = self.gaze_tracker.gaze_direction  # Get the updated flag
        
        if flag_variable is not None:  # Check if the flag is updated
            if flag_variable == 0:  # Left position
                self.servo1 = 30
            elif flag_variable == 2:  # Right position
                self.servo1 = 150
            elif flag_variable == 1:  # Center position
                self.servo1 = 90
            
            # Send the updated servo position
            self.TCP.sendData( self.Servo_H + self.break_Char + str(self.servo1) + self.stop_Char)
            print(f"Servo moved to position {self.servo1}")

            #rest variable after use
            self.gaze_tracker.gaze_direction=None
        else:
            #if no detection then slow down and eventually stop motors/servo
            pass

        # Refresh every seconds
        self.root.after(1000, self.update_servo)  # Call again after 2 seconds

        flag_variable=False #reset variable
        
        
    #===========Window Close Function=============
    def on_closing(self):
        # Release the video capture and close the window
        self.cap.release()
        self.root.destroy()

   
    
    #===========Combined Motor Actuation Commands=============
    def update_motor(self):
        print("Received:", self.gaze_tracker.gaze_direction)
        command_flag = self.gaze_tracker.gaze_direction  # Get the updated flag

        # First send command to stop the motor after 1s of execution of previous loops if any
        try:
            self.TCP.sendData(self.Motor_STOP)
            print("Stop Method executed")
        except Exception as e:
            print("Could NOT send stop message:", e)

        if command_flag is not None:  # Check if the flag is updated
            if command_flag == 1:  # Move forward
                try:
                    self.TCP.sendData(self.Motor_FORWARD)
                    print("Sending Forward:", self.Motor_FORWARD)
                except:
                    print("DIDN'T send:", self.Motor_FORWARD)

            elif command_flag == 0:  # Move left
                try:
                    self.TCP.sendData(self.Motor_LEFT)                    
                    print("Sending Left:", self.Motor_LEFT)
                except:
                    print("DIDN'T send:",self.Motor_LEFT)

            elif command_flag == 3:  # Move right

                try:
                    self.TCP.sendData(self.Motor_RIGHT)
                    print("Sending Right:", self.Motor_RIGHT)
                except:
                    print("DIDN'T send:",self.Motor_RIGHT)

            # Reset the command after use
            self.gaze_tracker.gaze_direction = None
        else:
            # If no command is detected, ensure the motor remains in a safe state
            try:
                self.TCP.sendData(self.Motor_STOP)
                print("Ensuring Stop")
            except:
                print("Motor Didn't STOP:",self.Motor_STOP )

        # Reset flag variable
        command_flag = None
        # Refresh every second
        self.root.after(1000, self.update_motor)  # Call again after 1 second

    
    #===========Combined Sensor Actuator method===========
    def send_command(self,actuator):

        "Send request for actuation or sensing to the server"
        if actuator == self.Buzzer_ON or actuator == self.Gyro_Request or actuator == self.Ultrasonic_Request:
            try:
                self.TCP.sendData(actuator + self.stop_Char)
            except:
                print("Error at actuator/sensor send command")
        




    #===========Received Message Handling Function=============
    def receive_command(self):
        """Establishes TCP connection and continuously listens for incoming commands."""
        self.TCP.socket1_connect(self.IP)
        cmd_buff = ""  # to handle unexecuted commands

        while True:
            # Receive and combine any remaining incomplete command
            raw_data = cmd_buff + str(self.TCP.recvData())
            cmd_buff = ""  # Reset for the next cycle
            print(raw_data)

            if raw_data == "":  # Exit loop if no data is received
                break
            else:
                # Split received data by newline for command separation
                commands = raw_data.split("\n")
                
                # Handle any incomplete commands left at the end of the received data
                if commands[-1] != "":
                    cmd_buff = commands[-1]  # Save last part into command buffer
                    commands = commands[:-1]  # Remove the incomplete command from command list

            # Process each complete command
            for command in commands:
                command_parts = command.split("%")  # Split command by delimiter

                # Check for command keyword
                if self.Ultrasonic_Update in command_parts:
                    print("Distance from object is: " + command_parts[1] +"cm")

                elif self.Gyro_Update_Update in command_parts:
                    print("Gyroscope readings are: " + command_parts[1] + "and accelorometer readings are: " +command_parts[2])
                else:
                    print("Received and decrypted:" + commands) #check what actually sent?

 





if __name__ == "__main__":
    root = tk.Tk()

    app = ClientWindow(root)

    # Handle closing of the window
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
