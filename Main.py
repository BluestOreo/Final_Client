"""
CODE TO RUN AS CLIENT:

This script implements a client interface for controlling a remote wheelchair via a graphical user interface (GUI).

It establishes a TCP connection to communicate with a server, providing functionalities such as:

1. Video streaming: Receives and displays live video from the server.
2. Motor control: Commands to move the vehicle forward, left, or right, or stop it, based on eye gaze direction detected via a gaze tracker.
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
from threading import Timer
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

        self.root = root
        self.root.title("Client interface window")

        #initialize variables
        self.IP='172.20.10.3' # IP address for rapsberry pi 4 when connected to 'Iphone' Hotspot
        self.h = self.IP

        self.endChar = '\n'
        self.intervalChar = '#'
        
        self.TCP = VideoStreaming() #class from video file

        self.servo1 = 90
        self.servo2 = 90
        self.Ultrasonic=0

        
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
        self.connect_button = ttk.Button(self.button_frame, text="Connect", command=self.connect)
        self.connect_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.disconnect_button = ttk.Button(self.button_frame, text="Disconnect", command=self.disconnect)
        self.disconnect_button.pack(side=tk.RIGHT, padx=10, pady=10)


        # Add command button to debug functions
        self.connect_button = ttk.Button(self.button_frame, text="Command", command=self.command)
        self.connect_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.gaze_tracker = None
        

        self.video_running = False
     
    #===========Connect Button Callback Function=============
    def connect(self):
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
            self.TCP.StartTcpClient(self.h)  # Establish TCP connection
        except Exception as e:
            print('TCP client connection error:', e)

        #==========Video Stream========================
        try:
            self.streaming = Thread(target=self.TCP.streaming, args=(self.h,))
            self.streaming.start()

            print("Connection succesful!")
            
            # If succesful intialize  Video update logic
            try:
            
                self.video_update() #contains an inbuilt self refresh call hence no need for extra threads

            except Exception as e:
                print('Video Update error:', e)
        
        except Exception as e:
            print('Stream video error:', e)

       

        #===========Servo/Motor Control thread=============
        try:
             
            #Also contains an inbuilt self refresh call hence no need for extra threads
            #self.update_servo()
            self.update_motor()

        except Exception as e:
            print('Servo update method error:', e)
        
        #===========Message receiving thread=============
        try:
            self.recv = Thread(target=self.recvmassage)
            self.recv.start()
        except Exception as e:
            print('Receive message thread error:', e)

        print('Server address:' + str(self.h) + '\n')

    
    #===========Supporting Method to validate image captured on screen before display=============

    def is_valid_jpg(self, jpg_file):
        try:
            # Check if the file extension is .jpg
            if jpg_file.lower().endswith('.jpg'):
                with open(jpg_file, 'rb') as f:
                    buf = f.read()

                    # Check if the file starts with JPEG start bytes
                    if not buf.startswith(b'\xff\xd8'):
                        return False

                    # Check for JFIF or Exif headers and verify end bytes
                    if buf[6:10] in (b'JFIF', b'Exif'):
                        if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                            return False
                    else:
                        # Attempt to open the file with PIL to verify it
                        try:
                            Image.open(f).verify()
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
                if self.is_valid_jpg('video.jpg'):
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

        self.root.after(5, self.video_update) # refresh every 5mS
        self.TCP.video_Flag = True
        

    #===========Disconnect Button Callback Function=============
    def disconnect(self):
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

    #===========Command Button Callback Function=============
    def command(self):
        
        print("Command button pressed!")
        print("current eye gaze is:", self.gaze_tracker.gaze_direction)

            
        method_list = [
            #self.WC_BackWard,
            #self.WC_ForWard,
            #self.WC_Moveleft,
            #self.Change_Left_Right(1) #Set it to centre
            self.WC_Stop,
            #self.WC_Moveright,
            
        ]
        
        # Loop through the methods and call each one
        for method in method_list:
            method()  # Call each method in the list
    
     #===========Update Servo Function=============
     # # Function to control the servo motor based on the gaze direction flag
    def update_servo(self):
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
            self.TCP.sendData(cmd.CMD_SERVO + self.intervalChar + '0' + self.intervalChar + str(self.servo1) + self.endChar)
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

   
    #===========Motor Actuation Commands=============
    def WC_ForWard(self):
        print("Forward Method  executed")
        ForWard = self.intervalChar + str(100) + self.intervalChar + str(100) + self.intervalChar + str(
            100) + self.intervalChar + str(100) + self.endChar
        try:
            self.TCP.sendData(cmd.CMD_MOTOR + ForWard)
            print("sent:"+cmd.CMD_MOTOR + ForWard)
        except:
            print("DIDN'T send:" + cmd.CMD_MOTOR + ForWard)

    def WC_BackWard(self):
        print("Backward Method  executed")
        BackWard = self.intervalChar + str(-1500) + self.intervalChar + str(-1500) + self.intervalChar + str(
             -1500) + self.intervalChar + str(-1500) + self.endChar
        try:
            self.TCP.sendData(cmd.CMD_MOTOR + BackWard)
        except:
            print(cmd.CMD_MOTOR + BackWard)

    def WC_Stop(self):
        
        Stop = self.intervalChar + str(0) + self.intervalChar + str(0) + self.intervalChar + str(
            0) + self.intervalChar + str(0) + self.endChar
        
        try:
            self.TCP.sendData(cmd.CMD_MOTOR + Stop)
            print("Stop Method  executed")
        except:
            print("Motor Didn't DO: " + cmd.CMD_MOTOR + Stop)

    def WC_Moveleft(self):
        print("Left Method  executed")
        R_Move_Left = self.intervalChar + str(0) + self.intervalChar + str(0) + self.intervalChar + str(
            90) + self.intervalChar + str(1500) + self.endChar
        try:
            self.TCP.sendData(cmd.CMD_CAR_ROTATE + R_Move_Left)
        except:
            print(cmd.CMD_CAR_ROTATE + R_Move_Left)

    def WC_Moveright(self):
        print("Right Method  executed")  
        R_Move_Right = self.intervalChar + str(0) + self.intervalChar + str(0) + self.intervalChar + str(
            -90) + self.intervalChar + str(1500) + self.endChar
        try:
            self.TCP.sendData(cmd.CMD_CAR_ROTATE + R_Move_Right)
        except:
            print(cmd.CMD_CAR_ROTATE + R_Move_Right)

    #===========Ultrasonic Reading Request Function=============
    def request_Ultrasonic(self):
        if self.Ultrasonic.text() == "Ultrasonic": # prompt to server to send ultrasonic data
            self.TCP.sendData(cmd.CMD_SONIC + self.intervalChar + '1' + self.endChar)
        else: # once value has been updated to anything but 'ultrasonic' send command to stop sending
            self.TCP.sendData(cmd.CMD_SONIC + self.intervalChar + '0' + self.endChar)
            self.Ultrasonic.setText("Ultrasonic")

    #===========Servo Actuation=============
    def Change_Left_Right(self, direction):  # Left or Right motion by Servo 1
        
        if direction == 0:  # Left position
            self.servo1 = 30
        elif direction == 2:  # Right position
            self.servo1 = 150
        elif direction == 1:  # Center position
            self.servo1 = 90
        else:
            print("Invalid input at Servo Control method")

        self.TCP.sendData(cmd.CMD_SERVO + self.intervalChar + '0' + self.intervalChar + str(self.servo1) + self.endChar)
        
    def Change_Up_Down(self):  # Up or Down motion by Servo 2
        self.servo2 = 22 #TO DO: define how the inputs are acquired
        self.TCP.sendData(cmd.CMD_SERVO + self.intervalChar + '1' + self.intervalChar + str(self.servo2) + self.endChar)
        #self.label_Servo2.setText("%d" % self.servo2)


    #===========Combined Motor Actuation Commands=============
    def update_motor(self):
        print("Received:", self.gaze_tracker.gaze_direction)
        command_flag = self.gaze_tracker.gaze_direction  # Get the updated flag

        # First send command to stop the motor after 1s of execution of previous loops if any
        Stop = self.intervalChar + str(0) + self.intervalChar + str(0) + self.intervalChar + str(0) + self.intervalChar + str(0) + self.endChar
        try:
            self.TCP.sendData(cmd.CMD_MOTOR + Stop)
            print("Stop Method executed")
        except Exception as e:
            print("Could NOT send stop message:", e)

        if command_flag is not None:  # Check if the flag is updated
            if command_flag == 1:  # Move forward
                ForWard = self.intervalChar + str(500) + self.intervalChar + str(500) + self.intervalChar + str(500) + self.intervalChar + str(500) + self.endChar
                try:
                    self.TCP.sendData(cmd.CMD_MOTOR + ForWard)
                    print("Sending Forward:", cmd.CMD_MOTOR + ForWard)
                except:
                    print("DIDN'T send:", cmd.CMD_MOTOR + ForWard)

            elif command_flag == 0:  # Move left
                R_Move_Left = self.intervalChar + str(200) + self.intervalChar + str(0) + self.intervalChar + str(1100) + self.intervalChar + str(900) + self.endChar
                try:
                    self.TCP.sendData(cmd.CMD_MOTOR + R_Move_Left)
                    print("Sending Left:", cmd.CMD_MOTOR + R_Move_Left)
                except:
                    print("DIDN'T send:", cmd.CMD_MOTOR + R_Move_Left)

            elif command_flag == 3:  # Move right
                R_Move_Right = self.intervalChar + str(200) + self.intervalChar + str(0) + self.intervalChar + str(1100) + self.intervalChar + str(900) + self.endChar
                try:
                    self.TCP.sendData(cmd.CMD_MOTOR + R_Move_Right)
                    print("Sending Right:", cmd.CMD_MOTOR + R_Move_Right)
                except:
                    print("DIDN'T send:", cmd.CMD_MOTOR + R_Move_Right)

            # Reset the command after use
            self.gaze_tracker.gaze_direction = None
        else:
            # If no command is detected, ensure the motor remains in a safe state
            try:
                self.TCP.sendData(cmd.CMD_MOTOR + Stop)
                print("Ensuring Stop")
            except:
                print("Motor Didn't STOP:", cmd.CMD_MOTOR + Stop)

        # Reset flag variable
        command_flag = None
        # Refresh every second
        self.root.after(1000, self.update_motor)  # Call again after 1 second

       

    #===========Message Handling Function=============
    #Function to receive TCP messages from Server, and Decrypt them.
    def recvmassage(self):
        self.TCP.socket1_connect(self.h)
        restCmd = "" # to handle incomplete commands

        while True: #listen continuously
            Alldata = restCmd + str(self.TCP.recvData())
            restCmd = ""
            print(Alldata)
            if Alldata == "": # nothing received
                break
            else:
                cmdArray = Alldata.split("\n")
                if (cmdArray[-1] != ""): # if not at the last instruction
                    restCmd = cmdArray[-1]
                    cmdArray = cmdArray[:-1]
            for oneCmd in cmdArray:
                Massage = oneCmd.split("#")
                if cmd.CMD_SONIC in Massage:
                    # self.Ultrasonic.setText('Obstruction:%s cm' % Massage[1])
                    u = 'Obstruction:%s cm' % Massage[1]
                    self.U.send(u)




 





if __name__ == "__main__":
    root = tk.Tk()

    app = ClientWindow(root)

    # Handle closing of the window
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
