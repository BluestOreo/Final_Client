"""
VIDEO STREAMING CLIENT ASSISTANT CLASS

This class acts as a client for video streaming and remote command communication with a server.
It establishes two TCP connections:
1. A connection to receive live video frames from the server. (Port 8000)
2. A connection to send and receive control commands to/from the server. (Port 5000)
3. Saves captured frames on to file to be read by parent class


"""

import numpy as np
import cv2
import socket
import io
import sys
import struct
from PIL import Image
from multiprocessing import Process
from Command import COMMAND as cmd

class VideoStreaming:
    def __init__(self):
        
        # Initialize flags and variables
        self.video_Flag = True  # Indicates if a new video frame is ready to be processed
        self.connect_Flag = False  # Indicates if the client is connected to the server


    def StartTcpClient(self, IP):
        # Create TCP sockets for client-server communication
        self.client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Socket for sending commands
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # Socket for receiving video

    def StopTcpcClient(self):
        # shut down and close the TCP connections
        try:
            self.client_socket.shutdown(2)
            self.client_socket1.shutdown(2)
            self.client_socket.close()
            self.client_socket1.close()
        except:
            pass  # Ignore any errors during shutdown

    def IsValidImage4Bytes(self, buf):
        # Validate if the image data buffer is a valid JPEG image
        bValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):
            # Check if the image has correct JPEG end bytes
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                bValid = False
        else:
            # Attempt to verify the image using PIL
            try:
                Image.open(io.BytesIO(buf)).verify()
            except:
                bValid = False
        return bValid

    def streaming(self, ip):
        # Start streaming video from the server
        stream_bytes = b' '
        try:
            # Connect to the server for video streaming
            self.client_socket.connect((ip, 8000))
            self.connection = self.client_socket.makefile('rb')
        except Exception as e:
            # Handle connection errors and print the exception
            print(f"Streaming method failed at streaming service: {e}")

        # Continuously read and process video frames
        while True:
            try:
                # Read the length of the incoming frame
                stream_bytes = self.connection.read(4)
                leng = struct.unpack('<L', stream_bytes[:4])
                
                # Read the actual JPEG image data
                jpg = self.connection.read(leng[0])
                
                # Check if the received data is a valid image
                if self.IsValidImage4Bytes(jpg):
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    # Save the image to a file and set the video flag
                    cv2.imwrite('video.jpg', image)
                    self.video_Flag = False  # Indicate that a new image is available to be received by parent file
            except Exception as e:
                # Print any exceptions and exit the loop
                print(e)
                break

    def sendData(self, s):
        # Send command data to the server if connected
        if self.connect_Flag:
            self.client_socket1.send(s.encode('utf-8'))

    def recvData(self):
        # Receive data from the server
        data = ""
        try:
            data = self.client_socket1.recv(1024).decode('utf-8')
        except:
            pass  # Ignore any errors during receiving
        return data

    def socket1_connect(self, ip):
        # Establish a connection for sending commands to the server
        try:
            self.client_socket1.connect((ip, 5000))
            self.connect_Flag = True
            print("Connection Successful!")
        except Exception as e:
            print("Connect to server Failed!: Server IP is right? Server is opened?")
            self.connect_Flag = False

if __name__ == '__main__':
    pass
