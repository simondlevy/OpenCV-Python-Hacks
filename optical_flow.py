#!/usr/bin/env python

'''
optical_flow.py - Optical-flow velocity calculation and display using OpenCV

    To test:

      % python optical_flow.py               # video from webcam
      % python optical_flow.py -f FILENAME   # video from file
      % python optical_flow.py -c CAMERA     # specific camera number
      % python optical_flow.py -s N          # scale-down factor for flow image
      % python optical_flow.py -m M          # move step in pixels

    Adapted from 
 
    https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/python/fback.py?rev=2271

    Copyright (C) 2014 Simon D. Levy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as 
    published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
'''

import cv2
import numpy as np

import time
import math
import optparse

class OpticalFlowCalculator:
    '''
    A class for optical flow calculations using OpenCV
    '''
    
    def __init__(self, frame_width, frame_height, scaledown=1,
                 perspective_angle=0, move_step=16, window_name=None, flow_color_rgb=(0,255,0)):
        '''
        Creates an OpticalFlow object for images with specified width and height.

        Optional inputs are:

          perspective_angle - perspective angle of camera, for reporting flow in meters per second
          move_step           - step size in pixels for sampling the flow image
          window_name       - window name for display
          flow_color_rgb    - color for displaying flow
        '''

        self.move_step = move_step
        self.mv_color_bgr = (flow_color_rgb[2], flow_color_rgb[1], flow_color_rgb[0])

        self.perspective_angle = perspective_angle

        self.window_name = window_name
        
        self.size = (int(frame_width/scaledown), int(frame_height/scaledown))

        self.prev_gray = None
        self.prev_time = None

    def processBytes(self, rgb_bytes, distance=None, timestep=1):
        '''
        Processes one frame of RGB bytes, returning summed X,Y flow.

        Optional inputs are:

          distance - distance in meters to image (focal length) for returning flow in meters per second
          timestep - time step in seconds for returning flow in meters per second
         '''

        frame = np.frombuffer(rgb_bytes, np.uint8)
        frame = np.reshape(frame, (self.size[1], self.size[0], 3))
        return self.processFrame(frame, distance, timestep)

    def processFrame(self, frame, distance=None, timestep=1):
        '''
        Processes one image frame, returning summed X,Y flow

        Optional inputs are:

          distance - distance in meters to image (focal length) for returning flow in meters per second
          timestep - time step in seconds for returning flow in meters per second
        '''

        frame2 = cv2.resize(frame, self.size)
 
        gray = cv2.cvtColor(frame2, cv2.cv.CV_BGR2GRAY)

        xsum, ysum = 0,0

        xvel, yvel = 0,0
        
        if self.prev_gray != None:

            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1, flags=0) 

            for y in range(0, flow.shape[0], self.move_step):

                for x in range(0, flow.shape[1], self.move_step):

                    fx, fy = flow[y, x]
                    xsum += fx
                    ysum += fy

                    cv2.line(frame2, (x,y), (int(x+fx),int(y+fy)), self.mv_color_bgr)
                    cv2.circle(frame2, (x,y), 1, self.mv_color_bgr, -1)

            # Default to system time if no timestep
            curr_time = time.time()
            if not timestep:
                timestep = (curr_time - self.prev_time) if self.prev_time else 1
            self.prev_time = curr_time

            xvel = self._get_velocity(flow, xsum, flow.shape[1], distance, timestep)
            yvel = self._get_velocity(flow, ysum, flow.shape[0], distance, timestep)

        self.prev_gray = gray

        if self.window_name:
            cv2.imshow(self.window_name, frame2)
            if cv2.waitKey(1) & 0x000000FF== 27: # ESC
                return None
        
       # Normalize and divide by timestep
        return  xvel, yvel

    def _get_velocity(self, flow, sum_velocity_pixels, dimsize_pixels, distance_meters, timestep_seconds):

        count =  (flow.shape[0] * flow.shape[1]) / self.move_step**2

        average_velocity_pixels_per_second = sum_velocity_pixels / count / timestep_seconds

        return self._velocity_meters_per_second(average_velocity_pixels_per_second, dimsize_pixels, distance_meters) \
               if self.perspective_angle and distance_meters \
               else average_velocity_pixels_per_second

    def _velocity_meters_per_second(self, velocity_pixels_per_second, dimsize_pixels, distance_meters):

        distance_pixels = (dimsize_pixels/2) / math.tan(self.perspective_angle/2)         

        pixels_per_meter = distance_pixels / distance_meters
         
        return velocity_pixels_per_second / pixels_per_meter

if __name__=="__main__":

    parser = optparse.OptionParser()

    parser.add_option("-f", "--file",  dest="filename", help="Read from video file", metavar="FILE")
    parser.add_option("-s", "--scaledown", dest="scaledown", help="Fractional image scaling", metavar="SCALEDOWN")
    parser.add_option("-c", "--camera", dest="camera", help="Camera number", metavar="CAMERA")
    parser.add_option("-m", "--movestep", dest="movestep", help="Move step (pixels)", metavar="MOVESTEP")

    (options, _) = parser.parse_args()

    camno = int(options.camera) if options.camera else 0

    cap = cv2.VideoCapture(camno if not options.filename else options.filename)

    width    = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    scaledown = int(options.scaledown) if options.scaledown else 1

    movestep = int(options.movestep) if options.movestep else 16

    flow = OpticalFlowCalculator(width, height, window_name='Optical Flow', scaledown=scaledown, move_step=movestep) 

    start_sec = time.time()
    count = 0
    while True:

        success, frame = cap.read()

        count += 1
            
        if not success:
            break

        result = flow.processFrame(frame)

        if not result:
            break

    elapsed_sec = time.time() - start_sec

    print('%dx%d image: %d frames in %3.3f sec = %3.3f frames / sec' % 
             (width/scaledown, height/scaledown, count, elapsed_sec, count/elapsed_sec))
