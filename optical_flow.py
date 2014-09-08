#!/usr/bin/env python

'''
optical_flow.py - Optical-flow velocity calculation and display using OpenCV

    To test:

      % optical_flow            # video from webcam
      % optical_flow FILENAME   # video from file

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

import cv
import sys
import time

class OpticalFlow:
    '''
    A class for optical flow using OpenCV
    '''
    
    def __init__(self, frame_width, frame_height, mv_step=16, window_name=None):
        '''
        Creates an OpticalFlow object for images with specified width and height.
        The mv_step parameter specifies the step size in pixels at which the flow
        will be sampled.  If window_name is specified, the video and flow will be
        displayed.
        '''

        self.mv_step = mv_step
        self.mv_color = (0, 255, 0)

        self.flow = None

        self.window_name = window_name
        
        if window_name:
            cv.NamedWindow(window_name, 1 )

        size = (frame_width, frame_height)

        self.bgrbytes = bytearray(frame_width*frame_height * 3)
        self.image = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 3)

        self.gray = cv.CreateImage(size, 8, 1)
        self.prev_gray = cv.CreateImage(size, 8, 1)
        self.flow = cv.CreateImage(size, 32, 2)

        self.frame_width = frame_width

    def processBytes(self, rgb_bytes):
        '''
        Processes one frame of RGB bytes, returning summed X,Y flow
        '''

        self.bgrbytes[0::3] = rgb_bytes[2::3]
        self.bgrbytes[1::3] = rgb_bytes[1::3]
        self.bgrbytes[2::3] = rgb_bytes[0::3]
                        
        cv.SetData(self.image, self.bgrbytes, self.frame_width*3)
        
        return self.processFrame(self.image)

    def processFrame(self, frame):
        '''
        Processes one image frame, returning summed X,Y flow
        '''

        cv.CvtColor(frame, self.gray, cv.CV_BGR2GRAY)
            
        cv.CalcOpticalFlowFarneback(self.prev_gray, self.gray, self.flow,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        xsum, ysum = 0,0
        
        for y in range(0, self.flow.height, self.mv_step):
            for x in range(0, self.flow.width, self.mv_step):
                fx, fy = self.flow[y, x]
                xsum += fx
                ysum += fy
                if self.window_name:
                    cv.Line(frame, (x,y), (int(x+fx),int(y+fy)), self.mv_color)
                    cv.Circle(frame, (x,y), 1, self.mv_color, -1)

        if self.window_name:
            thickness = 3
            ctrx = int(self.flow.width/2)
            ctry = int(self.flow.height/2)
            scale = 1. / self.mv_step
            xsum = int(xsum*scale)
            ysum = int(ysum*scale)
            cv.Line(frame, (ctrx, ctry), (ctrx+xsum, ctry), (0,0,255), thickness)
            cv.Line(frame, (ctrx, ctry), (ctrx, ctry+ysum), (255,0,0), thickness)
            cv.ShowImage(self.window_name, frame)
            if cv.WaitKey(1) == 27:
                return None

        self.prev_gray, self.gray = self.gray, self.prev_gray        

        return xsum, ysum

if __name__=="__main__":

    capture = cv.CaptureFromCAM(0) if len(sys.argv) < 2 else cv.CaptureFromFile(sys.argv[1])

    sensor = OpticalFlow(int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)), 
                          int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT)),
                          window_name='Optical Flow')

    start_sec = time.time()
    count = 0

    while True:

        frame = cv.QueryFrame(capture)

        count += 1
            
        try:
            if not sensor.processFrame(frame):
                break
        except:
            break

    elapsed_sec = time.time() - start_sec
    print('%d frames in %3.3f sec = %3.3f frames / sec' % (count, elapsed_sec, count/elapsed_sec))
