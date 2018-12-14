#!/usr/bin/env python

'''
lkdemo.py - Lucas-Kanade optical flow demo

Adapted from https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

My version doesn't error-out on failure to find flow; instead, it 
searches for a new set of features to track.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
        qualityLevel = 0.3,
        minDistance = 7,
        blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
        maxLevel = 2,
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:

    ret,frame = cap.read()

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # If no flow, look for new points
    if p0 is None or p1 is None:

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)        

        cv.imshow('frame',frame)

    # Otherwise, show flow
    else:

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

        # Display the image with the flow lines
        img = cv.add(frame,mask)
        cv.imshow('frame',img)

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    # For display, quitting on ESC
    if (cv.waitKey(1) & 0xff) == 27:
        break
       

cv.destroyAllWindows()
cap.release()
