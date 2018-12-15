/*
   lkdemo.cpp - simple C++ demo of Lucas-Kanade optical flow

   Copyright (C) 2018 Simon D. Levy

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as 
   published by the Free Software Foundation, either version 3 of the 
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   Requires: OpenCV
 */

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

int main(int argc, char** argv)
{
    VideoCapture cap;

    // Use camera for capture
    if(!cap.open(0)) {
        return 0;
    }

    // Take first frame and find corners in it
    Mat old_frame;
    cap >> old_frame;
    Mat old_gray;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    //p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params);

    while (true) {

        // Capture a frame
        Mat frame;
        cap >> frame;

        // Convert it to grayscale
        Mat frame_gray;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // Display the frame with the flow lines
        imshow("frame)", frame_gray);

        // Force display, quitting on ESC
        if( waitKey(1) == 27 ) {
            break; 
        }
    }

    // Shut down
    destroyAllWindows();
    cap.release();
    return 0;
}
